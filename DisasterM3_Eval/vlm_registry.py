"""
vlm_registry.py
================
Model loading and inference for the 5 Windows-target VLMs.

Supported models (all fit on RTX A4500 20GB VRAM):
  1. moondream2           (~3 GB)  - single-image only
  2. phi-3.5-vision       (~9 GB)  - multi-image via image_N placeholders
  3. kimi-vl-a3b          (~7 GB)  - multi-image via message content
  4. llava-1.5-7b         (~14 GB) - single-image only
  5. qwen2.5-vl-7b        (~16 GB) - multi-image via qwen_vl_utils
"""

import gc
import os
import torch
from PIL import Image

MULTI_IMAGE_MODELS = {"phi-3.5-vision", "kimi-vl-a3b", "qwen2.5-vl-7b"}

PHI_IMG_1 = "<" + "|image_1|" + ">"
PHI_IMG_2 = "<" + "|image_2|" + ">"
PHI_USER  = "<" + "|user|" + ">"
PHI_END   = "<" + "|end|" + ">"
PHI_ASST  = "<" + "|assistant|" + ">"


def concat_images_side_by_side(path1, path2, max_height=512):
    """Concatenate two images horizontally for single-image VLMs."""
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    h = min(img1.height, img2.height, max_height)
    img1 = img1.resize((int(img1.width * h / img1.height), h))
    img2 = img2.resize((int(img2.width * h / img2.height), h))
    combined = Image.new("RGB", (img1.width + img2.width, h))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined


# ── Model Loading ────────────────────────────────────────────

def load_vlm(model_key, hf_id):
    """Load a VLM model and its processor/tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    import transformers

    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        transformers.PreTrainedModel.all_tied_weights_keys = {}

    # Monkey-patch DynamicCache for Phi-3.5-Vision compatibility.
    # get_max_length() was removed in transformers 4.45+; the original returned None.
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    token = os.environ.get("HF_TOKEN", None)
    print(f"  Loading {hf_id} ...")
    if token:
        print(f"  Using HF_TOKEN: {token[:8]}...")

    if model_key == "moondream2":
        # Pin to 2025-06-21 release — the latest stable revision.
        # Older cached revisions produce garbage with transformers>=5.x.
        MOONDREAM_REV = "2025-06-21"
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, token=token,
            revision=MOONDREAM_REV,
            device_map={"": "cuda"},
        )
        # New Moondream2 API (2025+): model.query(image, question)
        # No tokenizer/processor needed.
        return model, None

    if model_key == "phi-3.5-vision":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.float16,
            device_map="cuda", _attn_implementation="eager", token=token,
        )
        proc = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True, num_crops=4, token=token
        )
        return model, proc

    if model_key == "kimi-vl-a3b":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", token=token,
        )
        proc = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True, token=token
        )
        return model, proc

    if model_key == "llava-1.5-7b":
        from transformers import LlavaForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    if model_key == "qwen2.5-vl-7b":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype="auto", device_map="auto", token=token
        )
        proc = AutoProcessor.from_pretrained(hf_id, token=token)
        return model, proc

    raise ValueError(f"Unknown model key: {model_key}")


# ── Inference ────────────────────────────────────────────────

CONCAT_NOTE = (
    "The image shows the pre-disaster scene on the left "
    "and the post-disaster scene on the right.\n\n"
)


def ask_vlm(model, proc, prompt_text, image_paths, needs_dual_image,
            model_key, max_new_tokens=256):
    """Run VLM inference with proper multi/single image handling."""

    if model_key == "moondream2":
        return _infer_moondream(model, proc, prompt_text, image_paths, needs_dual_image)

    if model_key == "phi-3.5-vision":
        return _infer_phi35(model, proc, prompt_text, image_paths, needs_dual_image, max_new_tokens)

    if model_key == "kimi-vl-a3b":
        return _infer_kimi(model, proc, prompt_text, image_paths, needs_dual_image, max_new_tokens)

    if model_key == "llava-1.5-7b":
        return _infer_llava15(model, proc, prompt_text, image_paths, needs_dual_image, max_new_tokens)

    if model_key == "qwen2.5-vl-7b":
        return _infer_qwen25(model, proc, prompt_text, image_paths, needs_dual_image, max_new_tokens)

    raise ValueError(f"Unknown model_key: {model_key}")


def _infer_moondream(model, proc, prompt_text, image_paths, needs_dual):
    if needs_dual and len(image_paths) == 2:
        img = concat_images_side_by_side(image_paths[0], image_paths[1])
        prompt_text = CONCAT_NOTE + prompt_text
    else:
        img = Image.open(image_paths[0]).convert("RGB")
    # New Moondream2 API (2025+): model.query(image, question)
    # Use temperature=0 (greedy) for deterministic, benchmark-quality output.
    # Default (temp=0.5, top_p=0.3) produces garbage on structured prompts.
    result = model.query(img, prompt_text, settings={
        "temperature": 0.0,
        "max_tokens": 512,
        "variant": None,
    })
    return result["answer"].strip()


def _infer_phi35(model, proc, prompt_text, image_paths, needs_dual, max_tokens):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if needs_dual and len(images) == 2:
        img_section = "pre-disaster image:\n" + PHI_IMG_1 + "\n\npost-disaster image:\n" + PHI_IMG_2 + "\n\n"
    else:
        img_section = PHI_IMG_1 + "\n"

    full_prompt = PHI_USER + "\n" + img_section + prompt_text + PHI_END + "\n" + PHI_ASST + "\n"
    inputs = proc(full_prompt, images, return_tensors="pt").to("cuda")
    out_ids = model.generate(
        **inputs, max_new_tokens=max_tokens,
        eos_token_id=proc.tokenizer.eos_token_id
    )
    out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def _infer_kimi(model, proc, prompt_text, image_paths, needs_dual, max_tokens):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = []
    if needs_dual and len(images) == 2:
        content.append({"type": "image"})
        content.append({"type": "text", "text": "Above: pre-disaster image."})
        content.append({"type": "image"})
        content.append({"type": "text", "text": "Above: post-disaster image."})
    else:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text, images=images, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    out_ids = out_ids[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def _infer_llava15(model, proc, prompt_text, image_paths, needs_dual, max_tokens):
    if needs_dual and len(image_paths) == 2:
        img = concat_images_side_by_side(image_paths[0], image_paths[1])
        prompt_text = CONCAT_NOTE + prompt_text
    else:
        img = Image.open(image_paths[0]).convert("RGB")
    llava_img_token = "<" + "image" + ">"
    prompt = "USER: " + llava_img_token + "\n" + prompt_text + "\nASSISTANT:"
    inputs = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens)
    txt = proc.decode(out[0], skip_special_tokens=True)
    return txt.split("ASSISTANT:")[-1].strip()


def _infer_qwen25(model, proc, prompt_text, image_paths, needs_dual, max_tokens):
    from qwen_vl_utils import process_vision_info
    content = []
    if needs_dual and len(image_paths) == 2:
        content.append({"type": "text", "text": "pre-disaster image:"})
        content.append({"type": "image", "image": "file:///" + image_paths[0].replace("\\", "/")})
        content.append({"type": "text", "text": "post-disaster image:"})
        content.append({"type": "image", "image": "file:///" + image_paths[1].replace("\\", "/")})
    else:
        content.append({"type": "image", "image": "file:///" + image_paths[0].replace("\\", "/")})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")
    out_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ── VRAM Cleanup ─────────────────────────────────────────────

def unload_model(model, proc):
    """Aggressively free VRAM after a model run."""
    del model, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  VRAM cleared.")
