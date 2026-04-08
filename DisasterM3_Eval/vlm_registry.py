"""
vlm_registry.py
================
Model loading and inference for all 8 VLMs.
Optimized for single-GPU execution with explicit VRAM management.

Supported models:
  1. moondream2           (~2 GB VRAM)
  2. blip2-opt-2.7b       (~6 GB VRAM)
  3. llava-1.5-7b         (~14 GB VRAM)
  4. qwen-vl-chat         (~14 GB VRAM)
  5. instructblip-vicuna   (~14 GB VRAM)
  6. idefics2-8b          (~16 GB VRAM)
  7. internvl2-4b         (~8 GB VRAM)
  8. llava-next-llama3    (~16 GB VRAM)
"""

import os
import gc
import torch
from PIL import Image


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_vlm(hf_id: str):
    """
    Load a VLM model and its processor/tokenizer.
    Returns (model, processor_or_tokenizer).
    """
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    import transformers

    # Patch for moondream2 + transformers compat
    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        transformers.PreTrainedModel.all_tied_weights_keys = {}

    token = os.environ.get("HF_TOKEN", None)
    print(f"  ⏳ Loading {hf_id} ...")
    if token:
        print(f"  🔑 Using HF_TOKEN: {token[:8]}...")

    # ── moondream2 ──
    if "moondream" in hf_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True,
            torch_dtype=torch.float16, revision="2024-08-26",
            token=token
        ).to("cuda")
        tok = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, revision="2024-08-26", token=token
        )
        return model, tok

    # ── InstructBLIP ──
    if "instructblip" in hf_id.lower():
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        proc = InstructBlipProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    # ── BLIP-2 ──
    if "blip2" in hf_id.lower():
        from transformers import Blip2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = Blip2ForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    # ── Idefics2 ──
    if "idefics2" in hf_id.lower():
        from transformers import Idefics2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = Idefics2ForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    # ── LLaVA-NeXT (must be checked BEFORE generic "llava") ──
    if "llava-next" in hf_id.lower() or "llama3-llava-next" in hf_id.lower():
        from transformers import LlavaNextForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    # ── LLaVA 1.5 ──
    if "llava" in hf_id.lower():
        from transformers import LlavaForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False, token=token)
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16, token=token
        )
        return model, proc

    # ── Qwen-VL-Chat ──
    if "qwen-vl" in hf_id.lower():
        proc = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="cuda", torch_dtype=torch.float16,
            trust_remote_code=True, token=token
        )
        return model, proc

    # ── InternVL2 ──
    if "internvl" in hf_id.lower():
        proc = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, use_fast=False, token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True, token=token
        )
        return model, proc

    # ── Fallback (generic) ──
    proc = AutoProcessor.from_pretrained(
        hf_id, trust_remote_code=True, use_fast=False, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True, token=token
    )
    return model, proc


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def ask_vlm(model, proc, img_path: str, question: str, hf_id: str) -> str:
    """
    Run a single VLM inference: image + question → answer string.
    """
    img = Image.open(img_path).convert("RGB")

    # ── Qwen-VL-Chat ──
    if "qwen-vl" in hf_id.lower():
        query = proc.from_list_format([
            {'image': img_path},
            {'text': question},
        ])
        response, _ = model.chat(proc, query=query, history=None)
        return response

    # ── moondream2 ──
    if "moondream" in hf_id.lower():
        enc = model.encode_image(img)
        return model.answer_question(enc, question, proc)

    # ── Idefics2 ──
    if "idefics2" in hf_id.lower():
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        decoded = proc.decode(out[0], skip_special_tokens=True)
        return decoded.split("Assistant:")[-1].strip()

    # ── InternVL2 ──
    if "internvl" in hf_id.lower():
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Lambda(lambda i: i.convert('RGB') if i.mode != 'RGB' else i),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        pixel_values = transform(img).unsqueeze(0).to(model.device).to(torch.bfloat16)
        if hasattr(model, 'chat'):
            return model.chat(proc, pixel_values, question, dict(max_new_tokens=128))
        else:
            # Fallback if .chat() is not available
            return "[InternVL2 model.chat() not available]"

    # ── LLaVA-NeXT (must come BEFORE generic llava check) ──
    if "llava-next" in hf_id.lower() or "llama3-llava-next" in hf_id.lower():
        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        txt = proc.decode(out[0], skip_special_tokens=True)
        # Extract only the assistant response
        if "assistant" in txt.lower():
            return txt.split("assistant")[-1].strip()
        return txt.strip()

    # ── LLaVA 1.5 ──
    if "llava" in hf_id.lower():
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        txt = proc.decode(out[0], skip_special_tokens=True)
        return txt.split("ASSISTANT:")[-1].strip()

    # ── BLIP-2 ──
    if "blip2" in hf_id.lower():
        prompt = f"Question: {question} Answer:"
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    # ── InstructBLIP ──
    if "instructblip" in hf_id.lower():
        inputs = proc(images=img, text=question, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        ans = proc.decode(out[0], skip_special_tokens=True)
        return ans.replace("<s>", "").replace("</s>", "").strip()

    # ── Fallback generic ──
    inputs = proc(images=img, text=question, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return proc.decode(out[0], skip_special_tokens=True)


# ──────────────────────────────────────────────
# VRAM Cleanup
# ──────────────────────────────────────────────

def unload_model(model, proc):
    """Aggressively free VRAM after a model run."""
    del model, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  🧹 VRAM cleared.")
