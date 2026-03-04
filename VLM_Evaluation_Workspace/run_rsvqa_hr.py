#!/usr/bin/env python3
"""
run_rsvqa_hr.py
===============
Evaluate pretrained VLMs on the RSVQA-HR (High Resolution) VQA dataset.

Usage examples
--------------
  # Quick sanity check
  python run_rsvqa_hr.py --model moondream2 --max_samples 10

  # Full evaluation
  python run_rsvqa_hr.py --model llava-1.5-7b

  # Results are saved as JSON in the results/ directory.
"""

import argparse, json, os, time, gc, sys
import torch
from PIL import Image
from datasets_loader import load_rsvqa_hr_data
from evaluation import evaluate_predictions, evaluate_by_type

# ── Model registry ──────────────────────────
MODELS = {
    "llava-1.5-7b":        "llava-hf/llava-1.5-7b-hf",
    "qwen-vl-chat":        "Qwen/Qwen-VL-Chat",
    "instructblip-vicuna":  "Salesforce/instructblip-vicuna-7b",
    "blip2-opt-2.7b":      "Salesforce/blip2-opt-2.7b",
    "idefics2-8b":         "HuggingFaceM4/idefics2-8b",
    "moondream2":           "vikhyatk/moondream2",
    "internvl2-4b":        "OpenGVLab/InternVL2-4B",
    "llava-next-llama3":   "llava-hf/llama3-llava-next-8b-hf",
}


def load_model(hf_id: str):
    """Compatible with transformers v5.x (AutoModelForVision2Seq removed)."""
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    import transformers
    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        transformers.PreTrainedModel.all_tied_weights_keys = {}
    print(f"  ⏳ Downloading / loading  {hf_id}  …")

    if "moondream" in hf_id.lower():
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True,
            torch_dtype=torch.float16, revision="2024-08-26").to("cuda")
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, revision="2024-08-26")
        return model, tok

    if "instructblip" in hf_id.lower():
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        proc = InstructBlipProcessor.from_pretrained(hf_id, use_fast=False)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16)
        return model, proc

    if "blip2" in hf_id.lower():
        from transformers import Blip2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False)
        model = Blip2ForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16)
        return model, proc

    if "idefics2" in hf_id.lower():
        from transformers import Idefics2ForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False)
        model = Idefics2ForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16)
        return model, proc

    if "llava-next" in hf_id.lower():
        from transformers import LlavaNextForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16)
        return model, proc

    if "llava" in hf_id.lower():
        from transformers import LlavaForConditionalGeneration
        proc = AutoProcessor.from_pretrained(hf_id, use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch.float16)
        return model, proc

    if "qwen-vl" in hf_id.lower():
        from transformers import AutoTokenizer
        proc = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
        return model, proc

    proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    return model, proc


def ask_vlm(model, proc, img_path: str, question: str, hf_id: str) -> str:
    img = Image.open(img_path).convert("RGB")

    if "qwen-vl" in hf_id.lower():
        query = proc.from_list_format([
            {'image': img_path},
            {'text': question},
        ])
        response, _ = model.chat(proc, query=query, history=None)
        return response

    if "moondream" in hf_id.lower():
        enc = model.encode_image(img)
        return model.answer_question(enc, question, proc)

    if "idefics2" in hf_id.lower():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        return proc.decode(out[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    if "internvl" in hf_id.lower():
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Lambda(lambda i: i.convert('RGB') if i.mode != 'RGB' else i),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        pixel_values = transform(img).unsqueeze(0).to(model.device).to(torch.float16)
        return model.chat(proc, pixel_values, question, dict(max_new_tokens=128))

    if "llava" in hf_id.lower():
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        txt = proc.decode(out[0], skip_special_tokens=True)
        return txt.split("ASSISTANT:")[-1].strip()

    if "blip2" in hf_id.lower():
        prompt = f"Question: {question} Answer:"
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    if "instructblip" in hf_id.lower():
        inputs = proc(images=img, text=question, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        ans = proc.decode(out[0], skip_special_tokens=True)
        return ans.replace("<s>", "").replace("</s>", "").strip()

    # Fallback generic path
    inputs = proc(images=img, text=question, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    return proc.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    hf_id = MODELS[args.model]

    data = load_rsvqa_hr_data(max_samples=args.max_samples)
    if not data:
        print("❌ No data loaded – check paths."); sys.exit(1)

    model, proc = load_model(hf_id)

    results = []
    t0 = time.time()
    for i, item in enumerate(data):
        pred = ask_vlm(model, proc, item["image_path"], item["question"], hf_id)
        results.append({
            "question_id":   item["question_id"],
            "image_id":      item["image_id"],
            "question":      item["question"],
            "question_type": item["question_type"],
            "ground_truth":  item["answer"],
            "prediction":    pred,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(data)}]  {elapsed/60:.1f} min elapsed")

    elapsed = time.time() - t0
    print(f"\n✅ Inference done – {len(results)} samples in {elapsed/60:.1f} min")

    overall = evaluate_predictions(results)
    by_type = evaluate_by_type(results)

    print("\n" + "=" * 50)
    print(f"RSVQA-HR  ▸  {args.model}")
    print("=" * 50)
    for k, v in overall.items():
        print(f"  {k:15s}: {v:.4f}")
    print("\nPer question-type breakdown:")
    for qt, m in by_type.items():
        line_parts = [f"EM={m.get('exact_match',0):.3f}",
                      f"BLEU1={m.get('bleu1',0):.3f}",
                      f"ROUGE-L={m.get('rougeL',0):.3f}",
                      f"METEOR={m.get('meteor',0):.3f}",
                      f"F1={m.get('token_f1',0):.3f}"]
        print(f"  {qt:20s}  {'  '.join(line_parts)}")

    out_file = os.path.join(args.results_dir, f"rsvqa_hr_{args.model}.json")
    with open(out_file, "w") as f:
        json.dump({"model": args.model, "dataset": "RSVQA-HR",
                   "overall_metrics": overall, "per_type_metrics": by_type,
                   "predictions": results}, f, indent=2)
    print(f"\n💾 Saved → {out_file}")

    del model, proc; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
