import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import transformers

# Patch for moondream
if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
    transformers.PreTrainedModel.all_tied_weights_keys = {}

img = Image.new('RGB', (224, 224), color='red')
question = "What is this?"

def test_instructblip():
    print("Testing InstructBLIP...")
    from transformers import InstructBlipForConditionalGeneration
    try:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        proc = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="cuda", torch_dtype=torch.float16)
        inputs = proc(images=img, text=question, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=10)
        print("InstructBLIP OK ->", proc.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print("InstructBLIP FAILED:", e)

def test_llava():
    print("Testing LLaVA...")
    from transformers import LlavaForConditionalGeneration
    try:
        proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="cuda", torch_dtype=torch.float16)
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=10)
        print("LLaVA OK ->", proc.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print("LLaVA FAILED:", e)

def test_idefics2():
    print("Testing IDEFICS2...")
    from transformers import Idefics2ForConditionalGeneration
    try:
        proc = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", use_fast=False)
        model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", device_map="cuda", torch_dtype=torch.float16)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=10)
        print("IDEFICS2 OK ->", proc.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print("IDEFICS2 FAILED:", e)

def test_internvl2():
    print("Testing InternVL2...")
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        proc = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained("OpenGVLab/InternVL2-4B", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        pixel_values = build_transform(448)(img).unsqueeze(0).to(model.device).to(torch.bfloat16)
        response = model.chat(proc, pixel_values, question, dict(max_new_tokens=10))
        print("InternVL2 OK ->", response)
    except Exception as e:
        print("InternVL2 FAILED:", e)

if __name__ == "__main__":
    test_instructblip()
    test_llava()
    test_idefics2()
    test_internvl2()
