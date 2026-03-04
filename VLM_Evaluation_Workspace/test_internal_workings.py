import sys, gc, torch, os
sys.path.append("/home/aipmu/Datasets for VLM/VLM_Evaluation_Workspace")

# We recycle the powerful robust pipeline we just patched!
from run_rsvlmqa import load_model, ask_vlm, MODELS
from datasets_loader import load_rsvlmqa_data, load_disasterm3_data

def run_test():
    print("="*70)
    print(" VLM INTERNAL WORKING VIEWER ".center(70))
    print("="*70)
    
    print("[*] Loading exactly 1 sample from each dataset...\n")
    try:
        rsvlmqa_item = load_rsvlmqa_data(max_samples=1)[0]
        disaster_item = load_disasterm3_data(max_samples=1)[0]
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    datasets_mapping = {
        "RSVLM-QA": rsvlmqa_item,
        "DisasterM3": disaster_item
    }

    for short_name, hf_id in MODELS.items():
        print("\n" + "="*70)
        print(f"  ► MODEL: {short_name}  ({hf_id})")
        print("="*70)
        
        try:
            model, proc = load_model(hf_id)
            
            for ds_name, item in datasets_mapping.items():
                print(f"\n  --- Dataset: {ds_name} ---")
                img_path = item["image_path"]
                question = item["question"]
                gt_answer = item["answer"]
                
                print(f"  [Image File]   : {os.path.basename(img_path)}")
                print(f"  [Question]     : {question}")
                print(f"  [Ground Truth] : {gt_answer}")
                print(f"  [VLM Thinking...]")
                
                # Generation Call
                pred = ask_vlm(model, proc, img_path, question, hf_id)
                
                print(f"  [VLM Answer]   : {pred}")
                
        except Exception as e:
            print(f"  [!] CRASH evaluating {short_name}: {e}")
            
        finally:
            print(f"\n  [*] Unloading {short_name} to free GPU Memory...")
            if 'model' in locals(): del model
            if 'proc' in locals(): del proc
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(" DONE! ".center(70))
    print("="*70)

if __name__ == "__main__":
    run_test()
