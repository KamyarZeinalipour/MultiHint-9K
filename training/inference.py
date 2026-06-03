import gc
import os
import time
import torch
import pandas as pd
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# !!! SET THIS TO TRUE TO TEST FIRST !!!
# If True, processes only 1 batch per model to verify memory/settings.
# If False, processes the entire dataset.
DRY_RUN = False

# Format: "Model Name": { "num_gpus": int, "batch_size": int }
MODEL_CONFIGS = {
    "mistralai/Mistral-Small-24B-Instruct-2501":      {"num_gpus": 2, "batch_size": 8},
    "Qwen/Qwen2.5-14B-Instruct":        {"num_gpus": 1, "batch_size": 4},
   "mistralai/Mistral-Nemo-Instruct-2407":        {"num_gpus": 1, "batch_size": 4},
    "meta-llama/Llama-3.1-8B-Instruct":        {"num_gpus": 1, "batch_size": 8},
     "mistralai/Mistral-7B-Instruct-v0.3":        {"num_gpus": 1, "batch_size": 8}
    
    # Example: A 70B model usually needs 2x A100 (80GB) or 4x A10G (24GB)
    # "meta-llama/Llama-3.1-70B-Instruct":     {"num_gpus": 2, "batch_size": 4}, 
}

INFERENCE_DIR   = "inference_recall"
DATASET_NAME    = "Kamyar-zeinalipour/recall-assistant-dataset"

# Generation hyperparameters
GEN_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# -----------------------------------------------------------------------------
# 1) Data Preparation
# -----------------------------------------------------------------------------
def prepare_dataset():
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        raw_ds = load_dataset(DATASET_NAME, split="test")
    except:
        print("Test split not found, using 'train' split...")
        #raw_ds = load_dataset(DATASET_NAME, split="train")
    
    df = pd.DataFrame(raw_ds)
    
    # Helper to extract conversation parts
    def process_row(messages):
        try:
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
            assistant_msg = "\n".join(assistant_msgs)
            return user_msg, assistant_msg
        except Exception:
            return "", ""

    if "messages" in df.columns:
        df[["User", "Assistant"]] = df["messages"].apply(
            lambda x: pd.Series(process_row(x))
        )
    else:
        df["User"] = ""
        df["Assistant"] = ""
        
    print(f"Dataset Loaded: {len(df)} rows.")
    return df

# -----------------------------------------------------------------------------
# 2) Worker Function
# -----------------------------------------------------------------------------
def model_runner(model_name, gpu_ids, config, dataframe):
    """
    Runs a specific model on a specific set of GPUs.
    """
    # 1. Isolate GPUs for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    pid = os.getpid()
    batch_size = config['batch_size']
    
    print(f"\n[PID {pid}] STARTING {model_name}")
    print(f"[PID {pid}] Assigned GPUs: {gpu_ids} (Logical: cuda:0...cuda:{len(gpu_ids)-1})")
    
    try:
        # 2. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Critical for generation

        # 3. Load Model
        print(f"[PID {pid}] Loading weights into VRAM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically spreads layers across visible GPUs
            low_cpu_mem_usage=True,
        )
        model.eval()

        # 4. Prepare Prompts (Templating)
        current_df = dataframe.copy()
        
        # If DRY RUN, slice the dataframe to just one batch
        if DRY_RUN:
            print(f"[PID {pid}] --- DRY RUN MODE: Processing only {batch_size} samples ---")
            current_df = current_df.head(batch_size)
        
        def apply_template(row):
            if "messages" in row and isinstance(row["messages"], list):
                msgs = row["messages"][:-1] if len(row["messages"]) > 1 else row["messages"]
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return ""

        current_df["Prompt"] = current_df.apply(apply_template, axis=1)
        prompts = current_df["Prompt"].tolist()
        
        # 5. Inference Loop
        outputs = []
        total = len(prompts)
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(model.device)

            # Generate
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    **GEN_CONFIG,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            input_len = inputs.input_ids.shape[1]
            generated_tokens = out_ids[:, input_len:] 
            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            outputs.extend(decoded_batch)
            
            if (i // batch_size) % 1 == 0:
                print(f"[PID {pid}] {model_name}: Batch {i//batch_size + 1} done.")

        elapsed = time.time() - start_time
        print(f"[PID {pid}] Finished {model_name} in {elapsed:.2f}s")

        # 6. Save Results
        current_df["Output"] = outputs
        
        # Append _DRY_RUN to filename if needed so we don't overwrite real results
        suffix = "_recall_DRY_RUN.csv" if DRY_RUN else "_recall.csv"
        safe_name = model_name.replace("/", "_")
        out_path = os.path.join(INFERENCE_DIR, f"{safe_name}{suffix}")
        
        current_df.to_csv(out_path, index=False)
        print(f"[PID {pid}] Results saved to {out_path}")

        # 7. Cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[PID {pid}] !!! ERROR on {model_name}: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 3) Resource Manager
# -----------------------------------------------------------------------------
def resource_manager(model_queue, gpu_queue, master_df):
    running_procs = []

    while not model_queue.empty() or len(running_procs) > 0:
        # Clean up finished processes
        running_procs = [p for p in running_procs if p.is_alive()]
        
        if not model_queue.empty():
            model_name = model_queue.get()
            config = MODEL_CONFIGS.get(model_name, {"num_gpus": 1, "batch_size": 4})
            required_gpus = config["num_gpus"]
            
            # Try to acquire GPUs
            acquired_gpus = []
            try:
                # Try to get 'required_gpus' number of IDs from the pool
                for _ in range(required_gpus):
                    acquired_gpus.append(gpu_queue.get(block=False))
                
                # If we got here, we have enough GPUs. Start worker.
                p = mp.Process(
                    target=model_runner_wrapper, 
                    args=(model_name, acquired_gpus, config, master_df, gpu_queue)
                )
                p.start()
                running_procs.append(p)
                
            except:
                # Not enough GPUs available right now. 
                # Return whatever we grabbed.
                for g in acquired_gpus:
                    gpu_queue.put(g)
                # Put model back in queue
                model_queue.put(model_name)
                # Wait a bit before retrying to avoid CPU spin
                time.sleep(2) 
        else:
            time.sleep(1)

def model_runner_wrapper(model_name, gpu_ids, config, df, gpu_queue):
    try:
        model_runner(model_name, gpu_ids, config, df)
    finally:
        # ALWAYS return GPUs to pool, even if model crashes
        print(f"[Manager] Releasing GPUs {gpu_ids}")
        for gid in gpu_ids:
            gpu_queue.put(gid)

# -----------------------------------------------------------------------------
# 4) Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    os.makedirs(INFERENCE_DIR, exist_ok=True)
    
    num_system_gpus = torch.cuda.device_count()
    print(f"Detected {num_system_gpus} GPUs.")
    if num_system_gpus == 0: raise RuntimeError("No CUDA GPUs found.")

    if DRY_RUN:
        print("\n" + "="*50)
        print("   DRY RUN MODE ENABLED")
        print("   Will process 1 batch per model to verify config.")
        print("="*50 + "\n")

    # 1. Load Data
    master_df = prepare_dataset()

    # 2. Setup Queues
    model_queue = mp.Queue()
    for m_name in MODEL_CONFIGS.keys():
        model_queue.put(m_name)
    
    gpu_queue = mp.Queue()
    for i in range(num_system_gpus):
        gpu_queue.put(i)

    # 3. Start Manager
    resource_manager(model_queue, gpu_queue, master_df)
    
    print("\nAll processing complete.")