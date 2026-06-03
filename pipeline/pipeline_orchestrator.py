import os
import time
import json
import logging
import pandas as pd
import re
import threading  # ### NEW: For thread-safe stopping
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm # ### NEW: For progress bars

# ------------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# ------------------------------------------------------------------------------
try:
    from models_apis import deepseek_chat, deepseek_reasoner, gpt4o, gpto4_mini, mistral, gemini
except ImportError:
    print("CRITICAL: 'models_apis.py' not found. Please ensure it exists.")
    exit(1)

from prompts_config import SYSTEM_PROMPTS

# ------------------------------------------------------------------------------
# 2. CONFIGURATION (USER CONTROL PANEL)
# ------------------------------------------------------------------------------

DRY_RUN = False 
OUTPUT_DIR = "output"

MAX_REFINE_LOOPS = 2 
MAX_PARSE_RETRIES = 3
MAX_RECOVERY_LOOPS = 2  

# ### NEW: Global Event to stop all threads if Rate Limit is hit
STOP_PIPELINE = threading.Event()

MODEL_CONFIG = {
    "English": {
        "generator":       "gemini",
        "qa_critic":       "mistral",
        "qa_refiner":      "gemini",
        "hint_architect": "gpt4o",
        "hint_critic":     "gemini",
        "hint_refiner":    "gpt4o",
        "qa_hint_solver": "gemini" 
    },
    "Italian": {
        "generator":       "mistral",
        "qa_critic":       "deepseek_chat",
        "qa_refiner":      "mistral",
        "hint_architect": "deepseek_chat",
        "hint_critic":     "gpt4o",
        "hint_refiner":    "deepseek_chat",
        "qa_hint_solver": "mistral"
    },
    "Farsi": {
        "generator":       "gemini",
        "qa_critic":       "gemini",
        "qa_refiner":      "gemini",
        "hint_architect": "deepseek_chat" ,
        "hint_critic":     "deepseek_chat" ,
        "hint_refiner":    "deepseek_chat" ,
        "qa_hint_solver": "gemini"
    }
}

MODEL_MAP = {
    "gpt4o": gpt4o,
    "mini": gpto4_mini,
    "deepseek_chat": deepseek_chat,
    "deepseek_reasoner": deepseek_reasoner,
    "gemini": gemini,
    "mistral": mistral
}

DATA_FILES = [
  #  {"language": "English", "input": "df_en_part1.csv", "output": "final_dataset_en.csv"},
  #  {"language": "Italian", "input": "df_ita_no_dup.csv", "output": "final_dataset_it.csv"},
    {"language": "Farsi",   "input": "df_fa_part5.csv", "output": "final_dataset_fa.csv"},
]

# ------------------------------------------------------------------------------
# 3. LOGGING & HELPERS
# ------------------------------------------------------------------------------

def setup_folders():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

setup_folders()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(get_path("pipeline.log"), encoding='utf-8'),
        # logging.StreamHandler() # ### CHANGED: Commented out to keep progress bars clean
    ]
)

def log_dry_run(language: str, role: str, prompt: str, raw_output: str, parsed_output: Any):
    lang_log = get_path(f"debug_{language}.txt")
    with open(lang_log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nROLE: {role} | TIME: {time.ctime()}\n{'='*60}\n")
        f.write(f"--- PROMPT SENT ---\n{prompt}\n")
        f.write(f"\n--- RAW OUTPUT ---\n{raw_output}\n")
        f.write(f"\n--- PARSED JSON ---\n{json.dumps(parsed_output, ensure_ascii=False, indent=2)}\n")
        f.write("\n" + "-"*60 + "\n")

def parse_json_robust(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return None

def call_agent(language: str, role: str, prompt_template: str, user_input: str) -> Tuple[Optional[Dict], str]:
    # ### NEW: Check global stop signal immediately
    if STOP_PIPELINE.is_set():
        return None, "PIPELINE_STOPPED_DUE_TO_RATE_LIMIT"

    model_key = MODEL_CONFIG[language][role]
    model_func = MODEL_MAP.get(model_key)
    if not model_func: return None, ""

    last_raw = ""
    for attempt in range(MAX_PARSE_RETRIES):
        # ### NEW: Double check signal inside retry loop
        if STOP_PIPELINE.is_set(): break

        try:
            current_input = user_input
            if attempt > 0:
                current_input += "\n\n(SYSTEM NOTE: Output was not valid JSON. STRICT JSON only.)"

            raw_response = model_func(current_input, prompt_template)
            last_raw = raw_response
            parsed_json = parse_json_robust(raw_response)

            if parsed_json:
                log_dry_run(language, role, f"{prompt_template}\n--INPUT--\n{current_input}", raw_response, parsed_json)
                return parsed_json, raw_response
            else:
                log_dry_run(language, f"{role}_FAILED_PARSE", current_input, raw_response, {"error": "Invalid JSON"})
        
        except Exception as e:
            # ### NEW: RATE LIMIT HANDLING LOGIC
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg or "too many requests" in error_msg:
                print(f"\n\n[CRITICAL] RATE LIMIT HIT in {language} ({role}). STOPPING ALL THREADS.\nError: {e}")
                logging.critical(f"[{language}] RATE LIMIT HIT: {e}")
                STOP_PIPELINE.set() # Triggers all loops to break
                return None, "RATE_LIMIT_STOP"
            
            logging.error(f"[{language}] {role} API Error: {e}")
            time.sleep(2)
            
    return None, last_raw

# ------------------------------------------------------------------------------
# 4. PIPELINES (Unchanged logic, just inheriting the STOP check from call_agent)
# ------------------------------------------------------------------------------

def run_qa_pipeline(language: str, text: str, initial_qa: Dict = None) -> Dict:
    prompts = SYSTEM_PROMPTS[language]
    history = {"attempts": [], "raw_responses": [], "reasons": [], "error_classes": [], "accepted_index": -1, "final_data": None}
    
    if initial_qa:
        current_qa = initial_qa
        history["attempts"].append(current_qa)
        history["raw_responses"].append("Generated by Recovery Agent")
    else:
        qa_data, raw_gen = call_agent(language, "generator", prompts["generator"], f"Text:\n{text}")
        if not qa_data: return history
        history["attempts"].append(qa_data)
        history["raw_responses"].append(raw_gen)
        current_qa = qa_data

    for i in range(MAX_REFINE_LOOPS + 1):
        if STOP_PIPELINE.is_set(): break # ### NEW: Quick Exit

        gen_q, gen_a = current_qa.get("question", "N/A"), current_qa.get("answer", "N/A")
        critic_input = f"Text:\n{text}\n\nGenerated Question:\n{gen_q}\nGenerated Answer:\n{gen_a}"
        
        critic_response, _ = call_agent(language, "qa_critic", prompts["qa_critic"], critic_input)
        
        reason = critic_response.get("reason", critic_response.get("Reasoning", "N/A")) if critic_response else "Failed"
        err_cls = critic_response.get("Error_Class", "N/A") if critic_response else "System_Fail"
        
        history["reasons"].append(reason)
        history["error_classes"].append(err_cls)

        if critic_response and ("PASS" in str(critic_response.get("status", "")).upper() or "A" in str(critic_response.get("Rating", "")).upper()):
            history["accepted_index"] = len(history["attempts"])
            history["final_data"] = current_qa
            return history
            
        if i == MAX_REFINE_LOOPS: break

        refiner_input = (f"Original Text:\n{text}\n\nFAILED Question:\n{gen_q}\nFAILED Answer:\n{gen_a}\n\n"
                         f"### JUDGE'S REASONING: {reason}\n"
                         f"### ERROR TYPE: {err_cls}")
                         
        refined_qa, raw_ref = call_agent(language, "qa_refiner", prompts["qa_refiner"], refiner_input)
        if refined_qa:
            current_qa = refined_qa
            history["attempts"].append(current_qa)
            history["raw_responses"].append(raw_ref)
        else: break 

    history["final_data"] = current_qa
    return history

def run_hint_pipeline(language: str, text: str, qa_data: Dict) -> Dict:
    prompts = SYSTEM_PROMPTS[language]
    qa_str = f"Q: {qa_data.get('question')}\nA: {qa_data.get('answer')}"
    history = {"attempts": [], "raw_responses": [], "reasons": [], "error_classes": [], "accepted_index": -1, "final_data": None}

    hint_data, raw_arch = call_agent(language, "hint_architect", prompts["hint_architect"], f"Text:\n{text}\n\n{qa_str}")
    if not hint_data: return history
    history["attempts"].append(hint_data)
    history["raw_responses"].append(raw_arch)
    current_hint = hint_data

    for i in range(MAX_REFINE_LOOPS + 1):
        if STOP_PIPELINE.is_set(): break # ### NEW: Quick Exit

        cur_h = current_hint.get("hint", current_hint.get("Hint", "N/A"))
        critic_input = f"Source Passage:\n{text}\n\nQuestion:\n{qa_data.get('question')}\nAnswer:\n{qa_data.get('answer')}\n\nHint:\n{cur_h}"
        
        critic_response, _ = call_agent(language, "hint_critic", prompts["hint_critic"], critic_input)
        
        reason = critic_response.get("reason", critic_response.get("Reasoning", "N/A")) if critic_response else "Failed"
        err_cls = critic_response.get("Error_Class", "N/A") if critic_response else "System_Fail"

        history["reasons"].append(reason)
        history["error_classes"].append(err_cls)

        if critic_response and ("PASS" in str(critic_response.get("status", "")).upper() or "A" in str(critic_response.get("Rating", "")).upper()):
            history["accepted_index"] = len(history["attempts"])
            history["final_data"] = current_hint
            return history
            
        if i == MAX_REFINE_LOOPS: break
            
        refiner_input = (f"Source Text:\n{text}\n\nQA:\n{qa_str}\n\nFAILED HINT:\n{cur_h}\n\n"
                         f"### CRITICAL ERROR CLASS: {err_cls}\n"
                         f"### REASONING: {reason}")
                         
        refined_hint, raw_ref = call_agent(language, "hint_refiner", prompts["hint_refiner"], refiner_input)
        if refined_hint:
            current_hint = refined_hint
            history["attempts"].append(current_hint)
            history["raw_responses"].append(raw_ref)
        else: break

    history["final_data"] = current_hint
    return history

def run_recovery_pipeline(language: str, text: str, failed_qa: Dict, hint_history: Dict) -> Dict:
    if STOP_PIPELINE.is_set(): return None # ### NEW

    prompts = SYSTEM_PROMPTS[language]
    failed_items = []
    attempts = hint_history.get("attempts", [])
    reasons = hint_history.get("reasons", [])
    
    for i, attempt in enumerate(attempts):
        h_text = attempt.get("hint", attempt.get("Hint", "N/A"))
        r_text = reasons[i] if i < len(reasons) else "Unknown"
        failed_items.append({"hint": h_text, "critic_rejection_reason": r_text})
    
    formatted_input = (
        f"Text:\n{text}\n"
        f"Failed Question: {failed_qa.get('question')}\n"
        f"Failed Answer: {failed_qa.get('answer')}\n"
        f"Failed Hints & Reasons: {json.dumps(failed_items, ensure_ascii=False)}"
    )

    recovery_data, raw_resp = call_agent(language, "qa_hint_solver", prompts["qa_hint_solver"], formatted_input)
    
    if recovery_data:
        recovery_data["raw_response"] = raw_resp
        return recovery_data
    return None

# ------------------------------------------------------------------------------
# 5. EXECUTION & RESUME LOGIC
# ------------------------------------------------------------------------------

def process_dataset(config):
    lang = config["language"]
    infile = config["input"].strip()
    outfile = get_path(config["output"])
    actual_outfile = get_path(f"sample_{config['output']}") if DRY_RUN else outfile
    
    debug_file = get_path(f"debug_{lang}.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"=== {lang.upper()} DEBUG LOG ===\n")

    if not os.path.exists(infile): 
        logging.error(f"[{lang}] Input file {infile} not found.")
        return

    df = pd.read_csv(infile)
    processed_hashes = set()
    
    if os.path.exists(actual_outfile):
        try:
            existing_df = pd.read_csv(actual_outfile)
            if "source_text" in existing_df.columns:
                processed_hashes = set(existing_df["source_text"].apply(lambda x: str(x)[:50]))
        except: pass

    # ### NEW: Filter DF first to calculate true remaining count
    rows_to_process = []
    for idx, row in df.iterrows():
        text = str(row.get("extract", "")).strip()
        if not text or len(text) < 20 or text[:50] in processed_hashes: 
            continue
        rows_to_process.append(row)

    # ### NEW: TQDM Progress Bar
    # We wrap the rows in tqdm to create the visual bar
    print(f"[{lang}] Starting... {len(rows_to_process)} rows remaining.")
    
    # We use 'position' to try and separate bars if running in parallel, but simple print is often safer.
    # Here we just use a standard bar with the language as the description.
    for row in tqdm(rows_to_process, desc=f"Processing {lang}", unit="row"):
        
        # ### NEW: CRITICAL CHECK - Stop if pipeline is halted
        if STOP_PIPELINE.is_set():
            logging.warning(f"[{lang}] Thread stopping due to Rate Limit event.")
            break

        text = str(row.get("extract", "")).strip()
        
        # --- ORCHESTRATION LOOP ---
        current_qa_source = None 
        qa_hist = None
        hint_hist = None
        recovery_logs = []
        final_success = False

        for recovery_step in range(MAX_RECOVERY_LOOPS + 1):
            if STOP_PIPELINE.is_set(): break # Quick Exit

            # 1. Run QA Pipeline
            qa_hist = run_qa_pipeline(lang, text, initial_qa=current_qa_source)
            
            if not qa_hist["final_data"] or qa_hist["accepted_index"] == -1:
                # logging.warning(f"[{lang}] QA failed step {recovery_step}.") # Optional: Reduce clutter
                if recovery_step == MAX_RECOVERY_LOOPS: break
                break 

            # 2. Run Hint Pipeline
            hint_hist = run_hint_pipeline(lang, text, qa_hist["final_data"])
            
            if hint_hist["accepted_index"] != -1:
                final_success = True
                break 
            
            if recovery_step < MAX_RECOVERY_LOOPS:
                # logging.info(f"[{lang}] Triggering Recovery ({recovery_step+1})...")
                bad_qa = qa_hist["final_data"] if qa_hist["final_data"] else {"question": "N/A", "answer": "N/A"}
                bad_hint_hist = hint_hist if hint_hist else {"attempts": []}
                
                recovery_data = run_recovery_pipeline(lang, text, bad_qa, bad_hint_hist)
                
                if recovery_data:
                    recovery_logs.append({
                        "step": recovery_step + 1,
                        "diagnosis": recovery_data.get("Diagnosis", ""),
                        "strategy": recovery_data.get("Strategy", ""),
                        "new_q": recovery_data.get("question"),
                        "new_a": recovery_data.get("answer")
                    })
                    current_qa_source = {"question": recovery_data["question"], "answer": recovery_data["answer"]}
                else:
                    break 
            
        # --- END ORCHESTRATION LOOP ---

        if not final_success:
            continue

        # PREPARE ROW DATA
        new_row = {
            "source_text": text, "language": lang,
            "question_final": qa_hist["final_data"].get("question"),
            "answer_final": qa_hist["final_data"].get("answer"),
            "hint_final": hint_hist["final_data"].get("hint", hint_hist["final_data"].get("Hint")),
            "qa_accepted_at_loop": qa_hist["accepted_index"],
            "hint_accepted_at_loop": hint_hist["accepted_index"],
            "recovery_count": len(recovery_logs)
        }

        # DYNAMIC QA/HINT COLUMNS
        for i in range(3):
            suffix = f"_{i+1}"
            if i < len(qa_hist["attempts"]):
                new_row[f"qa_q_step{suffix}"] = qa_hist["attempts"][i].get("question", "")
                new_row[f"qa_a_step{suffix}"] = qa_hist["attempts"][i].get("answer", "")
                new_row[f"qa_error_step{suffix}"] = qa_hist["error_classes"][i] if i < len(qa_hist["error_classes"]) else "None"
            else:
                new_row[f"qa_q_step{suffix}"] = ""
                new_row[f"qa_a_step{suffix}"] = ""
                new_row[f"qa_error_step{suffix}"] = ""
            
            if i < len(hint_hist["attempts"]):
                new_row[f"hint_step{suffix}"] = hint_hist["attempts"][i].get("hint", hint_hist["attempts"][i].get("Hint", ""))
                new_row[f"hint_error_step{suffix}"] = hint_hist["error_classes"][i] if i < len(hint_hist["error_classes"]) else "None"
            else:
                new_row[f"hint_step{suffix}"] = ""
                new_row[f"hint_error_step{suffix}"] = ""

        for i in range(MAX_RECOVERY_LOOPS):
            suffix = f"_{i+1}"
            if i < len(recovery_logs):
                log = recovery_logs[i]
                new_row[f"rec_diag{suffix}"] = log["diagnosis"]
                new_row[f"rec_strat{suffix}"] = log["strategy"]
                new_row[f"rec_new_q{suffix}"] = log["new_q"]
                new_row[f"rec_new_a{suffix}"] = log["new_a"]
            else:
                new_row[f"rec_diag{suffix}"] = ""
                new_row[f"rec_strat{suffix}"] = ""
                new_row[f"rec_new_q{suffix}"] = ""
                new_row[f"rec_new_a{suffix}"] = ""

        # ### SAVE IMMEDIATELY (Safety mechanism already in place)
        pd.DataFrame([new_row]).to_csv(actual_outfile, mode='a', header=not os.path.exists(actual_outfile), index=False, encoding='utf-8-sig')
        if DRY_RUN: return

def main():
    logging.info(f"Starting Pipeline (Parallel=True, DryRun={DRY_RUN})")
    # ### NEW: Inform user how to stop gracefully
    print("Press Ctrl+C to force quit (though data is saved row-by-row).")
    
    with ThreadPoolExecutor(max_workers=len(DATA_FILES)) as executor:
        executor.map(process_dataset, DATA_FILES)

    if STOP_PIPELINE.is_set():
        print("\n\n!!! PROCESS STOPPED EARLY DUE TO API RATE LIMITS !!!")
        print("Data generated up to the limit has been saved.")

if __name__ == "__main__":
    main()