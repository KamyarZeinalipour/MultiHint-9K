import argparse
import pathlib
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import hazm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LANG_CODES = {
    'Farsi': 'fa', 'Persian': 'fa',
    'English': 'en', 'Italian': 'it'
}

print("Loading models... (This may take a moment)")

# 1. ROUGE & Hazm Setup
hazm_tokenizer = hazm.WordTokenizer()

class HazmTokenizerWrapper:
    def tokenize(self, text):
        return hazm_tokenizer.tokenize(text)

scorer_general = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
scorer_farsi = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=HazmTokenizerWrapper())

# -----------------------------------------------------------------------------
# Metric Functions
# -----------------------------------------------------------------------------

def calculate_rouge(ref, pred, lang_code):
    if not isinstance(ref, str) or not isinstance(pred, str): return 0.0
    # Use Farsi-specific tokenizer if needed
    scorer = scorer_farsi if lang_code == 'fa' else scorer_general
    return scorer.score(ref, pred)['rougeL'].fmeasure

def calculate_bertscore(refs, preds, lang_code):
    if not refs: return []
    # Using Multilingual BERT to handle EN, IT, FA
    P, R, F1 = bert_score(
        preds, refs, 
        model_type="bert-base-multilingual-cased", 
        lang=lang_code, 
        verbose=False, 
        rescale_with_baseline=False
    )
    return F1.tolist()

# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------

def evaluate_single_file(filepath):
    """Calculates ROUGE and BERTScore for a file."""
    try:
        df = pd.read_csv(filepath)
    except:
        return []

    if not {'Output', 'Assistant', 'language'}.issubset(df.columns):
        return []

    file_results = []
    
    # Per Language Group
    for language_label, group in df.groupby('language'):
        lang_code = LANG_CODES.get(language_label, 'en')
        refs = group['Assistant'].fillna("").astype(str).tolist()
        preds = group['Output'].fillna("").astype(str).tolist()
        
        # Calculate Metrics
        rouge_vals = [calculate_rouge(r, p, lang_code) for r, p in zip(refs, preds)]
        bert_vals = calculate_bertscore(refs, preds, lang_code)
        
        file_results.append({
            'language': language_label,
            'count': len(group),
            'rougeL': np.mean(rouge_vals),
            'bert_score': np.mean(bert_vals)
        })

    # Global Aggregate for this file
    if file_results:
        total = sum(r['count'] for r in file_results)
        file_results.append({
            'language': 'GLOBAL',
            'count': total,
            'rougeL': sum(r['rougeL'] * r['count'] for r in file_results) / total,
            'bert_score': sum(r['bert_score'] * r['count'] for r in file_results) / total
        })
        
    return file_results

def pair_files(base_files, ft_files):
    """Matches Base files to Fine-Tuned files by name similarity."""
    pairs = []
    ft_pool = list(ft_files)
    
    for base in base_files:
        best_match = None
        best_score = 0.0
        
        for ft in ft_pool:
            score = SequenceMatcher(None, base.name, ft.name).ratio()
            if base.stem in ft.name: score += 0.5 # Boost if substring
            
            if score > best_score:
                best_score = score
                best_match = ft
        
        if best_match and best_score > 0.4:
            # Create a clean Model Name from the filename
            model_name = base.stem.replace("_recall", "").replace("meta-llama_", "")
            pairs.append((model_name, base, best_match))
            ft_pool.remove(best_match)

    return pairs

# -----------------------------------------------------------------------------
# Main Driver
# -----------------------------------------------------------------------------

def process_directories(base_dir, ft_dir, output_file="final_report_err.csv"):
    base_path = pathlib.Path(base_dir)
    ft_path = pathlib.Path(ft_dir)
    
    base_files = sorted(list(base_path.glob("*.csv")))
    ft_files = sorted(list(ft_path.glob("*.csv")))
    
    model_pairs = pair_files(base_files, ft_files)
    
    all_comparisons = []
    print(f"\nFound {len(model_pairs)} matching model pairs.")
    
    # 1. Compute Raw Scores
    for model_name, base_p, ft_p in model_pairs:
        print(f"Processing Pair: {model_name}")
        base_res = evaluate_single_file(base_p)
        ft_res = evaluate_single_file(ft_p)
        
        base_dict = {r['language']: r for r in base_res}
        ft_dict = {r['language']: r for r in ft_res}
        
        languages = set(base_dict.keys()) | set(ft_dict.keys())
        
        for lang in languages:
            b_data = base_dict.get(lang)
            f_data = ft_dict.get(lang)
            
            if b_data and f_data:
                for metric in ['rougeL', 'bert_score']:
                    all_comparisons.append({
                        'Model': model_name,
                        'Language': lang,
                        'Metric': metric,
                        'Base': b_data[metric],
                        'FineTuned': f_data[metric]
                    })

    if not all_comparisons:
        print("No comparisons generated.")
        return

    df = pd.DataFrame(all_comparisons)

    # 2. Add 'ALL_MODELS' Average Row
    summary_rows = []
    for (lang, metric), group in df.groupby(['Language', 'Metric']):
        summary_rows.append({
            'Model': 'ALL_MODELS_AVG',
            'Language': lang,
            'Metric': metric,
            'Base': group['Base'].mean(),
            'FineTuned': group['FineTuned'].mean()
        })
    
    df_final = pd.concat([pd.DataFrame(summary_rows), df], ignore_index=True)

    # 3. Calculate Percentages
    # Standard Improvement: (New - Old) / Old
    df_final['Improvement (%)'] = ((df_final['FineTuned'] - df_final['Base']) / df_final['Base']) * 100
    
    # Error Reduction Rate: (New - Old) / (1 - Old)
    # Assumes max score is 1.0 (True for ROUGE and BERTScore)
    df_final['Error_Reduction (%)'] = ((df_final['FineTuned'] - df_final['Base']) / (1.0 - df_final['Base'])) * 100

    # 4. Formatting & Sorting
    df_final['Base'] = df_final['Base'].round(4)
    df_final['FineTuned'] = df_final['FineTuned'].round(4)
    df_final['Improvement (%)'] = df_final['Improvement (%)'].round(2)
    df_final['Error_Reduction (%)'] = df_final['Error_Reduction (%)'].round(2)
    
    # Sort: ALL_MODELS at top, then by Model Name, then Language
    df_final['sort_key'] = df_final['Model'].apply(lambda x: 0 if x == 'ALL_MODELS_AVG' else 1)
    df_final = df_final.sort_values(by=['sort_key', 'Model', 'Language', 'Metric']).drop(columns=['sort_key'])
    
    # Select final columns
    cols = ['Model', 'Language', 'Metric', 'Base', 'FineTuned', 'Improvement (%)', 'Error_Reduction (%)']
    df_final = df_final[cols]

    print("\n\n=== FINAL COMPARISON REPORT ===")
    print(df_final.to_string(index=False))
    
    df_final.to_csv(output_file, index=False)
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Folder containing base model CSVs")
    parser.add_argument("--finetuned", required=True, help="Folder containing fine-tuned model CSVs")
    args = parser.parse_args()
    
    process_directories(args.base, args.finetuned)