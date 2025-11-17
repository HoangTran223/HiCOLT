# Adapted for Kaggle environment (no argparse, inline key)
import json
import random
import requests
import time
import os
from types import SimpleNamespace

# === CONFIGURATION ===
API_BASE = "https://api.openai.com/v1"
API_KEY = ""
args = SimpleNamespace(
    API_KEY=API_KEY,
    API_BASE=API_BASE,
    dataset_description="include",        # or 'exclude'
    model="gpt-4.1"
)

# === OpenAI API Initialization (using direct HTTP requests) ===
print("Using direct HTTP requests to OpenAI API.")


# === Prompt Generation Function ===
def get_prompts_for_openai(include_dataset_description=True):
    # Using system prompt without dataset description (as per your requirement)
    system_prompt = """You are a helpful assistant evaluating the top words of a topic model output for a given topic. Please rate how related the following words are to each other on a scale from 1 to 3 ("1" = not very related, "2" = moderately related, "3" = very related). 
    Reply with a single number, indicating the overall appropriateness of the topic."""
    return system_prompt

# === Main Logic ===
import sys
random.seed(42)

# Specify the result directories to process
result_dirs = [
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-23_09-46-40", 
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-23_09-20-43", 
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-23_08-32-45", 
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-23_06-48-58", 
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-22_18-35-26",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-22_13-57-43",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-22_13-41-50",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-22_13-40-30",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-22_13-35-13",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-21_23-28-23",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-21_09-33-19",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-21_05-26-36",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-21_01-44-41",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-19_15-35-23",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-18_15-16-15",
    "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/s_result/2025-07-17_16-24-23"
]
result_file = "/mnt/nam_x/nam_x/NeuroMax_/HiCOT/result_llm_1.txt"

# Prepare prompt (using default without dataset description)
use_description = (args.dataset_description == 'include')
system_prompt= get_prompts_for_openai(include_dataset_description=use_description)

# Ensure output folder exists and clear previous results
os.makedirs(os.path.dirname(result_file), exist_ok=True)
open(result_file, "w").close() # Clear the file for a new run

print(f"Starting batched evaluation. Summary will be saved to '{result_file}'")
for dir_path in result_dirs:
    # Determine dataset and model from a results_*.txt filename
    dataset = None
    model_name = None
    try:
        for fname in os.listdir(dir_path):
            if fname.startswith("results_dataset") and fname.endswith(".txt"):
                parts = fname.split("_")
                dataset = parts[1].replace("dataset", "")
                # model parts up to 'topics'
                idx = next(i for i, p in enumerate(parts) if p.startswith("topics"))
                model_parts = parts[2:idx]
                # first part has 'model' prefix
                model_name = "_".join([model_parts[0].replace("model", "")] + model_parts[1:])
                break
    except Exception:
        print(f"Failed to parse dataset/model in {dir_path}, skipping.")
        continue

    if not dataset or not model_name:
        print(f"Could not find results file in {dir_path}, skipping.")
        continue

    # Read topics from top_words_10.txt
    topics = []
    topics_file = os.path.join(dir_path, "top_words_10.txt")
    try:
        with open(topics_file) as tf:
            for line in tf:
                words = line.strip().split()
                if words:
                    topics.append(words)
        print(f"Loaded {len(topics)} topics for {dataset} / {model_name}")
    except Exception as e:
        print(f"Error reading topics file {topics_file}: {e}")
        continue

    # Evaluate each topic (3 runs each)
    ratings = []
    for i, topic in enumerate(topics):
        if len(topic) < 2:
            continue
        
        print(f"  > Evaluating {dataset}/{model_name}: Topic {i+1}/{len(topics)}", end='\r')

        current_topic_words = topic[:10]
        for run in range(3):
            words_to_shuffle = list(current_topic_words)
            random.shuffle(words_to_shuffle)
            user_prompt = ", ".join(words_to_shuffle)
            
            # Retry logic for transient errors (like 502, 503, 504)
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Use direct HTTP request to OpenAI API
                    url = f"{args.API_BASE}/chat/completions"
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {args.API_KEY}'
                    }
                    payload = {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 1.0,
                        "max_tokens": 1,
                        "logit_bias": {
                            "16": 100,  # token for "1"
                            "17": 100,  # token for "2"
                            "18": 100   # token for "3"
                        }
                    }
                    
                    http_response = requests.post(url, headers=headers, json=payload)
                    http_response.raise_for_status()  # Raise an exception for bad status codes
                    
                    response_json = http_response.json()
                    text = response_json['choices'][0]['message']['content'].strip()

                    if text in ['1', '2', '3']:
                        ratings.append(int(text))
                    else:
                        print(f"Unexpected response '{text}' for {dataset}/{model_name}")
                    time.sleep(0.5)  # OpenAI rate limit - matching original script
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    # Check if it's a transient error (502, 503, 504, connection errors)
                    is_transient = False
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        # Rate limit error - exit immediately
                        if status_code == 429:
                            print(f"\nQuota limit reached (429). Exiting.")
                            sys.exit(1)
                        # Transient errors - retry
                        if status_code in [502, 503, 504]:
                            is_transient = True
                    else:
                        # Connection errors are also transient
                        is_transient = True
                    
                    if is_transient and retry < max_retries - 1:
                        wait_time = (retry + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"\nTransient error for {dataset}/{model_name}, run {run}, retry {retry+1}/{max_retries}: {e}")
                        print(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        # Non-transient error or final retry - log and continue
                        print(f"\nHTTP Error for {dataset}/{model_name}, run {run}: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            print(f"Response body: {e.response.text[:200]}")
                        time.sleep(0.5)
                        break
                        
                except (KeyError, IndexError) as e:
                    print(f"\nError parsing response for {dataset}/{model_name}, run {run}: {e}")
                    if 'response_json' in locals():
                        print(f"Full response: {response_json}")
                    time.sleep(0.5)
                    break
                    
                except Exception as e:
                    print(f"\nUnexpected error for {dataset}/{model_name}, run {run}: {e}")
                    time.sleep(0.5)
                    break

    # Compute average rating over all runs
    if ratings:
        # Clear the progress line before printing the final result for the model
        print(" " * 80, end='\r') 
        avg_rating = sum(ratings) / len(ratings)
        # Append result to file immediately
        with open(result_file, "a") as rf:
            rf.write(f"\"{dataset}\" - \"{model_name}\": {avg_rating:.4f}\n")
        print(f"{dataset} - {model_name}: average rating {avg_rating:.4f}")
    else:
        print(f"No valid ratings for {dataset}/{model_name}")

print(f"\nAll evaluations complete. Summary saved to '{result_file}'")