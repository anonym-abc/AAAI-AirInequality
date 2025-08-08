import os
import warnings
warnings.filterwarnings(
    "ignore", 
    message=r"The following generation flags are not valid and may be ignored: \['temperature', 'top_p'\]"
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Prevent fragmentation

import torch
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Data ===
df = pd.read_csv("AirQuality/Dataset/CitySubset_GroundTruth.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")

# === Settings ===
model_id = "meta-llama/Llama-3.3-70B-Instruct"
hf_token = "Your Token"
save_path = "AirQuality/RQ1/Dataset/PV3/llama3_70b_it_2023.csv"
batch_size = 16

# === Load Model & Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={3: "78GiB", "cpu": "30GiB"}
)
model.eval()

# === Prompt Templates ===
SYSTEM_PROMPT = (
    "You are an air pollution assistant. "
    "Strictly respond to queries with a single real number only. "
    "Do not include any units, explanation, or punctuation. Just a single number."
)
USER_TEMPLATE = (
    "What is the average PM2.5 concentration (in μg/m³) in {city}, {state} during {month}, {year}? "
    "Give a single number only."
)

# Prompt variation 1:
# USER_TEMPLATE = (
#     "Please estimate the average PM2.5 level (μg/m3) for {city}, {state} in {month} {year}."
# )

# Prompt variation 2:
# USER_TEMPLATE = (
#     "How polluted was the air in {city}, {state} in {month} {year}? Give the PM2.5 value."
# )

# Prompt variation 3:
USER_TEMPLATE = (
    "Report the PM2.5 (μg/m³) in {city}, {state} for {month}, {year}."
)

# === Resume Logic ===
rows = []
if os.path.exists(save_path):
    existing_df = pd.read_csv(save_path)
    print(f"Resuming from saved file with {len(existing_df)} rows.")
    processed_keys = set(zip(existing_df["city"], existing_df["state"], existing_df["month"]))
    rows = existing_df.to_dict("records")
else:
    print("No previous file found. Starting fresh.")
    processed_keys = set()

# === Query LLM Batch ===
def query_llm_batch(batch):
    prompts = []
    for item in batch:
        city, state, month, year = item
        user_prompt = USER_TEMPLATE.format(
            city=city.title(),
            state=state.title(),
            month=month,
            year=year,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:3")

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        preds = []
        for response in decoded:
            match = re.search(r"\d+(\.\d+)?", response)
            preds.append(float(match.group()) if match else float("nan"))
        return preds

    finally:
        # Release GPU memory
        del inputs
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# === Prepare Unprocessed Items ===
counter = len(rows)
items_to_process = []
grouped = df.groupby(["city", "state", df["YearMonth"].dt.month])

for (city, state, month_num), group_df in grouped:
    month_name = datetime(1900, month_num, 1).strftime("%B")
    key = (city, state, month_name)
    if key in processed_keys:
        continue
    year = 2023
    items_to_process.append((city, state, month_name, year))

# === Run Batches ===
for i in tqdm(range(0, len(items_to_process), batch_size), desc="Querying remaining"):
    batch = items_to_process[i:i + batch_size]

    try:
        preds = query_llm_batch(batch)
    except Exception as e:
        print(f"[ERROR] Batch {i}-{i+batch_size}: {str(e)}")
        preds = [f"ERROR: {str(e)}"] * len(batch)

    for j, (city, state, month, year) in enumerate(batch):
        rows.append({
            "city": city,
            "state": state,
            "year": year,
            "month": month,
            "model": model_id,
            "pm2.5": preds[j]
        })
        counter += 1

    try:
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"[{datetime.now()}] Saved at {counter} rows.")
    except Exception as e:
        print(f"[{datetime.now()}] Save error at row {counter}: {e}")

print(f"Done! Saved total {len(rows)} rows to {save_path}")