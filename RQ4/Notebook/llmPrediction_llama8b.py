import os
import warnings
warnings.filterwarnings(
    "ignore", 
    message=r"The following generation flags are not valid and may be ignored: \['temperature', 'top_p'\]"
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Data ===
df = pd.read_csv("AirQuality/RQ4/Dataset/GroundTruth_With_Aux_Data.csv")
df = df.dropna(subset=["AT", "avg_ndvi", "population"])
df["month"] = pd.to_datetime(df["month"], format="%Y-%m")

# === Settings ===
model_id = "meta-llama/Llama-3.1-8B-Instruct"
hf_token = "Your Token"
save_path = "AirQuality/RQ4/Dataset/llama3_8b_it_2023_pop.csv"
batch_size = 8

# === Load Model & Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={1: "78GiB", "cpu": "30GiB"}
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
    "Give a single number only.\n"
    "Additional context: Population = {pop}"
)
# "Additional context: Temperature (AT) = {at}°C, NDVI = {ndvi}, Population = {pop}."
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
        city, state, month_name, year, at, ndvi, population = item

        # Handle missing values
        at_str = "NaN" if pd.isna(at) else f"{round(at, 2)}"
        ndvi_str = "NaN" if pd.isna(ndvi) else f"{round(ndvi, 4)}"
        pop_str = "NaN" if pd.isna(population) else f"{int(population)}"

        user_prompt = USER_TEMPLATE.format(
            city=city.title(),
            state=state.title(),
            month=month_name,
            year=year,
            # at=at_str,
            # ndvi=ndvi_str,
            pop=pop_str
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:1")

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
        del inputs
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# === Prepare Unprocessed Items ===
counter = len(rows)
items_to_process = []

for _, row in df.iterrows():
    city = row["city"]
    state = row["state"]
    month_ts = row["month"]
    month_name = month_ts.strftime("%B")
    year = month_ts.year
    key = (city, state, month_name)
    if key in processed_keys:
        continue

    at = row.get("AT")
    ndvi = row.get("avg_ndvi")
    population = row.get("population")
    items_to_process.append((city, state, month_name, year, at, ndvi, population))

# === Run Batches ===
for i in tqdm(range(0, len(items_to_process), batch_size), desc="Querying remaining"):
    batch = items_to_process[i:i + batch_size]

    try:
        preds = query_llm_batch(batch)
    except Exception as e:
        print(f"[ERROR] Batch {i}-{i+batch_size}: {str(e)}")
        preds = [f"ERROR: {str(e)}"] * len(batch)

    for j, (city, state, month, year, _, _, _) in enumerate(batch):
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