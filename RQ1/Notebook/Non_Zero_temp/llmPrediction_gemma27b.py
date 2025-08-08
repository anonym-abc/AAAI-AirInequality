from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd
import re
import os
torch.set_float32_matmul_precision('high')

# === Load Data ===
df = pd.read_csv("AirQuality/Dataset/Ground_Truth_2023_Final.csv")
print(df.shape)
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")

# === Model Setup ===
model_id = "google/gemma-2-27b-it"
hf_token = "Your Token"
device = "cuda:1"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 1},
    torch_dtype=dtype,
    token=hf_token,
)

# === Prompt Templates ===
SYSTEM_PROMPT = (
    "You are an air pollution assistant. "
    "Strictly respond to queries with a single real number only. "
    "Do not include any units, explanation, or punctuation. Just a single number. "
)

USER_TEMPLATE = (
    "What is the average PM2.5 concentration (in μg/m³) in {city}, {state} during {month}, {year}? "
    "Give a single number only."
)

# === Grouping by city/state/month ===
grouped = df.groupby(["city", "state", df["YearMonth"].dt.month])

# === Output CSV Path ===
save_path = "AirQuality/RQ1/Dataset/gemma2_27b_it_2023_With_Temp.csv"

# === Resume if file exists ===
if os.path.exists(save_path):
    existing_df = pd.read_csv(save_path)
    print(f"Resuming from saved file with {len(existing_df)} rows.")
    processed_keys = set(zip(existing_df['city'], existing_df['state'], existing_df['month']))
else:
    print("No previous file found. Starting fresh.")
    processed_keys = set()

# === Batched Query Function ===
def batched_query_llm(city, state, month, year, num_samples=10):
    messages = []
    for _ in range(num_samples):
        user_prompt = USER_TEMPLATE.format(city=city, state=state, month=month, year=year)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        messages.append({"role": "user", "content": full_prompt})

    prompts = [tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.5,
            eos_token_id=tokenizer.eos_token_id,
        )

    generations = []
    for i in range(num_samples):
        decoded = tokenizer.decode(outputs[i][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        match = re.search(r"\d+(\.\d+)?", decoded)
        generations.append(float(match.group()) if match else float("nan"))

    return generations

# === Main Loop ===
counter = 0
for (city, state, month_num), group in tqdm(grouped, desc="Querying model"):
    month_name = datetime(1900, month_num, 1).strftime("%B")
    key = (city, state, month_name)

    if key in processed_keys:
        continue

    year = 2023
    try:
        preds = batched_query_llm(city, state, month_name, year, num_samples=10)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM — stopping execution.")
            raise
        else:
            preds = [float("nan")] * 10
            print(f"Runtime error at {key}: {e}")
        time.sleep(0.1)

    row = {
        "city": city,
        "state": state,
        "year": year,
        "month": month_name,
        "model": model_id,
    }
    for i in range(10):
        row[f"pm2.5_{i}"] = preds[i]

    try:
        pd.DataFrame([row]).to_csv(
            save_path,
            mode='a',
            index=False,
            header=not os.path.exists(save_path) or counter == 0
        )
        counter += 1
        print(f"Appended row {counter}: {city}, {state}, {month_name}")
    except Exception as e:
        print(f"[{datetime.now()}] Append error at row {counter}: {e}")
