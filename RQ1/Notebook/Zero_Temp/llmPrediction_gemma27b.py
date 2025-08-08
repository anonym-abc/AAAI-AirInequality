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
df = pd.read_csv("AirQuality/Dataset/CitySubset_GroundTruth.csv")
print(df.shape)
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")

# === Model Setup ===
model_id = "google/gemma-2-27b-it"
hf_token = "Your Token"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    torch_dtype=dtype,
    token=hf_token,
)

# === Prompt Templates ===
SYSTEM_PROMPT = (
    "You are an air pollution assistant. "
    "Strictly respond to queries with a single real number only. "
    "Do not include any units, explanation, or punctuation. Just a single number."

)
# Originak prompt
USER_TEMPLATE = (
    "What is the average PM2.5 concentration (in μg/m³) in {city}, {state} during {month}, {year}? "
    "Give a single number only."
)

# Prompt variation 1:
USER_TEMPLATE = (
    "Please estimate the average PM2.5 level (μg/m3) for {city}, {state} in {month} {year}."
)

# Prompt variation 2:
USER_TEMPLATE = (
    "How polluted was the air in {city}, {state} in {month} {year}? Give the PM2.5 value."
)

# Prompt variation 3:
USER_TEMPLATE = (
    "Report the PM2.5 (μg/m³) in {city}, {state} for {month}, {year}."
)

# === Grouping by city/state/month ===
grouped = df.groupby(["city", "state", df["YearMonth"].dt.month])

# === Output CSV Path ===
save_path = "AirQuality/RQ1/Dataset/PV3/gemma2_27b_it_2023.csv"

# === Resume if file exists ===
if os.path.exists(save_path):
    existing_df = pd.read_csv(save_path)
    print(f"Resuming from saved file with {len(existing_df)} rows.")
    processed_keys = set(zip(existing_df['city'], existing_df['state'], existing_df['month']))
else:
    print("No previous file found. Starting fresh.")
    processed_keys = set()

# === Query Function ===
def query_llm(city, state, month, year):
    user_prompt = USER_TEMPLATE.format(city=city, state=state, month=month, year=year)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    messages = [{"role": "user", "content": full_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    match = re.search(r"\d+(\.\d+)?", decoded)
    return float(match.group()) if match else float("nan")

# === Main Loop ===
counter = 0
for (city, state, month_num), group in tqdm(grouped, desc="Querying model"):
    month_name = datetime(1900, month_num, 1).strftime("%B")
    key = (city, state, month_name)

    if key in processed_keys:
        continue

    year = 2023
    try:
        pred = query_llm(city, state, month_name, year)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM — stopping execution.")
            raise
        else:
            pred = f"ERROR: {str(e)}"
        time.sleep(0.1)

    row = {
        "city": city,
        "state": state,
        "year": year,
        "month": month_name,
        "model": model_id,
        "pm2.5": pred
    }

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
