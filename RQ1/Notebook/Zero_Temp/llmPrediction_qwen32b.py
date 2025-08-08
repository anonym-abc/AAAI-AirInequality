import os
import warnings
import torch
import time
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Suppress warnings ===
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# === Load Ground Truth Data ===
df = pd.read_csv("AirQuality/Dataset/CitySubset_GroundTruth.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")
df_2023 = df[df["YearMonth"].dt.year == 2023]

# === Model & Tokenizer Settings ===
model_id = "Qwen/Qwen3-32B"
hf_token = "Your Token"
device = torch.device("cuda:0")
dtype = torch.bfloat16
save_path = "AirQuality/RQ1/Dataset/PV3/qwen3_32b_2023.csv"
save_every = 25

# === Load Qwen3-32B ===
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map={"": 0},
    token=hf_token
)
model.eval()

if hasattr(torch, "compile"):
    model = torch.compile(model)

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

# === Inference Function ===
def query_llm(city, state, month, year):
    user_prompt = USER_TEMPLATE.format(city=city, state=state, month=month, year=year)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist()
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    # print(f"{city}, {state}, {month}, {year} -> {decoded}")
    match = re.search(r"\d+(\.\d+)?", decoded)
    return float(match.group()) if match else float("nan")

# === Run Inference ===
grouped = df_2023.groupby(["city", "state", df_2023["YearMonth"].dt.month])
counter = len(rows)

for (city, state, month_num), _ in tqdm(grouped, desc="Querying Qwen3-32B"):
    month_name = datetime(1900, month_num, 1).strftime("%B")
    key = (city, state, month_name)
    if key in processed_keys:
        continue

    try:
        pred = query_llm(city, state, month_name, 2023)
    except Exception as e:
        pred = f"ERROR: {str(e)}"
    time.sleep(0.05)

    row = {
        "city": city,
        "state": state,
        "year": 2023,
        "month": month_name,
        "model": model_id,
        "pm2.5": pred
    }
    rows.append(row)
    counter += 1

    if counter % save_every == 0:
        try:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"[{datetime.now()}] Saved at {counter} rows.")
        except Exception as e:
            print(f"[{datetime.now()}] Save error at row {counter}: {e}")

# === Final Save ===
df_result = pd.DataFrame(rows)
df_result.to_csv(save_path, index=False)