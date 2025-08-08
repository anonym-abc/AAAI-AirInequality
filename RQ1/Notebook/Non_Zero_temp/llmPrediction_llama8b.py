import torch
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Data ===
df = pd.read_csv("AirQuality/Dataset/Ground_Truth_2023_Final.csv")
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")
df_2023 = df[df["YearMonth"].dt.year == 2023]

# === Settings ===
model_id = "meta-llama/Llama-3.1-8B-Instruct"
hf_token = "Your Token"
device = torch.device("cuda:2")          
dtype = torch.bfloat16
save_path = "AirQuality/RQ1/Dataset/llama3_8b_it_2023_predictions_With_Temp.csv"
save_every = 25
num_iterations = 10                   

# === Load Model ===
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 2},               # Load model on GPU 2 explicitly
    torch_dtype=dtype,
    token=hf_token,
)
model.eval()

if hasattr(torch, "compile"):
    model = torch.compile(model)

# === Prompt templates ===
SYSTEM_PROMPT = (
    "You are an air pollution assistant. "
    "Strictly respond to queries with a single real number only. "
    "Do not include any units, explanation, or punctuation. Just a single number."
    
)
USER_TEMPLATE = (
    "What is the average PM2.5 concentration (in μg/m³) in {city}, {state} during {month}, {year}? "
    "Give a single number only."
)

# === Resume logic ===
rows = []
if os.path.exists(save_path):
    existing_df = pd.read_csv(save_path)
    print(f"Resuming from saved file with {len(existing_df)} rows.")
    processed_keys = set(zip(existing_df["city"], existing_df["state"], existing_df["month"]))
    rows = existing_df.to_dict("records")
else:
    print("No previous file found. Starting fresh.")
    processed_keys = set()

# === Batched inference function ===
def query_llm_batch(city, state, month, year, batch_size=10):
    # Prepare batch of identical prompts
    messages = []
    for _ in range(batch_size):
        user_prompt = USER_TEMPLATE.format(city=city, state=state, month=month, year=year)
        full_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=True)
        messages.append(prompt)
        
    inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.5,              
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i in range(batch_size):
        decoded = tokenizer.decode(outputs[i][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        match = re.search(r"\d+(\.\d+)?", decoded)
        results.append(float(match.group()) if match else float("nan"))
    return results

# === Group and Run ===
grouped = df_2023.groupby(["city", "state", df_2023["YearMonth"].dt.month])

counter = len(rows)
for (city, state, month_num), _ in tqdm(grouped, desc="Querying model"):
    month_name = datetime(1900, month_num, 1).strftime("%B")
    key = (city, state, month_name)
    if key in processed_keys:
        continue

    year = 2023
    try:
        preds = query_llm_batch(city, state, month_name, year, batch_size=num_iterations)
    except Exception as e:
        preds = [float("nan")] * num_iterations
        print(f"Error for {key}: {e}")
    time.sleep(0.05)

    row = {
        "city": city,
        "state": state,
        "year": year,
        "month": month_name,
        "model": model_id,
    }
    for i, val in enumerate(preds):
        row[f"pm2.5_{i}"] = val

    rows.append(row)
    counter += 1

    if counter % save_every == 0:
        try:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"[{datetime.now()}] Saved at {counter} rows.")
        except Exception as e:
            print(f"[{datetime.now()}] Save error at row {counter}: {e}")

# Final save
pd.DataFrame(rows).to_csv(save_path, index=False)
