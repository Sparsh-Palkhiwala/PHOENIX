import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import json

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get probabilities from GPT-Neo (batched)
def get_token_probabilities(input_texts, target_tokens):
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Logits for the last token
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    return [{token: probs[i][tokenizer.convert_tokens_to_ids(token)].item() for token in target_tokens}
            for i in range(len(input_texts))]

# Define Bayesian Model
bayesian_model = BayesianNetwork([("Evidence", "Answer")])
cpt_data = pd.DataFrame({
    "Evidence": ["Low", "High", "High", "Low"],
    "Answer": ["yes", "yes", "no", "no"],
    "p": [0.2, 0.8, 0.7, 0.3]
})
bayesian_model.fit(cpt_data)
inference = VariableElimination(bayesian_model)

# Combine GPT-Neo with BIRD
def integrate_gpt_bird(input_text, target_tokens):
    gpt_probs = get_token_probabilities([input_text], target_tokens)[0]
    evidence = "High" if gpt_probs["yes"] > gpt_probs["no"] else "Low"
    result = inference.map_query(variables=["Answer"], evidence={"Evidence": evidence})
    return {
        "GPT_Neo_Prediction": max(gpt_probs, key=gpt_probs.get),
        "BIRD_Adjusted_Prediction": result["Answer"]
    }

# Load the CLadder dataset
cladder_file = "cladder-v1-q-balanced.json"
with open(cladder_file, "r") as f:
    data = json.load(f)

# Convert the JSON data to a DataFrame
cladder_data = pd.DataFrame(data)

# Ensure the DataFrame has 'input' (questions) and 'label' (answers)
cladder_data.rename(columns={"question": "input", "answer": "label"}, inplace=True)

# Debugging: Check the dataset structure
cladder_data = cladder_data.head(10)  # Limit the dataset for testing
print(cladder_data.head())  # Print the first few rows of the dataset
print(cladder_data.columns)  # Ensure column names are correct

# Evaluate CLadder dataset
def evaluate_model(data, target_tokens, use_bird=False, batch_size=16):
    predictions = []
    truths = []
    for idx in range(0, len(data), batch_size):
        batch = data.iloc[idx:idx + batch_size]
        input_texts = batch['input'].tolist()
        labels = batch['label'].tolist()

        if use_bird:
            batch_predictions = []
            for input_text in input_texts:
                result = integrate_gpt_bird(input_text, target_tokens)
                batch_predictions.append(result["BIRD_Adjusted_Prediction"])
            predictions.extend(batch_predictions)
        else:
            gpt_probs_list = get_token_probabilities(input_texts, target_tokens)
            predictions.extend([max(gpt_probs, key=gpt_probs.get) for gpt_probs in gpt_probs_list])

        truths.extend(labels)
        print(f"Processed {idx + len(batch)}/{len(data)} rows")
    return accuracy_score(truths, predictions)

# Evaluate Performance
target_tokens = ["yes", "no"]
accuracy_gpt = evaluate_model(cladder_data, target_tokens, use_bird=False)
accuracy_bird = evaluate_model(cladder_data, target_tokens, use_bird=True)

print(f"Accuracy (GPT-Neo): {accuracy_gpt}")
print(f"Accuracy (BIRD-Integrated): {accuracy_bird}")

