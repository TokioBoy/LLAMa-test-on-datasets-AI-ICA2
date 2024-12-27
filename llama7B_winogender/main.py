import torch
import csv
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load WinoGrande dataset
dataset = load_dataset("oskarvanderwal/winogender", "all")

# Load the LLaMA model and tokenizer
model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# CSV setup: Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"winogender_results_{timestamp}"
existing_results_file = "winogender_results.csv"  # Set to your existing results file path, or leave as None if not applicable
csv_headers = ["sentence", "pronoun", "correct_referent", "occupation", "participant",
               "correct_perplexity", "incorrect_perplexity", "result"]

# Load existing results if the CSV file exists
processed_sentences = set()
total_correct_existing = 0
total_existing = 0
if os.path.exists(existing_results_file):
    print(f"Loading existing results from {existing_results_file}...")
    with open(existing_results_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_sentences.add(row["sentence"])
            if row["result"] == "Correct":
                total_correct_existing += 1
            total_existing += 1
else:
    print("No existing results file found. Starting fresh.")

# Function to calculate perplexity for a given continuation
def calculate_perplexity(sentence, continuation):
    input_text = f"{sentence} {continuation}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return torch.exp(torch.tensor(loss)).item()  # Convert loss to perplexity

# Evaluate co-reference resolution
results = []
total_correct_new = 0
total_new = 0
print("Starting Evaluation...")

with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    if len(processed_sentences) == 0:  # Write header only if starting fresh
        writer.writeheader()

    for idx, example in enumerate(tqdm(dataset['test'])):
        # Skip already processed sentences
        sentence = example['sentence']
        if sentence in processed_sentences:
            print(f"Skipping already processed sentence: {sentence}")
            continue

        # Extract sentence components
        pronoun = example['pronoun']
        occupation = example['occupation']
        participant = example['participant']
        correct_referent = example['label']  # 'occupation' or 'participant'

        # Create continuations for evaluation
        continuation_occupation = f"{pronoun} refers to {occupation}."
        continuation_participant = f"{pronoun} refers to {participant}."

        # Calculate perplexity
        perplexity_occupation = calculate_perplexity(sentence, continuation_occupation)
        perplexity_participant = calculate_perplexity(sentence, continuation_participant)

        # Determine result
        if correct_referent == "occupation":
            correct_perplexity = perplexity_occupation
            incorrect_perplexity = perplexity_participant
            result = perplexity_occupation < perplexity_participant
        else:
            correct_perplexity = perplexity_participant
            incorrect_perplexity = perplexity_occupation
            result = perplexity_participant < perplexity_occupation

        # Update counters
        total_new += 1
        if result:
            total_correct_new += 1

        # Save results
        result_row = {
            "sentence": sentence,
            "pronoun": pronoun,
            "correct_referent": correct_referent,
            "occupation": occupation,
            "participant": participant,
            "correct_perplexity": f"{correct_perplexity:.4f}",
            "incorrect_perplexity": f"{incorrect_perplexity:.4f}",
            "result": "Correct" if result else "Incorrect"
        }
        writer.writerow(result_row)

        # Add sentence to processed set and results list
        processed_sentences.add(sentence)
        results.append(result_row)

        # Print progress
        print(f"\nSentence {idx + 1}: {sentence}")
        print(f"Pronoun: {pronoun}")
        print(f"Perplexity (Occupation): {perplexity_occupation:.4f}")
        print(f"Perplexity (Participant): {perplexity_participant:.4f}")
        print(f"Correct Referent: {correct_referent} | Result: {'Correct' if result else 'Incorrect'}\n")

        # Print current accuracy
        current_accuracy = (total_correct_new / total_new) * 100
        print(f"Current Accuracy (New Data): {current_accuracy:.2f}%")

# Calculate and display final accuracy
overall_total_correct = total_correct_existing + total_correct_new
overall_total = total_existing + total_new
overall_accuracy = (overall_total_correct / overall_total) * 100 if overall_total > 0 else 0

print(f"\n--- Evaluation Summary ---")
print(f"Existing Results: {total_existing} | Correct: {total_correct_existing}")
print(f"New Results: {total_new} | Correct: {total_correct_new}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print(f"Results saved to {csv_file}.")
