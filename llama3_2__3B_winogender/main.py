import torch
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load WinoGender dataset
dataset = load_dataset("oskarvanderwal/winogender", "all")

# Load the LLaMA-7B model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# CSV setup: Save results
csv_file = "winogender_results.csv"
csv_headers = ["sentence", "pronoun", "correct_referent", "occupation", "participant",
               "correct_perplexity", "incorrect_perplexity", "result"]


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
print("Starting Evaluation...")

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    writer.writeheader()

    for idx, example in enumerate(tqdm(dataset['test'])):
        # Extract sentence components
        sentence = example['sentence']
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

        # Save results
        results.append({
            "sentence": sentence,
            "pronoun": pronoun,
            "correct_referent": correct_referent,
            "occupation": occupation,
            "participant": participant,
            "correct_perplexity": correct_perplexity,
            "incorrect_perplexity": incorrect_perplexity,
            "result": "Correct" if result else "Incorrect"
        })
        writer.writerow({
            "sentence": sentence,
            "pronoun": pronoun,
            "correct_referent": correct_referent,
            "occupation": occupation,
            "participant": participant,
            "correct_perplexity": f"{correct_perplexity:.4f}",
            "incorrect_perplexity": f"{incorrect_perplexity:.4f}",
            "result": "Correct" if result else "Incorrect"
        })

        # Print progress
        print(f"\nSentence {idx + 1}: {sentence}")
        print(f"Pronoun: {pronoun}")
        print(f"Perplexity (Occupation): {perplexity_occupation:.4f}")
        print(f"Perplexity (Participant): {perplexity_participant:.4f}")
        print(f"Correct Referent: {correct_referent} | Result: {'Correct' if result else 'Incorrect'}\n")

# Calculate and display final accuracy
total_correct = sum(1 for r in results if r['result'] == "Correct")
accuracy = (total_correct / len(results)) * 100
print(f"\nWinoGender Evaluation Accuracy: {accuracy:.2f}%")
print(f"Results saved to {csv_file}.")
