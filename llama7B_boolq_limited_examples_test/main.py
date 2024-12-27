import torch
import csv
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime

# Load dataset
print("Loading BoolQ dataset...")
ds = load_dataset("google/boolq")

# Load model and tokenizer
print("Loading model and tokenizer...")
model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto" if torch.cuda.is_available() else torch.float32
)
model.eval()

# Prepare results file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"boolq_results_{timestamp}.csv"
existing_results_file = "boolq_results.csv"  # Set to your existing results file path, or leave as None if not applicable
csv_headers = ['question', 'passage', 'prompt', 'response', 'ground_truth', 'correct', 'response_time']


def create_prompt(question, passage):
    return f"""Question: {question}
Context: {passage}
Is this statement true? Answer with yes or no.

Answer:"""


def get_model_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()


def evaluate_response(response, label):
    first_word = response.split()[0] if response else ""

    if first_word in ["yes", "true"]:
        pred = True
    elif first_word in ["no", "false"]:
        pred = False
    else:
        return False

    return pred == label


# Load existing results into a set for prompt lookup
processed_prompts = {}
if existing_results_file and os.path.exists(existing_results_file):
    print(f"Loading existing results from {existing_results_file}...")
    with open(existing_results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_prompts[row['prompt']] = row
else:
    print("No existing results file found. Starting fresh.")

# Create new results CSV file with headers
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    writer.writeheader()

# Initialize counters and timers
correct = 0
total = 0
processed_count = 0  # Counter for processed questions
limit = 327  # Limit for number of questions to process
total_response_time = 0.0  # To track cumulative response time

# Evaluation loop
print("Starting evaluation...")
for split in ['validation']:  # Adjust split to match dataset
    print(f"\nEvaluating {split} split...")

    for example in tqdm(ds[split]):
        if processed_count >= limit:
            print(f"Reached processing limit of {limit} questions. Stopping.")
            break

        # Create prompt
        prompt = create_prompt(example['question'], example['passage'])

        # Check if prompt has already been processed
        if prompt in processed_prompts:
            # Retrieve existing result
            existing_result = processed_prompts[prompt]
            is_correct = existing_result['correct'] == 'True'  # Convert string to boolean
            response_time = float(existing_result['response_time'])
            correct += 1 if is_correct else 0
            total += 1
            total_response_time += response_time
            processed_count += 1  # Increment processed count

            # Copy existing result to the new CSV file
            with open(results_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(existing_result)

            print(f"Skipped already processed prompt: {prompt}")
            continue

        # Get model response
        start_time = datetime.now()
        response = get_model_response(prompt)
        response_time = (datetime.now() - start_time).total_seconds()
        total_response_time += response_time  # Accumulate response time

        # Evaluate response
        is_correct = evaluate_response(response, example['answer'])

        # Update counters
        if is_correct:
            correct += 1
        total += 1
        processed_count += 1  # Increment processed count

        # Save new result
        result = {
            'question': example['question'],
            'passage': example['passage'],
            'prompt': prompt,
            'response': response,
            'ground_truth': example['answer'],
            'correct': is_correct,
            'response_time': response_time
        }

        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(result)

        # Print progress
        print(f"\nProcessed prompt: {prompt}")
        print(f"Model response: {response}")
        print(f"Correct: {is_correct}")
        print(f"Current accuracy: {(correct / total) * 100:.2f}%")

    if processed_count >= limit:
        break  # Exit the outer loop if the limit is reached

# Calculate average response time
avg_response_time = total_response_time / total if total > 0 else 0

# Print final results
final_accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nEvaluation completed!")
print(f"Final accuracy: {final_accuracy:.2f}%")
print(f"Average response time: {avg_response_time:.2f} seconds")
print(f"Results saved to: {results_file}")
print(f"Total questions processed: {total}")
print(f"Correct answers: {correct}")
print(f"Incorrect answers: {total - correct}")
