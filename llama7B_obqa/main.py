import torch
import csv
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime

# Load OpenBookQA dataset
print("Loading OpenBookQA dataset...")
ds = load_dataset("allenai/openbookqa", "main")

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
results_file = f"openbookqa_results_{timestamp}.csv"
existing_results_file = "openbookqa_results.csv"  # Path to existing results file, if any
csv_headers = ['id', 'question', 'choices', 'prompt', 'response', 'ground_truth', 'correct', 'response_time']


def create_prompt(question_stem, choices_text):
    """
    Create a prompt for OpenBookQA questions.
    """
    choices_str = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_text)])
    return f"""Question: {question_stem}
Choices:
{choices_str}
Which option (A, B, C, or D) is correct?

Answer:"""


def get_model_response(prompt):
    """
    Get the model's response to the provided prompt.
    """
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

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate_response(response, answer_key):
    """
    Evaluate the model's response against the correct answer key.
    """
    if response and response[0].upper() in "ABCD":
        return response[0].upper() == answer_key
    return False


# Load existing results into a dictionary for prompt lookup
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
total_response_time = 0.0  # To track cumulative response time

# Evaluation loop
print("Starting evaluation...")
for split in ['validation']:  # Adjust split if needed
    print(f"\nEvaluating {split} split...")

    for example in tqdm(ds[split]):
        # Extract question, choices, and answer key
        question_stem = example['question_stem']
        choices_text = example['choices']['text']
        answer_key = example['answerKey']
        question_id = example['id']

        # Create prompt
        prompt = create_prompt(question_stem, choices_text)

        # Check if prompt has already been processed
        if prompt in processed_prompts:
            # Retrieve existing result
            existing_result = processed_prompts[prompt]
            is_correct = existing_result['correct'] == 'True'  # Convert string to boolean
            response_time = float(existing_result['response_time'])
            correct += 1 if is_correct else 0
            total += 1
            total_response_time += response_time

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
        is_correct = evaluate_response(response, answer_key)

        # Update counters
        if is_correct:
            correct += 1
        total += 1

        # Save new result
        result = {
            'id': question_id,
            'question': question_stem,
            'choices': choices_text,
            'prompt': prompt,
            'response': response,
            'ground_truth': answer_key,
            'correct': is_correct,
            'response_time': response_time
        }

        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(result)

        # Print progress
        print(f"\nProcessed question ID: {question_id}")
        print(f"Model response: {response}")
        print(f"Correct: {is_correct}")
        print(f"Current accuracy: {(correct / total) * 100:.2f}%")

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
