import torch
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime

# Load OpenBookQA dataset
print("Loading OpenBookQA dataset...")
ds = load_dataset("allenai/openbookqa", "main")

# Load model and tokenizer
print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.2-1B"
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
csv_headers = ['id', 'question', 'choices', 'prompt', 'response', 'ground_truth', 'correct', 'response_time']


def create_prompt(question_stem, choices_text):
    choices_str = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_text)])
    return f"""Question: {question_stem}
Choices:
{choices_str}
Which option (A, B, C, or D) is correct?

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

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate_response(response, answer_key):
    if response and response[0].upper() in "ABCD":
        return response[0].upper() == answer_key
    return False


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
