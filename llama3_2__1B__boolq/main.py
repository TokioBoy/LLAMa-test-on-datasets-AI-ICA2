import torch
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime

# Load dataset
print("Loading BoolQ dataset...")
ds = load_dataset("google/boolq")

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
results_file = f"boolq_results_{timestamp}.csv"
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


# Initialize results storage
correct = 0
total = 0

# Create CSV file with headers
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_headers)
    writer.writeheader()

# Evaluation loop
print("Starting evaluation...")
for split in ['validation', 'test']:
    print(f"\nEvaluating {split} split...")

    for example in tqdm(ds[split]):
        # Create prompt
        prompt = create_prompt(example['question'], example['passage'])

        # Get model response
        start_time = datetime.now()
        response = get_model_response(prompt)
        response_time = (datetime.now() - start_time).total_seconds()

        # Evaluate response
        is_correct = evaluate_response(response, example['answer'])

        # Update counters
        if is_correct:
            correct += 1
        total += 1

        # Save results
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

# Print final results
final_accuracy = (correct / total) * 100
print(f"\nEvaluation completed!")
print(f"Final accuracy: {final_accuracy:.2f}%")
print(f"Results saved to: {results_file}")