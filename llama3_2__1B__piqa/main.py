import torch
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import gc

# Initialize logging
print("Loading PIQA dataset...")

# Create results filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"piqa_results_{timestamp}.csv"

print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.2-1B"


def get_device_config():
    return {'cpu': '24GB'}


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cpu',
    max_memory=get_device_config(),
    offload_folder="model_offload",
    offload_state_dict=True,
    low_cpu_mem_usage=True
)
model.eval()


def create_prompt(question, choice):
    """Create a formatted prompt for the model"""
    return f"""Question: {question}
Answer: {choice}"""


def get_choice_probs(model, tokenizer, question, choice):
    """Calculate probability for a given choice"""
    try:
        with torch.no_grad():
            # Format input
            prompt = create_prompt(question, choice)
            encodings = tokenizer(prompt, return_tensors="pt")

            # Get model output
            outputs = model(**encodings)
            logits = outputs.logits

            # Get probabilities for choice tokens
            choice_ids = tokenizer(choice, add_special_tokens=False)['input_ids']
            choice_len = len(choice_ids)

            # Calculate mean probability for choice tokens
            probs = torch.softmax(logits[0, -(choice_len + 1):-1], dim=-1)
            choice_probs = torch.tensor([probs[i][id].item() for i, id in enumerate(choice_ids)])

            return choice_probs.mean().item()

    except Exception as e:
        print(f"\nError in get_choice_probs: {str(e)}")
        return None


def evaluate_piqa(model, tokenizer, results_file, num_samples=None):
    """Evaluate model on PIQA dataset"""
    print("\nEvaluating PIQA...")

    # Load dataset
    split = "validation" if num_samples is None else f"validation[:{num_samples}]"
    dataset = load_dataset("piqa", split=split, trust_remote_code=True)

    # Initialize counters and timers
    correct = 0
    total = 0
    total_response_time = 0.0

    # Create results CSV file with headers
    csv_headers = ['question', 'choices', 'prompt', 'predicted_answer', 'correct_answer', 'correct', 'response_time']
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # Evaluation loop
    for idx, example in enumerate(tqdm(dataset)):
        if idx % 5 == 0:  # Garbage collection every 5 examples
            gc.collect()

        try:
            question = example['goal']
            choices = [example['sol1'], example['sol2']]
            correct_answer = example['label']

            # Get model responses
            start_time = datetime.now()
            scores = []
            for choice in choices:
                score = get_choice_probs(model, tokenizer, question, choice)
                if score is not None:
                    scores.append(score)
                else:
                    scores.append(0.0)

            response_time = (datetime.now() - start_time).total_seconds()
            total_response_time += response_time

            # Evaluate response
            predicted_answer = scores.index(max(scores))
            is_correct = (predicted_answer == correct_answer)

            # Update counters
            if is_correct:
                correct += 1
            total += 1

            # Save result
            result = {
                'question': question,
                'choices': str(choices),
                'prompt': create_prompt(question, choices[predicted_answer]),
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'correct': is_correct,
                'response_time': response_time
            }

            with open(results_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(result)

            # Print progress
            if (idx + 1) % 10 == 0:
                print(f"\nProcessed question: {question}")
                print(f"Model prediction: Choice {predicted_answer + 1}")
                print(f"Correct: {is_correct}")
                print(f"Current accuracy: {(correct / total) * 100:.2f}%")

        except Exception as e:
            print(f"\nError processing example {idx}: {str(e)}")
            continue

    # Calculate final metrics
    accuracy = correct / total if total else 0
    avg_response_time = total_response_time / total if total else 0

    # Print final results
    print(f"\nEvaluation completed!")
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Total questions processed: {total}")
    print(f"Correct answers: {correct}")
    print(f"Incorrect answers: {total - correct}")

    return accuracy


# Run evaluation
try:
    evaluate_piqa(model, tokenizer, results_file, num_samples=1838)
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
finally:
    gc.collect()