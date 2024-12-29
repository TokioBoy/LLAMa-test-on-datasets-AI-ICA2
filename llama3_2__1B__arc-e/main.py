import torch
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import gc

# Initialize logging
print("Loading ARC-Easy dataset...")

# Create results filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"arc_easy_results_{timestamp}.csv"

print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto" if torch.cuda.is_available() else torch.float32
)
model.eval()


def create_prompt(question, choices):
    """Create a formatted prompt for the model"""
    choices_text = "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    return f"""Question: {question}
Choices:
{choices_text}

Answer:"""


def get_model_response(model, tokenizer, prompt, device):
    """Get model's response for a given prompt"""
    try:
        with torch.no_grad():
            # Format and tokenize input
            encodings = tokenizer(prompt, return_tensors="pt").to(device)

            # Get model output
            outputs = model(**encodings)
            logits = outputs.logits

            # Get probabilities for the last token
            probs = torch.softmax(logits[0, -1:], dim=-1)

            return probs.squeeze()

    except Exception as e:
        print(f"\nError in get_model_response: {str(e)}")
        return None


def evaluate_arc_easy(model, tokenizer, results_file, num_samples=None):
    """Evaluate model on ARC-Easy dataset"""
    print("\nEvaluating ARC Easy...")

    # Load dataset
    split = "test" if num_samples is None else f"test[:{num_samples}]"
    dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
    device = model.device

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
        gc.collect()  # Collect garbage to avoid memory buildup

        try:
            question = example['question']
            choices = example['choices']['text']
            correct_answer = example['choices']['label'].index(example['answerKey'])

            # Create prompt
            prompt = create_prompt(question, choices)

            # Get model response
            start_time = datetime.now()
            scores = []
            for choice in choices:
                choice_prompt = f"{prompt} {choice}"
                score = get_model_response(model, tokenizer, choice_prompt, device)
                if score is not None:
                    scores.append(score.max().item())
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
                'prompt': prompt,
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
    evaluate_arc_easy(model, tokenizer, results_file, num_samples=2251)
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
finally:
    gc.collect()