import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc

def get_device_config():
    # Set a static memory configuration for CPU
    return {'cpu': '24GB'}

def get_choice_probs(model, tokenizer, context, choice, device):
    """Calculate probability for a given choice"""
    with torch.no_grad():
        # Format input
        input_text = f"{context} {choice}"

        # Tokenize
        encodings = tokenizer(input_text, return_tensors="pt").to(device)

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

def evaluate_winograd(model, tokenizer, num_samples=25):
    print("\nEvaluating Winograd...")
    dataset = load_dataset("winograd_wsc", "wsc273", split=f"test[:{num_samples}]", trust_remote_code=True)

    device = torch.device('cpu')  # Set device to CPU
    correct = total = 0

    for idx, example in enumerate(tqdm(dataset)):
        if idx % 5 == 0:
            gc.collect()

        try:
            text = example['text']
            pronoun = example['pronoun']
            candidates = example['options']
            correct_answer = example['label']

            scores = []
            for candidate in candidates:
                modified_text = text.replace(pronoun, candidate)
                score = get_choice_probs(model, tokenizer, "", modified_text, device)
                scores.append(score)

            if scores.index(max(scores)) == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

    accuracy = correct / total if total else 0
    print(f"\nWinograd Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

def main():
    try:
        # Load original LLaMa 7B model
        model = AutoModelForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            low_cpu_mem_usage=True
        )

        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        tokenizer.pad_token = tokenizer.eos_token

        evaluate_winograd(model, tokenizer, 1000)  # The size of the validation test: 273

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
