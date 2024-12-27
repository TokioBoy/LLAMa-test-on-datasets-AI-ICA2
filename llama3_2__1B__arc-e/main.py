import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc

def get_device_config():
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

def evaluate_arc_easy(model, tokenizer, num_samples=10):
    print("\nEvaluating ARC Easy...")
    dataset = load_dataset("ai2_arc", "ARC-Easy", split=f"test[:{num_samples}]")
    device = torch.device("cpu")
    correct = total = 0

    for idx, example in enumerate(tqdm(dataset)):
        gc.collect()  # Collect garbage to avoid memory buildup

        try:
            question = example['question']
            choices = example['choices']['text']
            correct_answer = example['choices']['label'].index(example['answerKey'])

            scores = []
            for choice in choices:
                score = get_choice_probs(model, tokenizer, question, choice, device)
                scores.append(score)

            if scores.index(max(scores)) == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

    accuracy = correct / total if total else 0
    print(f"\nARC Easy Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

def main():
    try:
        # Load original LLama 3.2 1B model
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        ).to("cpu")  # Ensure the model is moved to CPU

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = tokenizer.eos_token

        evaluate_arc_easy(model, tokenizer, 2251)  # The size of the validation test: 2251

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        gc.collect()  # Collect garbage at the end

if __name__ == "__main__":
    main()
