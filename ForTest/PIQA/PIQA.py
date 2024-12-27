import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gc


def get_device_config():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        usable_memory = total_memory - 1
        return {'cpu': '24GB', 0: f'{usable_memory}GB'}
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


def evaluate_piqa(model, tokenizer, num_samples=10):
    print("\nEvaluating PIQA...")
    dataset = load_dataset("piqa", split=f"validation[:{num_samples}]", trust_remote_code=True)
    device = 0 if torch.cuda.is_available() else 'cpu'
    correct = total = 0

    for idx, example in enumerate(tqdm(dataset)):
        if idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        try:
            question = example['goal']
            choices = [example['sol1'], example['sol2']]
            correct_answer = example['label']

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
    print(f"\nPIQA Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.max_split_size_mb = 128

    try:
        # Load original Mistral 7B
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            device_map='auto',
            max_memory=get_device_config(),
            offload_folder="model_offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True
        )

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token

        accuracy = evaluate_piqa(model, tokenizer, 10000) # the size of the validation test: 1838
        print(f"\nFinal Result - PIQA: {accuracy:.2%}")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()