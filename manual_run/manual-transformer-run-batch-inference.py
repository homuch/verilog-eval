from re import fullmatch
from transformer_model import SimpleModelInference
from tqdm import tqdm
import re
import sys
import argparse
from constants import MODEL_PATH_DICT, BUILD_PATH, DATASET_PATH_PREFIX
from transformers import pipeline

def get_tasks(task_name):
    assert task_name in ["code-complete-iccad2023", "spec-to-rtl"], f"Unknown task '{task_name}'"
    with open(f"{DATASET_PATH_PREFIX}{task_name}/problems.txt", "r") as f:
        return f.read().splitlines()

def load_prompt(prompt_file_prefix):
    with open(f"{prompt_file_prefix}_fullprompt.txt", "r") as f:
        full_prompt = f.read()
    with open(f"{prompt_file_prefix}_systemprompt.txt", "r") as f:
        system_prompt = f.read()
    return full_prompt, system_prompt


def remove_thinking(in_text):
    """
    Splits on '</think>' and returns only the part after it,
    effectively discarding any "thinking" text in the generation.
    """
    return in_text.split("</think>")[-1]

def pipeline_output_to_text(response):
    assert response["generated_text"][-1]["role"] == "assistant"
    return remove_thinking(response["generated_text"][-1]["content"])


def main():
    #  -v --verbose        Display the prompt
    #  -h --help           Display this message
    #  -m --model          LLM model to use (default: gpt-3.5-turbo)
    #  -t --temperature    LLM model temperature (default: 0.85)
    #  -p --top-p          LLM model top_p (default: 0.95)
    #  -n --max-tokens     LLM model max_tokens (default: 1024)
    #     --task           Task to prompt (default: code-complete-iccad2023)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--top_p", type=float, default=0.95)
    argparser.add_argument("--temperature", type=float, default=0.8)
    argparser.add_argument("--max_tokens", type=int, default=1024)
    # task = ["code-complete-iccad20", "spec-to-rtl"]
    argparser.add_argument("--task", type=str, choices=["code-complete-iccad2023", "spec-to-rtl"], default="code-complete-iccad2023")
    argparser.add_argument("--test", action="store_true", default=False)
    argparser.add_argument("--samples", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=2)

    args = argparser.parse_args()

    try:
        model_path = MODEL_PATH_DICT[args.model]
    except KeyError:
        print(f"Model '{args.model}' not found.")
        sys.exit(1)

    task_list = get_tasks(args.task)
    samples_num = args.samples

    pipe = pipeline("text-generation", model=model_path)

    # Determine sampling parameters
    do_sample = (args.temperature != 0)
    top_p = args.top_p if do_sample else None
    temperature = args.temperature if do_sample else None
    top_k = None
    generate_kwargs = {
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": args.max_tokens
    }

    if args.test:
        print("Running in test mode.")
        print("task list:")
        print(task_list)
        print("testing llama cpp:")
        system_msg = "You are a story writing assistant."
        full_prompt = "Write a story about llamas."
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_prompt},
        ]
        resp = pipe(msgs, **generate_kwargs)
        print(resp)
        print(pipeline_output_to_text(resp[0]))
        sys.exit(0)
    else:
        print("Running in production mode.")

    def dataset():
        for task in task_list:
            for sample in range(1, samples_num + 1):
                prompt_file_prefix = f"{BUILD_PATH}/{task}/{task}_sample{sample:02d}"
                full_prompt, system_msg = load_prompt(prompt_file_prefix)
                msgs = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_prompt},
                ]
                yield msgs

    generated_text_list = []
    i=0
    for out in tqdm(pipe(dataset(), batch_size=args.batch_size, **generate_kwargs), total=len(task_list)*samples_num):
        generated_text = pipeline_output_to_text(out[0])
        generated_text_list.append(generated_text)
        task=task_list[i//samples_num]
        sample=i%samples_num+1
        prompt_file_prefix = f"{BUILD_PATH}/{task}/{task}_sample{sample:02d}"
        with open(f"{prompt_file_prefix}_response.txt", "w") as f:
            f.write(generated_text)

        i+=1


if __name__ == "__main__":
    main()
