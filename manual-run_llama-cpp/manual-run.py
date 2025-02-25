from re import fullmatch
from llama_cpp import Llama
from tqdm import tqdm
import re
import sys
import argparse
from constants import MODEL_PATH_DICT, BUILD_PATH, DATASET_PATH_PREFIX
n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.


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

def clean_thinking_prompt(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

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

    args = argparser.parse_args()

    try:
        model_path = MODEL_PATH_DICT[args.model]
    except KeyError:
        print(f"Model '{args.model}' not found.")
        sys.exit(1)

    task_list = get_tasks(args.task)
    samples_num = 1

    llm = Llama(
        model_path=model_path,
        # top_k=10,
        top_p=args.top_p,
        temperature=args.temperature,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=8192,
        max_tokens=args.max_tokens
    )
    if args.test:
        print("Running in test mode.")
        print("task list:")
        print(task_list)
        print("testing llama cpp:")
        system_msg = "You are a story writing assistant."
        full_prompt = "Write a story about llamas."
        system_prompt = system_msg
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_prompt},
        ]
        resp = llm.create_chat_completion(
            messages=msgs
        )
        print(resp)

        content = resp["choices"][0]["message"]["content"]
        content = clean_thinking_prompt(content)
        print(content)
        sys.exit(0)
    else:
        print("Running in production mode.")
    # start_from = 120
    # n=0
    for task in tqdm(task_list):
        # if n < start_from:
        #     n += 1
        #     continue
        for sample in range(1, samples_num + 1):
            prompt_file_prefix = f"{BUILD_PATH}/{task}/{task}_sample{sample:02d}"
            full_prompt, system_msg = load_prompt(prompt_file_prefix)

            msgs = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": full_prompt,
                },
            ]
            resp = llm.create_chat_completion(
                messages=msgs
            )

            resp_str = resp["choices"][0]["message"]["content"]
            resp_str = clean_thinking_prompt(resp_str)

            with open(f"{prompt_file_prefix}_response.txt", "w") as f:
                f.write(resp_str)
            # print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
