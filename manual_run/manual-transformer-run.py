from re import fullmatch
from transformer_model import SimpleModelInference as SimpleModelInference_transformer
from vllm_model import SimpleModelInference as SimpleModelInference_vllm
from tqdm import tqdm
import re
import sys
import argparse
from constants import MODEL_PATH_DICT, BUILD_PATH, DATASET_PATH_PREFIX

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

def main():
    #  -v --verbose        Display the prompt
    #  -h --help           Display this message
    #  -m --model          LLM model to use (default: gpt-3.5-turbo)
    #  -t --temperature    LLM model temperature (default: 0.85)
    #  -p --top-p          LLM model top_p (default: 0.95)
    #  -n --max-tokens     LLM model max_tokens (default: 1024)
    #     --task           Task to prompt (default: code-complete-iccad2023)
    # --add_system_prompt
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
    argparser.add_argument("--remove_system_prompt", type=str, choices=["True", "False"], default="False")
    argparser.add_argument("--model_runner", type=str, choices=["transformer", "vllm"], default="transformer")

    args = argparser.parse_args()

    if args.model_runner == "transformer":
        SimpleModelInference = SimpleModelInference_transformer
        print("Using transformer model runner")
    elif args.model_runner == "vllm":
        SimpleModelInference = SimpleModelInference_vllm
        print("Using vllm model runner")
    else:
        raise ValueError(f"Invalid model runner: {args.model_runner}")

    try:
        model_path = MODEL_PATH_DICT[args.model]
    except KeyError:
        print(f"Model '{args.model}' not found.")
        sys.exit(1)

    task_list = get_tasks(args.task)
    samples_num = args.samples
    llm = SimpleModelInference(model_path)
    remove_system_prompt = True if args.remove_system_prompt == "True" else False
    add_system_prompt = not remove_system_prompt
    if args.test:
        print("Running in test mode.")
        print("task list:")
        print(task_list)
        print("testing llama cpp:")
        system_msg = "You are a story writing assistant."
        user_msg = "Write a story about llamas."

        if add_system_prompt:
            msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        else:
            msgs = [
                {"role": "user", "content": system_msg + '\n\n' + user_msg},
            ]

        resp = llm.inference(
            messages=msgs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            remove_thinking=True
        )
        print(resp)
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

            if add_system_prompt:
                msgs = [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": full_prompt,
                    },
                ]
            else:
                msgs = [
                    {
                        "role": "user",
                        "content": system_msg + '\n\n' + full_prompt,
                    },
                ]

            resp = llm.inference(
                messages=msgs,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                remove_thinking=True
            )

            resp_str = resp

            with open(f"{prompt_file_prefix}_response.txt", "w") as f:
                f.write(resp_str)
            # print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
