# VerilogEval Overview

This is an evaluation harness for the VerilogEval problem solving dataset originally described in the paper "[VerilogEval: Evaluating Large Language Models for Verilog Code Generation](https://arxiv.org/abs/2309.07544)," published in 2023. In August 2024, this repository was revised to cover specification-to-RTL tasks in addition to the original code completion task, add in-context learning examples to prompts, and categorize common iverilog failures. Please see the related apaper "[Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks](https://arxiv.org/abs/2408.11053)," published in 2024.

## Ways to run manual model

1. Setup the parameters in `configurations.sh` (the parameters are explained in [Configuration](#configuration))
2. If using `vllm` for inference, serve the model (e.g. `vllm serve /path/to/model --enable-reasoning --reasoning-parser deepseek_r1`) before starting the benchmark
3. Execute `./run_all.sh`

### Configuration

```sh
export task="code-complete-iccad2023" # [code-complete-iccad2023, spec-to-rtl]
export model="manual-qwen2.5-coder-7b" # model name in `scripts/sv-generate`
export shots=0 # 0~3 (in context learning by given examples)
export samples=3 # number of samples to generate
export temperature=0.8
export top_p=0.95
export max_tokens=1024
export model_runner="transformer" # transformer (pipe) or vllm

# reasoning model may work better without system prompt
# the original system prompt will be prepended to the user prompt if the option is set to True
export remove_system_prompt="False"

export IVERILOG_PATH="/path/to/iverilog"
```

### Add new local model

1. In `scripts/sv-generate`, add a new model name to the `manual_models` list (at line 99).
2. Add the model and its path to `manual_models/model-path-dict.json`

## Citation

For this VerilogEval v2, please cite the following paper:

```
@misc{pinckney2024revisitingverilogevalnewerllms,
      title={Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks},
      author={Nathaniel Pinckney and Christopher Batten and Mingjie Liu and Haoxing Ren and Brucek Khailany},
      year={2024},
      eprint={2408.11053},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2408.11053},
}
```

For the original VerilogEval v1, please use:

```
@inproceedings{liu2023verilogeval,
  title={{VerilogEval:} Evaluating Large Language Models for Verilog Code Generation},
  author={Liu, Mingjie and Pinckney, Nathaniel and Khailany, Brucek and Ren, Haoxing},
  booktitle={2023 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  year={2023}
}
```
