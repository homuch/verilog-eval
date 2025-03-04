import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleModelInference:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")

    def inference(self, messages, temperature=0, top_p=0.95, max_tokens=1024, remove_thinking=True):
        """
        Generates text from the model given a list of message dictionaries.

        Args:
            messages (list): A list of dicts, where the first message must have role == 'system'.
            temperature (float): Sampling temperature. 0 means greedy (no sampling).
            top_p (float): Nucleus sampling parameter (only used if temperature > 0).
            max_tokens (int): Maximum number of new tokens to generate.
            remove_thinking (bool): If True, strips out any text after '</think>'.

        Returns:
            str: The generated text from the model.
        """

        # Determine sampling parameters
        do_sample = (temperature != 0)
        top_p = top_p if do_sample else None
        temperature = temperature if do_sample else None

        # Ensure first message is 'system'
        assert messages[0]["role"] == "system"

        # Prepare inputs (these are presumably custom methods in your code)
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        # Generate output
        output = self.model.generate(
            inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )

        # Decode
        generated_text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # Attempt to remove the initial system prompt from the final output
        # by splitting on the system content and slicing accordingly.
        system_content = messages[0]["content"]
        # We split the *generated* text on the system content ...
        parts_after_system = generated_text.split(system_content)
        # ... and also split the *input* text on that same content to see how long it was.
        parts_input = input_text.split(system_content)
        if len(parts_after_system) > 1 and len(parts_input) > 1:
            # Slice off the portion corresponding to the system prompt length
            generated_text = parts_after_system[-1][len(parts_input[-1]):]

        # Optionally remove any '<think>' block
        if remove_thinking:
            generated_text = self._remove_thinking(generated_text)

        return generated_text

    def _remove_thinking(self, in_text):
        """
        Splits on '</think>' and returns only the part after it,
        effectively discarding any "thinking" text in the generation.
        """
        return in_text.split("</think>")[-1]
