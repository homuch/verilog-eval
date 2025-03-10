import torch
from transformers import pipeline

class SimpleModelInference:
    def __init__(self, model_path):
        self.pipe = pipeline("text-generation", model=model_path)

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
        top_k = None
        generate_kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_tokens
        }
        outputs = self.pipe(messages, **generate_kwargs)
        assert outputs[0]["generated_text"][-1]["role"] == "assistant"
        generated_text = outputs[0]["generated_text"][-1]["content"]

        generated_text = generated_text.replace("<|EOT|>", "").strip()

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
