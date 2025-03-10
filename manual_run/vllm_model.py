from openai import OpenAI

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

class SimpleModelInference:
    def __init__(self, model_path):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

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

        models = self.client.models.list()
        model_name = models.data[0].id

        output = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )


        # Decode (only preserve output)
        generated_text = output.choices[0].message.content

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
