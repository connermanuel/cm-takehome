import os

from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, logging, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity(logging.CRITICAL)

BATCH_SIZE = 1
MODEL_PATH = "TheBloke/Llama-2-7b-Chat-GPTQ"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
MODEL = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    use_safetensors=True,
    trust_remote_code=True,
    device="cuda:0",
    use_triton=True,
    quantize_config=None,
)
PIPE = pipeline(
    "text-generation",
    model=MODEL,
    tokenizer=TOKENIZER,
    max_new_tokens=512,
    repetition_penalty=1.15,
    batch_size=BATCH_SIZE,
)


def format_llama_prompt(query: str, system_prompt: str, few_shot_examples: list) -> str:
    """Returns a formatted version of the prompt with a system message included."""

    formatted_few_shot_examples = ""
    for utterance, plan in few_shot_examples:
        formatted_few_shot_examples += f"<s>[INST] {utterance} [/INST] {plan} </s>\n"

    prompt = (
        f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>\n\n"
        f"{formatted_few_shot_examples[10:]}"
        f"<s>[INST] {query} [/INST]"
    )
    return prompt


def get_prediction_llama(
    query: str, system_prompt: str, few_shot_examples: list
) -> str:
    """Given a prompt, returns the prediction from llama quantized.

    Args:
        query: The utterance that we want to make a prediction for
        system_prompt: The system prompt providing instructions to the model
        few_shot_examples: A list of (utterance, plan) pairs to be used as few shot examples

    Returns:
        The predicted plan from the utterance.
    """
    formatted_prompt = format_llama_prompt(query, system_prompt, few_shot_examples)
    result = PIPE(formatted_prompt)[0]["generated_text"][len(formatted_prompt) :]
    count_open = result.count("(")
    while result.count(")") > count_open:
        result = result[:-1]
    return result
