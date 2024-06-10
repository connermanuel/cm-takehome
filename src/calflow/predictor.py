from calflow.llama import get_prediction_llama
from calflow.openai import get_prediction_openai


def create_decision_function_length(length: int) -> function:
    """Creates a decision function based on the length of the utterance."""

    def decision_function(utterance: str) -> bool:
        if len(utterance) > length:
            return True
        return False

    return decision_function


def get_prediction(
    query: str, system_prompt: str, few_shot_examples: list, decision_function: function
) -> str:
    """Given a prompt, returns the prediction from GPT3.5 Turbo.

    Args:
        query: The utterance that we want to make a prediction for
        system_prompt: The system prompt providing instructions to the model
        few_shot_examples: A list of (utterance, plan) pairs to be used as few shot examples
        decision_function: A function that takes in an utterance and decides whether or not the large model should be used.

    Returns:
        The predicted plan from the utterance.
    """
    if decision_function(query):
        return get_prediction_openai(query, system_prompt, few_shot_examples)
    else:
        return get_prediction_llama(query, system_prompt, few_shot_examples)
