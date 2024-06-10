from openai import OpenAI

client = OpenAI()


def get_prediction_openai(
    query: str, system_prompt: str, few_shot_examples: list
) -> str:
    """Given a prompt, returns the prediction from GPT3.5 Turbo.

    Args:
        query: The utterance that we want to make a prediction for
        system_prompt: The system prompt providing instructions to the model
        few_shot_examples: A list of (utterance, plan) pairs to be used as few shot examples

    Returns:
        The predicted plan from the utterance.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for utterance, plan in few_shot_examples:
        messages += [
            {
                "role": "user",
                "content": utterance,
            },
            {
                "role": "assistant",
                "content": plan,
            },
        ]

    messages += [
        {
            "role": "user",
            "content": query,
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    result = completion.choices[0].message.content
    count_open = result.count("(")
    while result.count(")") > count_open:
        result = result[:-1]
    return result
