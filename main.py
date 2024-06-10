import json

from dataflow.core.lispress import parse_lispress, render_pretty
from tqdm import tqdm

from calflow.metrics import metric_f1
from calflow.predictor import create_decision_function_length, get_prediction

FEW_SHOT_LEN = 20
DATA_PATH = "/home/manuel.conner.g/calflow/data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num128"
OUT_PATH = "/home/manuel.conner.g/calflow/out"

with open(f"{DATA_PATH}/train.jsonl", "r") as f:
    TRAIN_DATA = [json.loads(line) for line in f.readlines()]
with open(f"{DATA_PATH}/val.jsonl", "r") as f:
    VAL_DATA = [json.loads(line) for line in f.readlines()]

SYSTEM_PROMPT = (
    "You will be provided with an utterance that describes a task that should be completed. "
    "Your task is to translate this utterance into a program in Lispress, a Lisp-like serialization format for programs. "
    "Example: \n"
    f"Utterance: {TRAIN_DATA[0]['utterance']}\n"
    f"Program: {render_pretty(parse_lispress(TRAIN_DATA[0]['plan']))}\n"
    "Notice how the program was constructed in relation to the utterance. "
    "The event has two constraints, attendees and recipient, both described based on information in the utterance. "
    "The event needs to be created, then it also needs to be scheduled, hence the two wrappers. "
    "Return ONLY the generated program."
)
FEW_SHOT_EXAMPLES = [
    (entry["utterance"], entry["plan"]) for entry in TRAIN_DATA[1:FEW_SHOT_LEN]
]


def main():
    """Runs the validation set using the indicated number of few shot examples and saves it to the output directory."""
    decision_function = create_decision_function_length(75)

    for entry in tqdm(VAL_DATA):
        predictions = []
        prompt = entry["utterance"]
        label = entry["plan"]
        prediction = get_prediction(
            prompt, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, decision_function
        )
        predictions.append(
            {
                "input": prompt,
                "label": label,
                "prediction": prediction,
                "score": metric_f1(label, prediction),
            }
        )
        with open(f"{OUT_PATH}/predictions-{FEW_SHOT_LEN}.json", "w") as f:
            json.dump(predictions, f)


if __name__ == "__main__":
    main()
