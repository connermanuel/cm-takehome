from dataflow.core.lispress import parse_lispress

def flatten_list(l: list) -> list:
    """
    Given a list which may contain nested lists inside it, return a list that contains all of the elements
    in flattened order as would be returned by depth first search.
    """

    result = []
    for entry in l:
        if isinstance(entry, list):
            result.extend(flatten_list(entry))
        else:
            result.append(entry)
    return result

def metric_f1(label: str, prediction: str) -> float:
    """
    Returns the F1 score of a predicted plan based on ground truth plan.

    Args:
        label: string representing ground truth plan
        prediction: string representing predicted plan
    
    Returns:
        An F1 score ranging from 0 to 1 assessing the prediction.
    """

    try:
        label = parse_lispress(label)
        prediction = parse_lispress(prediction)
    except (AssertionError, IndexError):
        return 0

    label_set = set(flatten_list(label))
    prediction_set = set(flatten_list(prediction))
    tp = label_set.intersection(prediction_set)

    return (2 * len(tp)) / (len(label_set) + len(prediction_set))