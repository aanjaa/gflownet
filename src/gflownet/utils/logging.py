from collections import defaultdict

def prepend_keys(dictionary, prefix):
    return {prefix + "_" + key: value for key, value in dictionary.items()}


def average_values_across_dicts(dicts):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for d in dicts:
        for key, value in d.items():
            totals[key] += value
            counts[key] += 1

    return {key: totals[key] / counts[key] for key in totals}