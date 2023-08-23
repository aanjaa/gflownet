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


def replace_dict_key(dictionary, key, value):
    keys = key.split(".")
    current_dict = dictionary

    for k in keys[:-1]:
        if k in current_dict:
            current_dict = current_dict[k]
        else:
            raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    last_key = keys[-1]
    if last_key in current_dict:
        current_dict[last_key] = value
    else:
        raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    return dictionary


def change_config(config,changes_config):
    for key, value in changes_config.items():
        config = replace_dict_key(config, key, value)
    return config