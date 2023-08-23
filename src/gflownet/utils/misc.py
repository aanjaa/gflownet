import logging
import sys
from collections import defaultdict
import os
from pathlib import Path


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_num_cpus() -> int:
    '''
    In a typical slurm allocation we only have access to a subset of the
    current node's CPUs.  Allocating ray with more CPUs than we actually
    have naturally leads to massive slowdowns in code performance, so
    we should avoid this.

    For any job allocation, Slurm writes a file to /sys listing the
    CPUs on the node which belong to the allocation.  If the code is
    being run on a slurm allocation this function reads the slurm CPU file
    and returns the number of CPUs allocated for the job.  If not on a slurm
    allocation, ray is initialized by default with the number of CPUs available
    on the system.
    '''

    _SLURM_JOB_CPUS_FILENAME = '/sys/fs/cgroup/cpuset/slurm/uid_%s/job_%s/cpuset.cpus'

    if 'SLURM_JOB_ID' not in os.environ:
        return os.cpu_count()

    uid, slurm_job_id = os.getuid(), os.environ['SLURM_JOB_ID']
    fname = Path(_SLURM_JOB_CPUS_FILENAME % (uid, slurm_job_id))

    num_cpus = 0
    with open(fname, 'r') as f:
        line = f.read().replace('\n', '')
        for substr in line.split(','):
            if '-' not in substr:
                num_cpus += 1
                continue

            cpu_nums = list(map(int, substr.split('-')))
            num_cpus += cpu_nums[1] - cpu_nums[0] + 1

    return num_cpus


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


def convert_str_to_bool(args_obj, args_to_convert):
    for arg in args_to_convert:
        # check format
        curr_value = getattr(args_obj, arg)
        try:
            assert curr_value in ['true', 'false']
        except:
            raise ValueError(f' Argument {arg}={curr_value} needs to provided as true or false')
        # set bool format
        bool_value = True if curr_value == 'true' else False
        setattr(args_obj, arg, bool_value)
    return args_obj