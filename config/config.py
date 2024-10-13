import json
import os
from pathlib import Path
import random
from easydict import EasyDict as edict


def get_config_regression( model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.

        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """

    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)

    model_dataset_args = config_all[model_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['test_mode'] = 'regression'
    config.update(dataset_args)
    config.update(model_dataset_args)
    config = edict(config)  # use edict for backward compatibility with MMSA v1.0

    return config


