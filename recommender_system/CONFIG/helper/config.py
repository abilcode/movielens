import os
import yaml

def load_config(config_name):
    """
    Load a configuration file in YAML format.

    :param config_name: Name of the configuration file.
    :return: Dictionary containing the configuration.
    """

    CONFIG_PATH = '../CONFIG'
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config