"""
Universal parser for yaml config files.
"""

import yaml
import json
from collections import OrderedDict
import logging

__all__ = ["read_config"]


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Function to load YAML file using an OrderedDict
    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def read_config(path):
    """Read the schedule from file"""
    with open(path, 'r') as stream:
        try:
            sched_dict = yaml_ordered_load(stream)
            return sched_dict
        except yaml.YAMLError as exc:
            logging.error(f"Fatal parsing error while parsing the schedule configuration file {path}")
            raise

def write_config(config, path):
    """Write the config used for the experiment into a file"""
    with open(path, 'w') as stream:
        dumped_config = yaml.dump(config, stream)
        print("Dumped config is ", dumped_config)