"""
Helper functions that fit into a more general category
"""

import json
import os
from typing import Optional


def check_folder(base_path: str, name: Optional[str] = None):
    """
    Create a folder if it does not exists
    """
    if name is not None:
        out_path = os.path.join(base_path, str(name))
    else:
        out_path = base_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)


def load_json(path: str):
    """
    Load the contents of a json file into a python dictionary
    """
    with open(path) as f:
        content = json.load(f)
    return content
