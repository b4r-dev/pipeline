# flake8: noqa


# standard library
import os
import re
import sys
from pathlib import Path


# dependent packages
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


# special path objects
def _get_path_from(env_var: str, default: str = ".") -> Path:
    """Get a resolved path from an environment variable."""
    path = Path(os.environ.get(env_var, default)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA = _get_path_from("DATA_PATH", "./data")
PRODUCTS = _get_path_from("PRODUCTS_PATH", "./products")
LDATA = _get_path_from("LDATA_PATH", "./large/data")
LPRODUCTS = _get_path_from("LPRODUCTS_PATH", "./large/products")
