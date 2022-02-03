"""
Helper functions that fit into a more general category
"""

import json
import os
from typing import Optional

import geopandas as gpd
import pandas as pd
from geopandas.tools import sjoin


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


def findPointsInPolys(
    pandas_df: pd.DataFrame,
    shape_df: gpd.GeoDataFrame,
    crs: str = "EPSG:4326",
):
    """
    Filter DataFrame by their spatial location within a
    GeoDataFrame
    """
    argo_geodf = gpd.GeoDataFrame(
        pandas_df,
        geometry=gpd.points_from_xy(pandas_df.longitude, pandas_df.latitude, crs=crs),
    )

    # Make spatial join to filer out values outside the shapefile
    pointInPolys = sjoin(argo_geodf, shape_df, predicate="within", how="inner")
    return pointInPolys
