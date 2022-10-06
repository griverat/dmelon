"""
Helper functions that fit into a more general category
"""

import json
import os
import warnings
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.tools import sjoin


def check_folder(base_path: str, name: Optional[str] = None) -> None:
    """
    Create a folder if it does not exists
    """
    if name is not None:
        out_path = os.path.join(base_path, str(name))
    else:
        out_path = base_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)


def load_json(path: str) -> dict:
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
) -> gpd.GeoDataFrame:
    """
    Filter DataFrame by their spatial location within a
    GeoDataFrame
    """
    argo_geodf = gpd.GeoDataFrame(
        pandas_df,
        geometry=gpd.points_from_xy(pandas_df.longitude, pandas_df.latitude, crs=crs),
    )

    # Return spatial join to filer out values outside the shapefile
    return sjoin(argo_geodf, shape_df, predicate="within", how="inner")


# Piece of code from xmip that I am testing
# not sure how it affects other cmip6 models


def _interp_nominal_lon(lon_1d):
    x = np.arange(len(lon_1d))
    idx = np.isnan(lon_1d)
    # the periodicity of the coordinates should be the length of the array
    # not a fixed 360 since the base coordinates is constructed as a range
    # from 0 to the length of the array
    # this is what I am testing
    return np.interp(x, x[~idx], lon_1d[~idx], period=len(lon_1d))


def replace_x_y_nominal_lat_lon(ds):
    """Approximate the dimensional values of x and y with mean lat and lon at the equator"""
    ds = ds.copy()

    def maybe_fix_non_unique(data, pad=False):
        """remove duplicate values by linear interpolation
        if values are non-unique. `pad` if the last two points are the same
        pad with -90 or 90. This is only applicable to lat values"""
        if len(data) == len(np.unique(data)):
            return data
        else:
            # pad each end with the other end.
            if pad:
                if len(np.unique([data[0:2]])) < 2:
                    data[0] = -90
                if len(np.unique([data[-2:]])) < 2:
                    data[-1] = 90

            ii_range = np.arange(len(data))
            _, indicies = np.unique(data, return_index=True)
            double_idx = np.array([ii not in indicies for ii in ii_range])
            # print(f"non-unique values found at:{ii_range[double_idx]})")
            data[double_idx] = np.interp(
                ii_range[double_idx],
                ii_range[~double_idx],
                data[~double_idx],
            )
            return data

    if "x" in ds.dims and "y" in ds.dims:
        # define 'nominal' longitude/latitude values
        # latitude is defined as the max value of `lat` in the zonal direction
        # longitude is taken from the `middle` of the meridonal direction, to
        # get values close to the equator

        # pick the nominal lon/lat values from the eastern
        # and southern edge, and
        eq_idx = len(ds.y) // 2

        nominal_x = ds.isel(y=eq_idx).lon.load()
        nominal_y = ds.lat.max("x").load()

        # interpolate nans
        # Special treatment for gaps in longitude
        nominal_x = _interp_nominal_lon(nominal_x.data)
        nominal_y = nominal_y.interpolate_na("y").data

        # eliminate non unique values
        # these occour e.g. in "MPI-ESM1-2-HR"
        nominal_y = maybe_fix_non_unique(nominal_y)
        nominal_x = maybe_fix_non_unique(nominal_x)

        ds = ds.assign_coords(x=nominal_x, y=nominal_y)
        ds = ds.sortby("x")
        ds = ds.sortby("y")

        # do one more interpolation for the x values, in case the boundary values were
        # affected
        ds = ds.assign_coords(
            x=maybe_fix_non_unique(ds.x.load().data),
            y=maybe_fix_non_unique(ds.y.load().data, pad=True),
        )

    else:
        warnings.warn(
            "No x and y found in dimensions for source_id:%s. This likely means that you forgot to rename the dataset or this is the German unstructured model"
            % ds.attrs["source_id"],
        )
    return ds
