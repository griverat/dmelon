#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

HQ_BORDER = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_countries",
    scale="50m",
    facecolor="white",
    edgecolor="grey",
)


def format_latlon(
    ax, proj, latlon_bnds=(-180, 180, -90, 90), lon_step=20, lat_step=10,
):
    (ilon, flon, ilat, flat) = latlon_bnds

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.set_xticks(np.arange(ilon, flon, lon_step), crs=proj)
    ax.set_yticks(np.arange(ilat, flat, lat_step), crs=proj)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    return ax
