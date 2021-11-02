"""
Plotting module that contains most boilerplate code
I use for my plots
"""
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

SD_BORDER = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_countries",
    scale="50m",
    facecolor="white",
    edgecolor="black",
    linewidth=1.5,
)

HQ_BORDER = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_countries",
    scale="10m",
    facecolor="white",
    edgecolor="black",
    linewidth=1.5,
)


def format_latlon(
    ax,
    proj,
    latlon_bnds=(-180, 180, -90, 90),
    lon_step=20,
    lat_step=10,
    nformat="g",
):
    """
    Format geoaxes nicely
    """
    (ilon, flon, ilat, flat) = latlon_bnds

    lon_formatter = LongitudeFormatter(number_format=nformat)
    lat_formatter = LatitudeFormatter(number_format=nformat)

    ax.set_xticks(np.arange(ilon, flon, lon_step), crs=proj)
    ax.set_yticks(np.arange(ilat, flat, lat_step), crs=proj)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    return ax
