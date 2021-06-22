import numpy as np
import xarray as xr
from eofs.xarray import Eof
from scipy.ndimage import convolve1d

# from scipy.stats import linregress


class ECindex:
    """
    Computes the E and C index according to Takahashi
    """

    def __init__(self, sst_data, **kwargs):
        self.sst_data = sst_data.sel(**kwargs)
        self.clim_period = None
        self.clim = None
        self.anom = None

    def compute_anomaly(self, clim_period={"time": slice("1979-01-01", "2019-12-30")}):
        self.clim_period = self.sst_data.sel(**clim_period)
        self.clim = self.clim_period.groupby("time.month").mean(dim="time")
        self.anom = self.sst_data.groupby("time.month") - self.clim

    def get_pcs(self, corr_factor=[1, -1]):
        coslat = np.cos(np.deg2rad(self.clim_period.lat.data))
        wgts = np.sqrt(coslat)[..., np.newaxis]

        if corr_factor is None:
            corr_factor = [1, 1]
        corr_factor = xr.DataArray(np.array(corr_factor), coords=[("mode", [0, 1])])
        self.solver = Eof(self.clim_period, weights=wgts, center=False)
        clim_std = self.solver.eigenvalues(neigs=2) ** (1 / 2)
        self.anom_pcs = (
            self.solver.projectField(self.sst_anom.sel(lat=slice(-10, 10)), neofs=2)
            * corr_factor
            / clim_std
        )
        kernel = np.array([1, 2, 1])
        kernel = xr.DataArray(kernel / kernel.sum(), dims=["time"])
        self.anom_smooth_pcs = self.xconvolve(self.anom_pcs, kernel, dim="time")

    @staticmethod
    def xconvolve(data, kernel, dim=None):
        res = xr.apply_ufunc(
            convolve1d,
            data,
            kernel,
            input_core_dims=[[dim], [dim]],
            exclude_dims={dim},
            output_core_dims=[[dim]],
        )
        res[dim] = data[dim]
        return res

    def get_index(self):
        cindex = (self.anom_pcs.sel(mode=0) + self.anom_pcs.sel(mode=1)) / (
            2 ** (1 / 2)
        )
        eindex = (self.anom_pcs.sel(mode=0) - self.anom_pcs.sel(mode=1)) / (
            2 ** (1 / 2)
        )
        return eindex, cindex
