"""
Python implementation of the Boulanger & Menkes 1995 paper
"""
import numpy as np
import xarray as xr
from scipy.special import eval_hermite, factorial


def scale_lats(lats, c=2.5, return_scales=False):
    """
    Nondimensionalze latitutes using the following scales:

    .. math::
        L = (c/\beta)^(1/2) \\
        T = 1/(\beta c)^(1/2)

    """
    e_r = 6.37122e6
    omega = 7.2921159e-5
    beta = 2 * omega * np.cos(lats * np.pi / 180) / e_r
    L = np.sqrt(c / beta)
    T = 1 / np.sqrt(beta * c)
    scaled = lats * 111.1949e3 / L
    if return_scales:
        return scaled, (L, T)
    else:
        return scaled


def _nantrapz(y, x=None, dx=1.0, axis=-1):
    """
    Trapezoidal method of integration from numpy with nan support
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = np.nansum((d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0), axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.nansum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret


def hermite_function(n, x):
    """
    Evaluates the hermite function of order n at a point x
    """
    n = np.atleast_2d(n)
    x = np.atleast_2d(x)
    coef = np.sqrt((2 ** n) * factorial(n) * np.sqrt(np.pi)) * np.exp((x ** 2) / 2)
    return eval_hermite(n, x) / coef


def meridional_structures(n, lats):
    """
    Compute the meridional structures using the formulas in BM95

    Parameters
    ----------
    n : array_like
        Meridional modes. This is passed down as the
        order of the underlying Hermite Functions
    lats : array_like
        Array of latitudes
    """
    sclats = scale_lats(lats)
    R = np.empty((2, n) + sclats.shape)
    R_0 = np.sqrt(1 / 2) * hermite_function(0, sclats)
    R[:, 0, :] = np.vstack((R_0, R_0))
    order = np.atleast_2d(np.arange(1, n)).T
    hforward = hermite_function(order + 1, sclats) / (np.sqrt(order + 1))
    hbackward = hermite_function(order - 1, sclats) / np.sqrt(order)
    coef = np.sqrt((order * (order + 1)) / (2 * (2 * order + 1)))
    R[:, 1:, :] = np.stack((hforward - hbackward, hforward + hbackward)) * coef

    R = xr.Dataset(
        {
            "R_u": (["hpoly", "lat"], R[0, :, :]),
            "R_h": (["hpoly", "lat"], R[1, :, :]),
        },
        coords={
            "hpoly": np.arange(n),
            "lat": lats,
            "scaled_lat": (["lat"], sclats),
        },
    )

    return R


class Projection:
    """Projection object that computes the projection vector, wave
    coefficient vector and decomposed sea level as calculated by
    J.-P.Boulanger & C.Menkes (1995).

    It constructs the meridional structures when instantiated.
    """

    def __init__(self, sea_level, nmodes):
        """
        Parameters
        ----------
        sea_level : xarray.DataArray
            Input sea level anomaly field [time, lat, lon] from which to
            compute the meridional decomposition.
        nmods : int
            Number of meridional modes to consider
        """
        self.sea_level = sea_level
        self.R = meridional_structures(nmodes, self.sea_level.lat)
        self.A = self._build_A(self.R.R_h.data)
        self.A_inv = self._minv(self.A)

    @staticmethod
    def _minv(xrobj):
        """
        Inverse matrix operation for xarray data structures
        """
        return xr.apply_ufunc(
            np.linalg.inv,
            xrobj,
            dask="parallelized",
            output_dtypes=[np.float],
        )

    @staticmethod
    def _build_A(Rh):
        """
        Build the A matrix of the sea level decomposition method
        """
        Rh = Rh[..., np.newaxis]
        A = np.trapz(Rh * Rh.T, dx=scale_lats(0.25), axis=1)

        A = xr.DataArray(
            A,
            coords=[
                ("hpoly", np.arange(A.shape[0])),
                ("_hpoly", np.arange(A.shape[0])),
            ],
        )
        return A

    @staticmethod
    def _integrate(xrobj, dim):
        """
        Trapezoidal integration with xarray data structures
        """
        return xr.apply_ufunc(
            _nantrapz,
            xrobj,
            input_core_dims=[[dim]],
            kwargs={"axis": -1, "dx": scale_lats(0.25)},
            dask="parallelized",
            output_dtypes=[np.float],
        )

    def projection_vector(self):
        """
        Build the projection vector `b`
        """
        self.b = self._integrate(
            (self.sea_level.interpolate_na(dim="lon", limit=2) / ((2.5 ** 2) / 9.81))
            * self.R.R_h,
            dim="lat",
        )
        self.b.name = "projection_vector"
        return self.b

    def wave_coefficient_vector(self):
        """
        Build the wave coefficient vector `r`
        """
        self.r = xr.dot(self.A_inv, self.b).rename({"_hpoly": "hpoly"})
        self.r.name = "wave_coefficient_vector"
        return self.r

    def decomposed_sea_level(self):
        """
        Decompose the sea level
        """
        self.h = (self.r * self.R.R_h).drop(["scaled_lat"]).transpose(
            "hpoly",
            "time",
            "lat",
            "lon",
        ) * ((2.5 ** 2) / 9.81)
        self.h.name = "wave_amp"
        return self.h
