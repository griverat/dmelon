"""
Module that contains algorithms mostly using the power spectra
"""

import numpy as np
import numpy.fft as fft
import xarray as xr
from scipy.ndimage import convolve1d
from scipy.signal import stft

from .. import statistics


def get_dispersion(
    data,
    Nx,
    Nt,
    xres,
    tres,
    window=None,
    nfft=None,
    **stft_kwargs,
):
    """
    Get the dispersion graph of series of maps
    """
    _fft = fft.fft(data, n=Nx)
    nsegs, _fft = stft(
        _fft,
        axis=0,
        nperseg=Nt,
        window=window,
        nfft=nfft,
        **stft_kwargs,
    )[1:]
    _fft = fft.fftshift(_fft * window.sum())[Nt // 2 :, ...]
    _fft = np.conjugate(_fft) * _fft
    _fft = _fft.real
    print(f"{nsegs.size=}")
    _fft = _fft.mean(axis=-1)
    if nfft is not None:
        Nt = nfft
    power = 2 * (_fft) * xres * tres / (Nx * Nt)
    yax = fft.fftshift(fft.fftfreq(Nt, tres))
    xax = fft.fftshift(fft.fftfreq(Nx, xres))
    return power, xax, yax[Nt // 2 :, ...]


def compute_power(xdata, Nx, Nt, xres, tres, window, noverlap, psmooth=True):
    """
    Filter the power spectra in the wavenumber-frequency space
    """
    power, xax, yax = get_dispersion(
        xdata,
        Nx,
        Nt,
        xres,
        tres,
        window=window,
        noverlap=noverlap,
    )
    dof = statistics.edof(xdata.sizes["time"], window, noverlap)
    print(f"{dof=}")
    power = xr.DataArray(power, coords=[("frequency", yax), ("wavenumber", xax)])

    if psmooth is None:
        return power
    elif psmooth is True:
        psmooth = dict(nt=20, nx=40)
    elif not isinstance(psmooth, dict):
        print(
            "psmooth needs to be a dictionary with keys 'nt' and 'nx' ",
            "and int values",
        )
    kernel = np.array([1, 2, 1])
    kernel = kernel / kernel.sum()
    kernel = xr.DataArray(kernel, dims=["kx"])
    smooth_power = power.data.T
    for i in range(psmooth["nt"]):
        smooth_power = convolve1d(smooth_power, kernel.data)
    smooth_power = smooth_power.T
    for i in range(psmooth["nx"]):
        smooth_power = convolve1d(smooth_power, kernel.data)
    smooth_power = xr.DataArray(smooth_power, coords=power.coords)
    return power, smooth_power
