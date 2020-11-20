#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import statistics

import numpy as np
import numpy.fft as fft
import xarray as xr
from scipy.ndimage import convolve1d
from scipy.signal import stft


# %%
def lanczos_filter_coef(Cf, M):
    hkcs = lowpass_cosine_filter_coef(Cf, M)
    sigma = np.sin(np.pi * np.arange(1, M + 1) / M) / (
        np.pi * np.arange(1, M + 1) / M
    )
    sigma = np.insert(sigma, 0, 1)
    hkB = hkcs * sigma
    hkA = -hkB
    hkA[0] = hkA[0] + 1
    coef = np.array([hkB, hkA])
    return coef


# %%
def lowpass_cosine_filter_coef(Cf, M):
    sig = np.sin(np.pi * np.arange(1, M + 1) * Cf) / (
        np.pi * np.arange(1, M + 1) * Cf
    )
    coef = Cf * np.insert(sig, 0, 1)
    return coef


# %%
def spectral_window(coef, N):
    Ff = np.atleast_2d(np.arange(0, 1 + 1e-9, 2 / N)).T
    window = coef[0] + 2 * np.sum(
        np.atleast_2d(coef[1:])
        * np.cos(np.atleast_2d(np.arange(1, len(coef))) * np.pi * Ff),
        axis=-1,
    )
    return window, Ff


# %%
def spectral_filtering(x, window):
    Nx = len(x)
    Cx = np.fft.fft(x)
    Cx = Cx[: (math.floor(Nx / 2)) + 1]
    CxH = Cx * window
    ext = np.conj(CxH[Nx - len(CxH) + 1 : 0 : -1])
    CxH = np.concatenate((CxH, ext))
    y = np.real(np.fft.ifft(CxH))
    return y, Cx


# %%
def lanczosfilter(X, Cf, dT=1, M=100, kind="low", *args):
    if np.isnan(X).all():
        return np.full_like(X, np.nan, dtype=np.float)
    kind_val = {"high": 1, "low": 0}
    Nf = 1 / (2 * dT)
    Cf = Cf / Nf
    coef = lanczos_filter_coef(Cf, M)[kind_val[kind]]
    window, Ff = spectral_window(coef, len(X))
    Ff = Ff * Nf
    X = np.nan_to_num(X, np.nanmean(X))
    y, Cx = spectral_filtering(X, window)
    return y


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
    _fft = fft.fft(data, n=Nx)
    nsegs, _fft = stft(
        _fft, axis=0, nperseg=Nt, window=window, nfft=nfft, **stft_kwargs
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
    power = xr.DataArray(
        power, coords=[("frequency", yax), ("wavenumber", xax)]
    )

    if psmooth is None:
        return power
    elif psmooth is True:
        smooth_power = dict(nt=20, nx=40)
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
