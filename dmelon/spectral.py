#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Sun Aug 2 08:40:23 2020
"""

import math

import numpy as np


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
    Ff = np.arange(0, 1 + 1e-9, 2 / N)
    window = np.zeros((len(Ff), 1))
    for i in range(len(Ff)):
        window[i] = coef[0] + 2 * np.sum(
            coef[1:] * np.cos(np.arange(1, len(coef)) * np.pi * Ff[i])
        )
    return window, Ff


# %%
def spectral_filtering(x, window):
    Nx = len(x)
    Cx = np.fft.fft(x)
    Cx = Cx[: (math.floor(Nx / 2)) + 1]
    CxH = Cx * window[:, 0]
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
