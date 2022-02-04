import numpy as np
import xarray as xr

from .core import wave_signif, wavelet


def ar1nv(x):
    x = x.data
    N = x.size
    x = x - x.mean()
    c0 = x.dot(x) / N
    c1 = x[:-1].dot(x[1:]) / (N - 1)
    g = c1 / c0
    a = np.sqrt((1 - g**2) * c0)
    return g, a


def wt(
    x: xr.DataArray,
    dt=1,
    pad=True,
    dj=1 / 12,
    s0=None,
    mother="MORLET",
    AR1="auto",
    plot=True,
):
    if s0 is None:
        s0 = 2 * dt
    if AR1 == "auto":
        AR1, _ = ar1nv(x)
    J1 = np.round(np.log2((x.size * 0.17 * 2 * dt) / 2) / (1 / 12))
    wave, period, scale, coi = wavelet(
        x,
        dt=dt,
        pad=pad,
        dj=dj,
        s0=s0,
        J1=J1,
        mother="MORLET",
    )

    power = (np.abs(wave)) ** 2
    signif = wave_signif(1, dt=dt, scale=scale, lag1=AR1, dof=2)
    sig95 = signif[:, np.newaxis].dot(np.ones(x.size)[np.newaxis, :])
    variance = x.var(ddof=1).data
    sig95 = power / (variance * sig95)

    if plot is True:
        import cmocean as cmo
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        mpl.style.use("default")

        cmap = cmo.cm.thermal

        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
        ax.contourf(
            x.time.data,
            period,
            np.log2(np.abs(power / variance)),
            cmap=cmap,
            levels=20,
            vmin=-9,
            vmax=9,
        )
        ax.contour(x.time.data, period, sig95, [-99, 1], colors="k")
        ax.fill_between(
            x.time.data,
            coi * 0 + period[-1],
            coi,
            facecolor="none",
            edgecolor="#00000040",
            hatch="x",
        )
        ax.plot(x.time.data, coi, "k")
        ax.set_yscale("log", base=2, subs=None)
        ax.set_ylim([np.min(period), np.max(period)])
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylabel("Period")
        ax.set_xlabel("Time")
