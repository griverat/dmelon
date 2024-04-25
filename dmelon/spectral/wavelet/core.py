"""
Most of the code here was adaptated from the work done by
Evgeniya Predybaylo in 2014, followed by Michael von Papen
(FZ Juelich, INM-6) in 2018, based on the original work of
Christopher Torrence and Gilbert P.Compo.

The idea behind this adaptation is to include as many modern
python features as possible aiming to have an xarray integration
"""

from typing import Optional

import numpy as np
from scipy.optimize import fminbound
from scipy.special._ufuncs import gamma, gammainc


def wavelet(
    Y: np.array,
    dt: float,
    pad: bool = False,
    dj: float = 1 / 4,
    s0: Optional[float] = None,
    J1: Optional[float] = None,
    mother: str = "MORLET",
    param=-1,
    freq=None,
):
    """
    Wavelet transform of a time series
    """
    n1 = len(Y)

    if s0 is None:
        s0 = 2 * dt
    if J1 is None:
        J1 = np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj)

    # construct time series to analyze, pad if necessary
    x = Y - np.mean(Y)
    if pad is True:
        # power of 2 nearest to N
        base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)
        nzeros = (2 ** (base2 + 1) - n1).astype(int)
        x = np.concatenate((x, np.zeros(nzeros)))
    n = len(x)

    # construct wavenumber array used in transform [Eqn(5)]
    kplus = np.arange(0, n // 2 + 1)
    kminus = np.arange((n - 1) // 2 * -1, 0)
    k = np.concatenate((kplus, kminus)) * 2 * np.pi / (n * dt)

    # compute FFT of the (padded) time series
    f = np.fft.fft(x)  # [Eqn(3)]

    # construct SCALE array & empty PERIOD & WAVE arrays
    if mother.upper() == "MORLET":
        if param == -1:
            param = 6.0
        fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param**2))
    elif mother.upper() == "PAUL":
        if param == -1:
            param = 4.0
        fourier_factor = 4 * np.pi / (2 * param + 1)
    elif mother.upper() == "DOG":
        if param == -1:
            param = 2.0
        fourier_factor = 2 * np.pi * np.sqrt(2.0 / (2 * param + 1))
    else:
        fourier_factor = np.nan

    if freq is None:
        scale = s0 * 2.0 ** (np.arange(0, J1 + 1) * dj)
        freq = 1.0 / (fourier_factor * scale)
        period = 1.0 / freq
    else:
        scale = 1.0 / (fourier_factor * freq)
        period = 1.0 / freq

    daughter, fourier_factor, coi, _ = wave_bases(mother, k, scale, param)
    # wavelet transform[Eqn(4)]
    wave = np.fft.ifft(f * daughter)

    # COI [Sec.3g]
    coi = (
        coi
        * dt
        * np.concatenate(
            (
                np.insert(np.arange(int((n1 + 1) / 2) - 1), [0], [1e-5]),
                np.insert(np.flipud(np.arange(0, int(n1 / 2) - 1)), [-1], [1e-5]),
            ),
        )
    )
    wave = wave[:, :n1]  # get rid of padding before returning

    return wave, period, scale, coi


def wave_bases(
    mother: str,
    k: np.array,
    scale: np.array,
    param: Optional[float] = None,
):
    """
    Compute the wavelet function as a function of Fourier frequency,
    """
    n = len(k)
    kplus = np.array(k > 0.0, dtype=float)
    scale = scale[..., np.newaxis]
    k = k[np.newaxis, ...]

    if mother == "MORLET":  # -----------------------------------  Morlet
        if param is None:
            param = 6.0

        # calc psi_0(s omega) from Table 1
        expnt = -((scale * k - param) ** 2) / 2.0 * kplus
        norm = np.sqrt(scale * k[0][1]) * (np.pi ** (-0.25)) * np.sqrt(n)
        daughter = norm * np.exp(expnt) * kplus  # Heaviside step function
        # Scale-->Fourier [Sec.3h]
        fourier_factor = (4 * np.pi) / (param + np.sqrt(2 + param**2))
        dofmin = 2  # Degrees of freedom

    elif mother == "PAUL":  # --------------------------------  Paul
        if param is None:
            param = 4.0

        # calc psi_0(s omega) from Table 1
        expnt = -scale * k * kplus
        norm_bottom = np.sqrt(param * np.prod(np.arange(1, (2 * param))))
        norm = np.sqrt(scale * k[0][1]) * (2**param / norm_bottom) * np.sqrt(n)
        daughter = norm * ((scale * k) ** param) * np.exp(expnt) * kplus
        fourier_factor = 4 * np.pi / (2 * param + 1)
        dofmin = 2

    elif mother == "DOG":  # --------------------------------  DOG
        if param is None:
            param = 2.0

        # calc psi_0(s omega) from Table 1
        expnt = -((scale * k) ** 2) / 2.0
        norm = np.sqrt(scale * k[0][1] / gamma(param + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j**param) * ((scale * k) ** param) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2.0 / (2 * param + 1))
        dofmin = 1

    else:
        print("Mother must be one of MORLET, PAUL, DOG")

    coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
    return daughter, fourier_factor, coi, dofmin


def wave_signif(
    Y,
    dt,
    scale,
    sigtest=0,
    lag1=0.0,
    siglvl=0.95,
    dof=None,
    mother="MORLET",
    param=None,
    gws=None,
):
    """
    Significance testing for the 1-d wavelet transform
    """
    n1 = len(np.atleast_1d(Y))
    J1 = len(scale) - 1
    dj = np.log2(scale[1] / scale[0])

    if n1 == 1:
        variance = Y
    else:
        variance = np.std(Y) ** 2

    # get the appropriate parameters [see Table(2)]
    if mother == "MORLET":  # ----------------------------------  Morlet
        empir = [2.0, -1, -1, -1]
        if param is None:
            param = 6.0
            empir[1:] = [0.776, 2.32, 0.60]
        k0 = param
        # Scale-->Fourier [Sec.3h]
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0**2))
    elif mother == "PAUL":
        empir = [2, -1, -1, -1]
        if param is None:
            param = 4
            empir[1:] = [1.132, 1.17, 1.5]
        m = param
        fourier_factor = (4 * np.pi) / (2 * m + 1)
    elif mother == "DOG":  # -------------------------------------Paul
        empir = [1.0, -1, -1, -1]
        if param is None:
            param = 2.0
            empir[1:] = [3.541, 1.43, 1.4]
        elif param == 6:  # --------------------------------------DOG
            empir[1:] = [1.966, 1.37, 0.97]
        m = param
        fourier_factor = 2 * np.pi * np.sqrt(2.0 / (2 * m + 1))
    else:
        print("Mother must be one of MORLET, PAUL, DOG")

    period = scale * fourier_factor
    dofmin = empir[0]  # Degrees of freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor

    freq = dt / period  # normalized frequency

    if gws is not None:  # use global-wavelet as background spectrum
        fft_theor = gws
    else:
        # [Eqn(16)]
        fft_theor = (1 - lag1**2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1**2)
        fft_theor = variance * fft_theor  # include time-series variance

    signif = fft_theor
    if dof is None:
        dof = dofmin

    if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
        dof = dofmin
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = fft_theor * chisquare  # [Eqn(18)]
    elif sigtest == 1:  # time-averaged significance
        if len(np.atleast_1d(dof)) == 1:
            dof = np.zeros(J1) + dof
        dof[dof < 1] = 1
        # [Eqn(23)]
        dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale) ** 2)
        dof[dof < dofmin] = dofmin  # minimum DOF is dofmin
        for a1 in range(0, J1 + 1):
            chisquare = chisquare_inv(siglvl, dof[a1]) / dof[a1]
            signif[a1] = fft_theor[a1] * chisquare
    elif sigtest == 2:  # time-averaged significance
        if len(dof) != 2:
            print("ERROR: DOF must be set to [S1,S2]," " the range of scale-averages")
        if Cdelta == -1:
            print(
                "ERROR: Cdelta & dj0 not defined"
                " for " + mother + " with param = " + str(param),
            )

        s1 = dof[0]
        s2 = dof[1]
        avg = np.logical_and(scale >= 2, scale < 8)  # scales between S1 & S2
        navg = np.sum(np.array(np.logical_and(scale >= 2, scale < 8), dtype=int))
        if navg == 0:
            print("ERROR: No valid scales between " + s1 + " and " + s2)
        Savg = 1.0 / np.sum(1.0 / scale[avg])  # [Eqn(25)]
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.0)  # power-of-two midpoint
        dof = (dofmin * navg * Savg / Smid) * np.sqrt(
            1 + (navg * dj / dj0) ** 2,
        )  # [Eqn(28)]
        fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
    else:
        print("ERROR: sigtest must be either 0, 1, or 2")

    return signif


def chisquare_inv(P, V):
    """
    Inverse of the Chi-square distribution function
    """

    if (1 - P) < 1e-4:
        print("P must be < 0.9999")

    if P == 0.95 and V == 2:  # this is a no-brainer
        X = 5.9915
        return X

    MINN = 0.01  # hopefully this is small enough
    MAXX = 1  # actually starts at 10 (see while loop below)
    X = 1
    TOLERANCE = 1e-4  # this should be accurate enough

    while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
        MAXX = MAXX * 10.0
        # this calculates value for X, NORMALIZED by V
        X = fminbound(chisquare_solve, MINN, MAXX, args=(P, V), xtol=TOLERANCE)
        MINN = MAXX

    X = X * V  # put back in the goofy V factor

    return X  # end of code


def chisquare_solve(XGUESS, P, V):
    """
    Solve for the Chi-square function
    """

    PGUESS = gammainc(V / 2, V * XGUESS / 2)  # incomplete Gamma function

    PDIFF = np.abs(PGUESS - P)  # error in calculated P

    TOL = 1e-4
    if PGUESS >= 1 - TOL:  # if P is very close to 1 (i.e. a bad guess)
        PDIFF = XGUESS  # then just assign some big number like XGUESS

    return PDIFF
