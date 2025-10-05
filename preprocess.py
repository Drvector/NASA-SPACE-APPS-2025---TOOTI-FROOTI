"""
preprocess.py
- read raw light curve, detrend, remove outliers, normalize
- create folded views (global + local) around a proposed period
"""

import numpy as np
import pandas as pd
from lightkurve import LightCurve
from scipy.signal import medfilt
from astropy.stats import sigma_clip

def detrend_lc(time, flux, kernel_size=101):
    # simple median filter detrending (replace with robust GP detrend for production)
    trend = medfilt(flux, kernel_size=kernel_size)
    detrended = flux - trend + np.median(trend)
    return detrended

def remove_outliers(flux, sigma=5.0):
    clipped = sigma_clip(flux, sigma=sigma)
    return np.ma.filled(clipped, np.nan)

def phase_fold(time, flux, period, t0, epoch_len=1.0):
    """
    Phase-fold time/flux on period around t0.
    Returns phase [-0.5, 0.5), folded flux, sorted by phase
    """
    phase = ((time - t0 + 0.5*period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]

def binned_view(phase, flux, bins=200):
    """
    Bin phase-folded data into fixed-size bins and return normalized flux.
    """
    inds = np.linspace(-0.5, 0.5, bins+1)
    digit = np.digitize(phase, inds) - 1
    binned = np.array([np.nanmedian(flux[digit==i]) for i in range(bins)])
    # handle NaNs by interpolation
    nans = np.isnan(binned)
    if np.any(nans):
        binned[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), binned[~nans])
    # normalize
    binned = (binned - np.nanmedian(binned)) / np.nanstd(binned)
    return binned

def make_global_local_views(time, flux, period, t0, global_bins=400, local_bins=200, local_window=0.2):
    phase, f = phase_fold(time, flux, period, t0)
    global_view = binned_view(phase, f, bins=global_bins)
    # local window centered on 0 of width local_window*period
    mask = np.abs(phase) <= (local_window/2.0)
    local_phase = phase[mask]
    local_flux = f[mask]
    # re-bin local to fixed size
    # shift phase to -1..1 across local region to bin
    if len(local_phase) < 10:
        # if too few points, pad from global view
        local_view = np.interp(np.linspace(-local_window/2, local_window/2, local_bins), phase, f)
    else:
        local_view = binned_view(local_phase, local_flux, bins=local_bins)
    return global_view.astype(np.float32), local_view.astype(np.float32)
