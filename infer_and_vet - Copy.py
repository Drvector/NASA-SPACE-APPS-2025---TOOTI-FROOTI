"""
infer_and_vet.py
- load saved model, run inference on new light curves, and perform quick vetting checks
- includes simple transit fit using batman to estimate depth and duration (optional)
"""

import numpy as np
import tensorflow as tf
from model import build_astronet
import batman
from preprocess import make_global_local_views
from scipy.signal import lombscargle
import json

MODEL_PATH = "outputs/astronet_best.h5"

def load_model(weights_path=MODEL_PATH, global_length=400, local_length=200):
    model = build_astronet(global_length=global_length, local_length=local_length)
    model.load_weights(weights_path)
    return model

def quick_period_search(time, flux):
    # very simple period search for demo using Lomb-Scargle (replace with Box Least Squares (BLS))
    freq = np.linspace(0.01, 10, 10000)
    pgram = lombscargle(time, flux - np.nanmedian(flux), 2*np.pi*freq)
    bestfreq = freq[np.argmax(pgram)]
    return 1.0 / bestfreq

def vet_candidate(time, flux, period, t0, depth_est, threshold=5.0):
    # Quick SNR estimate: depth / rms of out-of-transit
    phase, f = ((time - t0 + 0.5*period) % period) / period - 0.5, flux
    oot = f[np.abs(phase) > 0.05]  # out of transit window
    snr = depth_est / np.nanstd(oot)
    return snr

def estimate_depth_duration(time, flux, period, t0):
    # Rough depth from median in small box around center vs outside
    phase, f = ((time - t0 + 0.5*period) % period) / period - 0.5, flux
    intrans = f[np.abs(phase) < 0.02]
    oot = f[np.abs(phase) > 0.05]
    depth = np.nanmedian(oot) - np.nanmedian(intrans)
    # approximate duration fraction
    duration_frac = 2*0.02
    return depth, duration_frac

def run_on_lightcurve(time, flux, model):
    # 1) quick period search
    period = quick_period_search(time, flux)
    # pick t0 as time of min flux found after folding
    phase, f = ((time - time[0] + 0.5*period) % period) / period - 0.5, flux
    t0_index = np.argmin(f)
    t0 = time[t0_index]
    global_v, local_v = make_global_local_views(time, flux, period, t0)
    g = global_v[None,:,None]; l = local_v[None,:,None]
    p = model.predict({'global_input': g, 'local_input': l})[0,0]
    depth, dur = estimate_depth_duration(time, flux, period, t0)
    snr = vet_candidate(time, flux, period, t0, depth)
    result = {'score': float(p), 'period': float(period), 't0': float(t0), 'depth': float(depth), 'snr': float(snr)}
    return result

if __name__ == "__main__":
    print("Run inference by calling run_on_lightcurve with time, flux arrays.")
