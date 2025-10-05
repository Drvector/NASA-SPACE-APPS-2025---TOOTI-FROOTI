"""
data_fetch.py
- utilities to download light curves (Kepler/TESS) and fetch KOI/labels
- uses lightkurve and astroquery/MAST
"""

import os
from lightkurve import search_lightcurvefile
from astropy.table import Table
import pandas as pd
import numpy as np
from astroquery.mast import Observations
from tqdm import tqdm

# Configure where to store data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_kepler_lightcurve(kepid, mission='Kepler', cadence='long', outdir=DATA_DIR):
    """
    Download all available lightcurve files for a Kepler target using lightkurve.
    Returns list of file paths (FITS-like objects are loaded in memory).
    """
    try:
        sr = search_lightcurvefile(f'KIC {kepid}', mission=mission)
        if len(sr) == 0:
            sr = search_lightcurvefile(f'{kepid}', mission=mission)
        lcf = sr.download_all(download_dir=outdir)
        return lcf
    except Exception as e:
        print("download error", e)
        return None

def download_tess_lightcurve(tic, sector=None, outdir=DATA_DIR):
    """Download TESS lightcurve(s) via lightkurve by TIC id"""
    try:
        query = f'TIC {tic}'
        sr = search_lightcurvefile(query, mission='TESS')
        if sector:
            sr = sr[(sr.table['obsID'].astype(str).str.contains(str(sector)))]
        lcf = sr.download_all(download_dir=outdir)
        return lcf
    except Exception as e:
        print("download error", e)
        return None

def fetch_koi_table(out_csv="data/koi_table.csv"):
    """
    Download KOI table from NASA Exoplanet Archive (simple approach: use the public CSV hosted by NASA).
    For full bulk DV reports use the Exoplanet Archive bulk download page.
    """
    koi_url = "https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html"
    # The KOI table can be downloaded programmatically from the archive APIs; here we keep a placeholder:
    # Users should fetch the cumulative KOI table from the Exoplanet Archive web interface.
    print("Please download the cumulative KOI table from NASA Exoplanet Archive and save as data/koi_table.csv")
    return

if __name__ == "__main__":
    # Example: download one Kepler target (change kepid)
    # download_kepler_lightcurve(2302548)
    pass
