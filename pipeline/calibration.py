__all__ = ["load_data_xffts", "calibrate_intensity"]


# standard library
from copy import deepcopy
from datetime import datetime


# dependent packages
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# constants
NUM_CHANS = 2 ** 15
BANDWIDTH = 2.5  # GHz
CHANWIDTH = BANDWIDTH / NUM_CHANS
LEN_TIMEFMT = 24


# main functions
def load_data_xffts(path_xffts, path_ant=None, sideband="USB", coordsys="RADEC"):
    """Load XFFTS data as an xarray's DataArray."""
    da = load_netcdf(path_xffts)

    array = da["array"]
    scanid = da["scancount"]
    scantype = da["bufpos"]
    time = da["date"].values
    integtime = da["integtime"] * 1e-6  # s

    # modify time
    time = np.array([t[:LEN_TIMEFMT] for t in time])
    time = time.astype("datetime64[ns]")
    time = correct_outlier_time(time)
    time = xr.DataArray(time, dims=("t",))

    # modify array
    array /= integtime

    # create DataArray
    P = xr.DataArray(array, dims=("t", "ch"))
    P.coords["t"] = time
    P.coords["scanid"] = scanid
    P.coords["scantype"] = scantype
    P.coords["integtime"] = integtime

    # add antenna info (if any)
    if path_ant is not None:
        freq = get_freq(path_ant, sideband)
        x, y = get_coordinates(path_ant, coordsys)
        x = x.interp_like(P)
        y = y.interp_like(P)

        P.coords["ch"] = freq
        P.coords["x"] = x
        P.coords["y"] = y

    return P


def calibrate_intensity(P_sci, P_cal, T_amb=273, progress=True):
    """Calibrate intensity by chopper wheel calibration."""
    P_on = P_sci[P_sci.scantype == "ON"]
    P_off = P_sci[P_sci.scantype == "REF"]
    P_r = P_cal[P_cal.scanid == 0].mean("t")

    T_cal = xr.zeros_like(P_on)
    T_sys = xr.zeros_like(P_on)

    for on_id in tqdm(np.unique(P_on.scanid), disable=not progress):
        # ON array of single scan
        P_on_ = P_on[P_on.scanid == on_id]

        # OFF arrays between ON array
        P_off_lead = P_off[P_off.scanid == on_id - 1]
        P_off_trail = P_off[P_off.scanid == on_id + 1]
        P_off_ = xr.concat([P_off_lead, P_off_trail], "t").mean("t")

        # calibrate intensity and calculate Tsys
        T_cal_ = T_amb * (P_on_ - P_off_) / (P_r - P_off_)
        T_cal[T_cal.scanid == on_id] = T_cal_

        T_sys_ = T_amb / (P_r / P_off_ - 1)
        T_sys[T_sys.scanid == on_id] = T_sys_

    T_cal.coords["T_sys"] = T_sys
    return T_cal


# helper functions
def load_netcdf(path, copy=True, unwrap=True):
    """Load netCDF as an xarray's Dataset or DataArray."""
    with xr.open_dataset(path) as ds:
        if copy:
            ds = deepcopy(ds)

        if len(ds) == 1 and unwrap:
            key = list(ds)[0]
            return ds[key]
        else:
            return ds


def get_freq(path_ant, sideband):
    """Get observed frequency in GHz as an xarray's DataArray."""
    ds = load_netcdf(path_ant)

    LO_1 = ds["Header.B4r.LineFreq"].values[0]  # GHz
    LO_2 = ds["Header.B4r.If2Freq"].values[0]  # GHz

    if sideband == "USB":
        freq = LO_1 + LO_2 + np.arange(0, -BANDWIDTH, -CHANWIDTH)
    elif sideband == "LSB":
        freq = LO_1 - LO_2 + np.arange(0, +BANDWIDTH, +CHANWIDTH)
    else:
        raise ValueError(f"Invalid sideband: {sideband}")

    return xr.DataArray(freq, dims=("ch",))


def get_coordinates(path_ant, coordsys):
    """Get coordinates with time as an xarray's DataArray."""
    ds = load_netcdf(path_ant)

    time = ds["Data.TelescopeBackend.TelTime"].values  # unix time
    time = np.array([datetime.utcfromtimestamp(t) for t in time])
    time = time.astype("datetime64[ns]")
    time = correct_outlier_time(time)

    if coordsys == "RADEC":
        x = np.rad2deg(ds["Data.TelescopeBackend.SourceRaDes"].values)
        y = np.rad2deg(ds["Data.TelescopeBackend.SourceDecDes"].values)
    elif coordsys == "AZEL":
        x = np.rad2deg(ds["Data.TelescopeBackend.TelAzMap"].values)
        y = np.rad2deg(ds["Data.TelescopeBackend.TelElMap"].values)
    else:
        raise ValueError(f"Invalid coordsys: {coordsys}")

    x = xr.DataArray(x, coords={"t": time}, dims=("t",))
    y = xr.DataArray(y, coords={"t": time}, dims=("t",))
    return x, y


def correct_outlier_time(time, sigma=3):
    """Detect time outliers and correct them by interpololation."""
    # lazy import of astropy
    from astropy.modeling.models import Polynomial1D
    from astropy.modeling.fitting import LinearLSQFitter
    from astropy.modeling.fitting import FittingWithOutlierRemoval
    from astropy.stats import sigma_clip

    # create fitting model
    model = Polynomial1D(1)
    fitter = LinearLSQFitter()
    fitter = FittingWithOutlierRemoval(fitter, sigma_clip, sigma=sigma)

    # apply model and get mask
    time = pd.Series(time.astype("int64"))
    model, mask = fitter(model, np.arange(len(time)), time)

    # apply mask and interpolation
    time[mask] = np.nan
    return pd.to_datetime(time.interpolate(), unit="ns")
