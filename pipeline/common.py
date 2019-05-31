# from standard library
from pathlib import Path

# from dependent packages
import numpy as np
import xarray as xr

# module constants
NUM_CHANS = 2**15
BANDWIDTH = 2.5 # GHz
CHANWIDTH = BANDWIDTH / NUM_CHANS
LEN_TIMEFMT = 24


# main functions
def load_data_xffts(path):
    """Load XFFTS data as an xarray's DataArray."""
    da = load_netcdf(path)

    time = da['date'].values
    array = da['array'].values
    scanid = da['scancount'].values
    scantype = da['bufpos'].values
    integtime = da['integtime'].values

    # modify time
    time = np.array([s[:LEN_TIMEFMT] for s in time])
    time = time.astype('datetime64[ns]')
    time = correct_outlier_time(time)

    # modify array
    array /= integtime[:, np.newaxis]

    # create DataArray
    P = xr.DataArray(array, dims=('time', 'freq'))
    P.coords['time'] = 'time', time
    P.coords['scanid'] = 'time', scanid
    P.coords['scantype'] = 'time', scantype
    P.coords['integtime'] = 'time', integtime
    return P


# utility functions
def load_netcdf(path, copy=True, unwrap=True):
    """Load netCDF as an xarray's Dataset or DataArray."""
    with xr.open_dataset(path) as ds:
        if copy:
            ds = ds.copy()

        if len(ds)==1 and unwrap:
            key = list(ds)[0]
            return ds[key]
        else:
            return ds


def get_freq(path_ant, sideband):
    """Get an array of observed frequency in GHz."""
    with xr.open_dataset(path_ant) as ds:
        LO_1 = ds['Header.B4r.LineFreq'].values[0] # GHz
        LO_2 = ds['Header.B4r.If2Freq'].values[0] # GHz

    if sideband == 'USB':
        return LO_1+LO_2 + np.arange(0, -BANDWIDTH, -CHANWIDTH)
    elif sideband == 'LSB':
        return LO_1-LO_2 + np.arange(0, +BANDWIDTH, +CHANWIDTH)
    else:
        raise ValueError(f'Invalid sideband: {sideband}')


def correct_outlier_time(time, sigma=3):
    """Detect time outliers and correct them by interpololation."""
    # lazy import of astropy
    from astropy.modeling.models import Polynomial1D
    from astropy.modeling.fitting import LinearLSQFitter
    from astropy.modeling.fitting import FittingWithOutlierRemoval
    from astropy.stats import sigma_clip

    model = Polynomial1D(1)
    fitter = LinearLSQFitter()
    fitter = FittingWithOutlierRemoval(fitter, sigma_clip, sigma=sigma)

    time = pd.Series(time.astype('int64'))
    model, mask = fitter(model, np.arange(len(time)), time)

    time[mask] = np.nan
    return pd.to_datetime(time.interpolate(), unit='ns')
