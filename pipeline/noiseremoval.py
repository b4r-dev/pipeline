__all__ = ["despike_outliers", "bin_channels", "estimate_baseline"]


# dependent packages
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


# main functions
def despike_outliers(T_cal, threshold=100):
    """Detect outliers and replace them by noise."""
    where = (np.abs(T_cal) > threshold) | np.isnan(T_cal)
    std = T_cal.values[~where].std()
    noise = np.random.normal(scale=std, size=int(where.sum()))

    T_cal_new = T_cal.copy()
    T_cal_new.values[where] = noise
    return T_cal_new


def bin_channels(T_cal, size=2):
    """Bin channels by given size."""
    shape = T_cal.shape[0], int(T_cal.shape[1] / size), size

    # compute binned values
    T_cal_bin = T_cal.values.reshape(shape).mean(2)
    ch_bin = T_cal.ch.values.reshape(shape[1:]).mean(1)
    T_sys_bin = T_cal.T_sys.values.reshape(shape).mean(2)

    # make binned data array
    T_cal_bin = xr.DataArray(T_cal_bin, dims=T_cal.dims)
    T_cal_bin["ch"] = "ch", ch_bin
    T_cal_bin["T_sys"] = ("t", "ch"), T_sys_bin

    for key, coord in T_cal.coords.items():
        if coord.dims == ("t",):
            T_cal_bin[key] = coord

    return T_cal_bin


def estimate_baseline(T_cal, order=1, weight=None):
    """Estimate polynomial baseline of each sample."""
    freq = T_cal.ch - T_cal.ch.mean()
    n_freq, n_poly = len(freq), order + 1

    # make design matrix
    X = np.zeros([n_freq, n_poly])

    for i in range(n_poly):
        poly = freq ** i
        X[:, i] = poly / np.linalg.norm(poly)

    y = T_cal.values.T

    # estimate coeffs by solving linear regression problem
    if weight is None:
        weight = 1.0

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y, sample_weight=weight)

    # estimate baseline
    T_base = xr.full_like(T_cal, model.coef_ @ X.T)

    for i in range(n_poly):
        T_base.coords[f"basis_{i}"] = "ch", X[:, i]
        T_base.coords[f"coeff_{i}"] = "t", model.coef_[:, i]

    return T_base
