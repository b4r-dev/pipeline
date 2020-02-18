__all__ = ["despike_outliers"]


# dependent packages
import numpy as np


# main functions
def despike_outliers(T_cal, threshold=100):
    """Detect outliers and replace them by NaN."""
    where = np.abs(T_cal) > threshold

    T_cal_new = T_cal.copy()
    T_cal_new[where] = np.nan
    return T_cal_new
