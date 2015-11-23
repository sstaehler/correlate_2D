
# coding: utf-8

import netCDF4
# from progressbar import ProgressBar, ETA, Bar, RotatingMarker, Percentage
import numpy as np
import scipy.stats
from multiprocessing import Pool
# import os

filename = 'BSWave2006.nc'
file_in = netCDF4.Dataset(filename)

variable = 'tmm10'


def calc_corr(iel):
    slope = 0
    intercept = -1
    lat = latlat.ravel()[iel]
    lon = lonlon.ravel()[iel]
    lat_idx = np.abs(lats[:] - lat).argmin()
    lon_idx = np.abs(lons[:] - lon).argmin()

    if not np.ma.is_masked(data_2d[:, lat_idx, lon_idx]):

        ts_here = data_2d[:, lat_idx, lon_idx]

        conv = np.sum(ts_here[:, None, None] * data_2d[:, :, :], axis=0)
        conv *= normvec * normvec[lat_idx, lon_idx]

        convs = conv.flatten()
        distances = np.sqrt(np.cos(lat)**2 * (lat-latlat)**2 +
                            (lon-lonlon)**2).flatten() * 111.

        slope, intercept, a, b, c = scipy.stats.linregress(distances,
                                                           np.log(convs))
    return slope  # , intercept

if __name__ == '__main__':
    data_2d = file_in.variables[variable]
    lats = file_in.variables['latitude']
    lons = file_in.variables['longitude']

    latlat, lonlon = np.meshgrid(lats[:], lons[:])
    nelements = np.prod(latlat.shape)

    data_2d.set_auto_mask(True)

    normvec = 1. / np.sqrt(np.sum((data_2d[:, :, :])**2, axis=0))

    slopes = np.zeros(nelements)
    intercepts = np.zeros(nelements)
    # widgets = ['Correlating: ', Percentage(), ' ',
    #             Bar(marker=RotatingMarker()),
    #         ' ', ETA()]
    # pbar = ProgressBar(widgets = widgets, maxval=nelements).start()

    p = Pool(processes=2)
    slopes = p.map(calc_corr, range(0, nelements))
