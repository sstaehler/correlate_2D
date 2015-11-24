
# coding: utf-8

import netCDF4
from progressbar import ProgressBar, ETA, Bar, Percentage
import numpy as np
import scipy.stats
import time
import argparse
from multiprocessing import Pool
# import os


def calc_corr(iel):
    slope = 0
    intercept = -1
    lat = latlat.ravel()[iel]
    lon = lonlon.ravel()[iel]
    lat_idx = np.abs(lats[:] - lat).argmin()
    lon_idx = np.abs(lons[:] - lon).argmin()

    if not np.ma.is_masked(data_2d[:, lat_idx, lon_idx]):

        distances = np.sqrt(np.cos(lat)**2 * (lat-latlat)**2 +
                            (lon-lonlon)**2).flatten() * 111.

        ts_here = data_2d[:, lat_idx, lon_idx]

        if dist_max>0:
            data_masked = np.ma.masked_where(distances>dist_max, data_2d[:,:,:])
        else:
            data_masked = data_2d[:,:,:]

        conv = np.sum(ts_here[:, None, None] * data_masked, axis=0)
        conv *= normvec * normvec[lat_idx, lon_idx]

        convs = conv.flatten()
        slope, intercept, a, b, c = scipy.stats.linregress(distances,
                                                           np.log(convs))
    return slope  # , intercept

if __name__ == '__main__':

    helptext = 'Calculate spatial correlation length in a 2D time series'
    formatter_class = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=helptext,
                                     formatter_class=formatter_class)

    helptext = 'NetCDF file name. Should follow BLABLA convention'
    parser.add_argument('file_name', help=helptext)

    helptext = 'Variable to calculate correlation on'
    parser.add_argument('-v', '--variable', help=helptext)

    helptext = 'Maximum correlation length. \n' + \
               'Only elements with a distance smaller max_length are \n' + \
               'correlated. Smaller max_length speed up the calculation'
    parser.add_argument('-m', '--max_length', help=helptext, default=-1)

    args = parser.parse_args()

    filename = args.file_name
    file_in = netCDF4.Dataset(filename)

    variable = args.variable
    data_2d = file_in.variables[variable]

    dist_max = args.max_length

    lats = file_in.variables['latitude']
    lons = file_in.variables['longitude']

    latlat, lonlon = np.meshgrid(lats[:], lons[:])
    nelements = np.prod(latlat.shape)

    data_2d.set_auto_mask(True)

    normvec = 1. / np.sqrt(np.sum((data_2d[:, :, :])**2, axis=0))

    slopes = np.zeros(nelements)
    intercepts = np.zeros(nelements)
    widgets = ['Correlating: ', Percentage(), ' ',
               Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=nelements).start()

    p = Pool(processes=2)
    result = p.map_async(calc_corr, range(0, nelements), chunksize=1)

    while not result.ready():
        ndone = nelements - result._number_left
        pbar.update(ndone)
        time.sleep(1)
        if ndone == nelements:
            break
    p.close()
    p.join()

    slopes = result.get()
