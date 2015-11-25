
# coding: utf-8

import netCDF4
from progressbar import ProgressBar, ETA, Bar, Percentage
import numpy as np
import time
import argparse
import sys
# import matplotlib.pyplot as plt
from multiprocessing import Pool
# import os


def define_argument_parser():
    helptext = 'Calculate spatial correlation length in a 2D time series'
    formatter_class = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=helptext,
                                     formatter_class=formatter_class)

    helptext = 'NetCDF input file name. Should follow BLABLA convention'
    parser.add_argument('input_file_name', help=helptext)

    helptext = 'Output file name. Default is input_file_name_corr.nc'
    parser.add_argument('-o', '--output_file_name', help=helptext)

    helptext = 'Variable to calculate correlation on'
    parser.add_argument('-v', '--variable', help=helptext)

    helptext = 'Maximum correlation length. \n' + \
               'Only elements with a distance smaller max_length are \n' + \
               'correlated. Smaller max_length speed up the calculation'
    parser.add_argument('-m', '--max_length', help=helptext, default=-1)

    helptext = 'Depth layer to correlate. '
    parser.add_argument('-d', '--depth_layer', help=helptext,
                        type=int, default=0)

    helptext = 'Keep original variable in output file?'
    parser.add_argument('-k', '--keep_original', help=helptext,
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


def distance(lat, lon, latgrid, longrid):
    # Since this is a small region, this approximation is a precise enough
    # and a lot faster than calculating the true Great Circle Arc distance
    R = 6371  # radius of the earth in km
    x = (longrid - lon) * np.cos(0.5*(latgrid+lat))
    y = latgrid - lat
    d = R * np.sqrt(x*x + y*y)
    return d


def conv_small_dist(A, idx_x, idx_y, dx_max, dy_max):

    x_min = np.max((idx_x - dx_max, 0))
    x_max = np.min((idx_x + dx_max + 1, A.shape[0] + 1))
    y_min = np.max((idx_y - dy_max, 0))
    y_max = np.min((idx_y + dy_max + 1, A.shape[1] + 1))

    ts_here = A[:, idx_x, idx_y]
    conv = np.sum(A[:, x_min:x_max, y_min:y_max] *
                  ts_here[:, None, None], axis=0)

    conv *= normvec[x_min:x_max, y_min:y_max] * normvec[idx_x, idx_y]

    distances = distance(latlat[idx_x, idx_y], lonlon[idx_x, idx_y],
                         latlat[x_min:x_max, y_min:y_max],
                         lonlon[x_min:x_max, y_min:y_max]).flatten()

    # plt.pcolormesh(normvec[x_min:x_max, y_min:y_max])
    # p = plt.pcolormesh((A[0, x_min:x_max, y_min:y_max]))
    # plt.colorbar(p)
    # plt.plot(distances, conv.flatten(), '+')
    # plt.show()
    return conv.flatten()[distances > 0], distances[distances > 0]


def calc_corr(iel):
    lat = latlat.ravel()[iel]
    lon = lonlon.ravel()[iel]
    lat_idx = np.abs(lats[:] - lat).argmin()
    lon_idx = np.abs(lons[:] - lon).argmin()

    # Check, whether we are outside of the model domain
    if np.mean(abs(data_2d_demean[:, lat_idx, lon_idx])) < 1e-4:
        corr_len = 0.0
    else:
        convs, distances = conv_small_dist(data_2d_demean,
                                           lat_idx, lon_idx,
                                           dxi, dyi)
        corr_len = 1. / \
            np.mean((-np.log(convs)/distances)[distances <= dist_max])
    return corr_len


if __name__ == '__main__':

    args = define_argument_parser()

    # Open NetCDF input file
    filename = args.input_file_name
    file_in = netCDF4.Dataset(filename)

    # Get name of NetCDF output file
    if not args.output_file_name:
        filename_split = filename.split('.')
        filename_out = ''.join(filename_split[0:-1]) \
            + '_corr.' + filename_split[-1]
    else:
        filename_out = args.output_file_name

    # Load variable to correlate
    variable_name = args.variable
    data_2d = file_in.variables[variable_name]

    # Get maximum distance to correlate
    dist_max = float(args.max_length)
    print 'Maximum distance to correlate: %d km' % dist_max

    # Get depth layer on which to correlate
    idepth = args.depth_layer

    # Read latitude and longitude vectors and create mesh grid
    lats = file_in.variables['latc'][:] * np.pi / 180.
    lons = file_in.variables['lonc'][:] * np.pi / 180.
    lonlon, latlat = np.meshgrid(lons, lats)

    # Calculate maximum distance to calculate CC on. Separately for X and Y.
    dx = np.mean(np.diff(lons)) * 180 / np.pi * np.cos(np.mean(lats)) * 111.
    dy = np.mean(np.diff(lats)) * 180 / np.pi * 111.
    print 'Grid spacing: %8.2f/%8.2f km' % (dx, dy)
    dxi = int(dist_max / dx)
    dyi = int(dist_max / dy)
    print 'Elements to evaluate: %d/%d' % (dxi, dyi)

    # Get number of grid points
    nelements = np.prod(latlat.shape)
    print 'Calculating correlation on %d elements' % nelements

    # Remove mean from data (correlation coefficient requires mean==0)
    data_2d_demean = data_2d[:, idepth, :, :] \
        - np.mean(data_2d[:, idepth, :, :], axis=0)

    # Calculate normalization matrix
    normvec = np.sqrt(np.sum((data_2d_demean)**2, axis=0))
    normvec[normvec > 1e-10] = 1./normvec[normvec > 1e-10]

    # Define progress bar
    widgets = ['Correlating: ', Percentage(), ' ',
               Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=nelements).start()

    # Define pool of parallel workers
    p = Pool()
    result = p.map_async(calc_corr, range(0, nelements), chunksize=1)
    p.close()

    # Calculate all elements on the workers until all done
    while not result.ready():
        ndone = nelements - result._number_left
        pbar.update(ndone)
        time.sleep(1)
    p.join()

    print '...done'

    # Save result into 2D variable
    corrlen = np.array(result.get()).reshape(latlat.shape)
    # pc = plt.pcolormesh(slopes)
    # plt.colorbar(pc)
    # plt.show()

    # Save correlated variable into new netCDF file
    print 'Saving results to %s' % filename_out
    with netCDF4.Dataset(filename_out, mode='w',
                         format='NETCDF3_64BIT') as file_out:

        # Copy all attributes
        for name, att in file_in.__dict__.iteritems():
            # Prepend this step to the history attribute
            if name == 'history':
                time_str = time.strftime("%a %b %d %X %Y", time.localtime())
                shell_str = " ".join(sys.argv[:])
                att = time_str + ': python ' + shell_str + '\n' + att
            file_out.setncattr(name, att)

        for name, dimension in file_in.dimensions.iteritems():
            if not dimension.isunlimited():
                file_out.createDimension(name, len(dimension))
            else:
                file_out.createDimension(name, None)

        for name, variable in file_in.variables.iteritems():

            if name == variable_name:
                # For the variable of choice, create a new one with the
                # correlation length

                # Keep the original one?
                if args.keep_original:
                    x = file_out.createVariable(name, variable.datatype,
                                                variable.dimensions)
                    file_out.variables[name][:] = file_in.variables[name][:]

                var_name = variable_name + '_corr'
                x = file_out.createVariable(var_name, variable.datatype,
                                            variable.dimensions[2:])
                file_out.variables[var_name][:, :] = corrlen

            else:
                x = file_out.createVariable(name, variable.datatype,
                                            variable.dimensions)
                file_out.variables[name][:] = file_in.variables[name][:]

            for name_att, att in variable.__dict__.iteritems():
                if name in file_out.variables:
                    file_out.variables[name].setncattr(name_att, att)

        var_corrlen = file_out.createVariable

    file_in.close()
