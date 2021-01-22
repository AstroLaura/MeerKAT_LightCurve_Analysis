# L. N. Driessen
# Last updated: 2021.01.22

import numpy as np
import glob
import pandas as pd

import scipy as sp
import scipy.stats as spstats

from astropy import units as un
from astropy.coordinates import SkyCoord


def filter_extended(lcfs, extended_file,
                    ra_col='ra',
                    dec_col='decl'):
    '''
    Filters sources from a list of source files
    
    This takes the file names of light curve files
    and outputs a list of sources that *don't*
    match the coordinates of the extended_file
    sources.
    
    Args:
    lcfs (list): list of file names (including the
                 path) of light curve files
    extended_file (str): the name (including the
                         path) of the numpy file
                         containing the list of source
                         coordinates that you want to
                         filter out
    kwargs:
    ra_col (str): The name of the right ascension column
                  Default = 'ra'
    dec_col (str): The name of the declination column
                   Default = 'decl'

    Returns:
    A list of filenames of sources that *don't*
    match the coordinates in extended_file
    '''
    extended_sources = np.load(extended_file)
    es_sc = SkyCoord(extended_sources, unit=(un.deg, un.deg))
    
    lcs_cut = []
    for s, source in enumerate(lcfs):
        lc = pd.read_csv(source)

        lc_coord = SkyCoord(np.array([[np.nanmean(lc[ra_col]),
                                       np.nanmean(lc[dec_col])]]),
                            unit=(un.degree, un.degree))

        seps = lc_coord.separation(es_sc)
        min_sep = np.nanmin(seps.deg)

        if min_sep > 3./60./60.:
            lcs_cut.append(source)

    return lcs_cut


def filter_sn(lcfs, min_sn,
              flux_col='f_int',
              flux_err_col='f_int_err'):
    '''
    Remove light curves with S/N less than min_sn
    
    This function removes any sources that have
    a signal to noise less than min_sn.
    The fluxes of every source are divided by the
    uncertainties to give the approximate
    signal to noise. If there is no detection greater
    than zero, or if there is no epoch where the source
    has a signal to noise greater than (default) 2, then
    the source is excluded from the analysis.
    
    Args:
    lcfs (list): list of file names (including the
                 path) of light curve files
    min_sn (float): the minimum allowed signal
                    to noise

    Kwargs:
    flux_col (str): The name of the flux column
                    Default ='f_int'
    flux_err_col (str): The name of the flux error column
                        Default ='f_int_err'
    Returns:
    A list of filenames of sources that have
    a signal to noise over min_sn
    '''    
    lcs_cut = []
    for s, source in enumerate(lcfs):
        lc = pd.read_csv(source)

        # Get an approximation of the
        # signal to noise for each epoch
        signal_to_noise = lc[flux_col] / lc[flux_err_col]
        
        min_val = lc[flux_col] - lc[flux_err_col]

        # If the source has at least one detection
        # greater than the sinal to noise limit,
        # include it in the analysis
        if ((np.nanmax(signal_to_noise) > min_sn) and
            (np.nanmax(min_val) > 0.)):
            lcs_cut.append(source)
    return lcs_cut


def filter_dpc(lcfs,
               distance_from_phase_centre,
               pc_col='dist_to_pc_DEG'):
    '''
    Filters sources that are
    too distance from the phase centre
    
    This takes the file names of light curve files
    and outputs a list of sources that are within
    distance_from_phase_centre from the phase
    centre
    
    Args:
    lcfs (list): list of file names (including the
                 path) of light curve files
    distance_from_phase_centre (float): the maximum allowed
                                        distance from the
                                        phase centre that
                                        a source can be
                                        to be included
                                        (in degrees)
    kwargs:
    pc_col (str): The name of the distance from phase
                  centre column
                  Default = 'dist_to_pc_DEG'

    Returns:
    A list of filenames of sources that are less than the
    maximum distance from the phase centre
    '''
    lc_cut = []
    for l, lcf in enumerate(lcfs):
        # Read the light curve file
        lc = pd.read_csv(lcf)
        # Make sure the information
        # is in chronological order
        lc = lc.sort_values(by=[mjd_col])

        pc_sep = np.nanmean(lc[pc_col])
        if pc_sep <= distance_from_phase_centre:
            lc_cut.append(lcf)

    return lc_cut


def filter_coords(lcfs, coords,
                  ra_col='ra',
                  dec_col='decl',
                  sep=3./60./60.):
    '''
    Filters sources from a list of coordinates
    
    This takes the file names of light curve files
    and outputs a list of sources that *don't*
    match the coordinates of coords.
    
    Args:
    lcfs (list): list of file names (including the
                 path) of light curve files
    coords (SkyCoord array): array of astropy SkyCoords
                             of sources you want to
                             filter out
    kwargs:
    ra_col (str): The name of the right ascension column
                  Default = 'ra'
    dec_col (str): The name of the declination column
                   Default = 'decl'
    sep (float): if two sources are less than sep (in degrees)
                 apart they are considered to match

    Returns:
    A list of filenames of sources that *don't*
    match the coordinates in coords
    '''
    lc_cut = []
    for l, lcf in enumerate(lcfs):
        # Read the light curve file
        lc = pd.read_csv(lcf)
        # Turn the source coordinates into an
        # astropy SkyCoord object
        lc_coord = SkyCoord(np.array([[np.nanmean(lc[ra_col]),
                                       np.nanmean(lc[dec_col])]]),
                            unit=(un.degree, un.degree))
        # Check if the source is one of the
        # known variables and move on if
        # it is one
        seps = lc_coord.separation(coords)
        if np.nanmin(seps.deg)<sep:
            print('Source {} is a known variable'.format(l))
        else:
            lc_cut.append(lcf)

    return lc_cut


def get_pearsons_r(flux0, flux1, remove_runcat=True):
    '''
    Get Pearson's r correlation coefficient for two arrays.
    
    This uses the scipy.stats.pearsonr function to
    get the Pearson's r correlation coefficient between
    two arrays, in this case lightcurves. It makes
    sure that nan values don't mess up the correlation,
    but removing nan values from both arrays. I.e. if
    arrey0 has a nan at index 2, the value at index 2 is
    removed from both arrays. Here we assume both arrays
    are one-dimensional numpy arrays of the same length.
    
    Args:
    flux0 (array): the light curve of the first source
    flux1 (array): the light curve of the second source
    
    Returns:
    Tuple: (pearson's r correlation coefficient,
            two-tailed p-value)
    If there is an issue with the arrays
    then (np.nan, np.nan) will be
    returned instead.
    '''
    # Remove the runcat from the start of each
    # flux
    if remove_runcat:
        flux0 = flux0[1:]
        flux1 = flux1[1:]
    
    # Check for infs and nans
    flux0_nans = np.where(np.isnan(flux0))[0]
    flux0_infs = np.where(np.isinf(flux0))[0]
    flux1_nans = np.where(np.isnan(flux1))[0]
    flux1_infs = np.where(np.isinf(flux1))[0]
    
    # Remove the infs and nans from both arrays
    nans = np.unique(np.concatenate((flux0_nans, flux1_nans)))
    infs = np.unique(np.concatenate((flux0_infs, flux1_infs)))
    remove = np.concatenate((nans, infs))
    try:
        flux0_nn = np.delete(flux0, remove)
        bob = True
        try:
            flux1_nn = np.delete(flux1, remove)
            bob = True
        except IndexError:
            print('Flux wrong length: {}'.format(len(flux1)))
            bob = False
    except IndexError:
        print('Flux wrong length: {}'.format(len(flux0)))
        bob = False

    # Get the Pearson's r and p-value
    if bob:
        try:
            r = spstats.pearsonr(flux0_nn, flux1_nn)
        except ValueError:
            # If there's something wrong, you'll
            # get back nans
            print('Value Error')
            print(s, len(flux0), len(flux1), len(notnans))
            r = (np.nan, np.nan)
    else:
        r = (np.nan, np.nan)
        
    return r


def make_lc_array(all_lcs,
                  mjd_col='mjd',
                  flux_col='f_int',
                  flux_err_col='f_int_err',
                  freq_col='freq_eff_int',
                  pc_col='dist_to_pc_DEG'):
    '''
    Gather the info about each source that your need.
    
    To work out the correlations and test things you
    need to have info about each source. It's much faster
    and easier to gather all that stuff at the start,
    rather than re-loading each source every time
    you need it. It also means that your can do array-wise
    calculations, which are again faster. So this function
    grabs each source and puts its flux densities into
    an array and also sticks the distance to phase centre,
    median flux density, ra and dec of the source and puts
    it into a pandas dataframe. The first column of the flux
    density array is the runcat of each source. This is also
    in the pandas array so that you can keep track of each
    Source, but you have to remember to remove that column
    when you do calculations!
    
    Args:
    all_lcs (list): list of file names (including the path)
                    of the light curve files for analysis

    *kwargs:
    min_sn (float): the minimum required signal to noise
                    for a source to be considered detected
                    Default: 2
    mjd_col (str): the label for the MJD column in the
                   lightcurve files
    flux_col (str): the label for the flux density column
                    in the lightcurve files
    flux_err_col (str): the label for the flux density error
                        column in the lightcurve files
    freq_col (str): the label for the frequency column
                    in the lightcurve files
    pc_col (str): the label for the distance to phase centre
                  column in the lightcurve files

    Returns:
    (flux, info)
    flux: an array where each row is the flux density
          over time of a source. The first column
          contains the runcat of the source
    info: a pandas dataframe with the runcat, distance
          to phase centre, median flux density, ra,
          and dec of each source
    '''
    for s, lcf in enumerate(all_lcs):
        source_lc = pd.read_csv(lcf)

        ra = np.nanmean(source_lc['ra'])
        dec = np.nanmean(source_lc['decl'])

        flux0 = np.array(source_lc[flux_col])
        flux0_errs = np.array(source_lc[flux_err_col])

        rcat = np.array(source_lc['runcat'])[0]
        pc_dist = np.nanmean(source_lc[pc_col])

        try:
            flux0 = np.concatenate((np.array([rcat]),
                                    flux0))
            flux = np.concatenate((flux,
                                   np.expand_dims(flux0, axis=0)))
            source_info = np.concatenate((source_info,
                                          np.array([[rcat,
                                                     pc_dist,
                                                     np.nanmedian(flux0),
                                                     ra, dec]])))
        except (NameError, UnboundLocalError):
            flux = np.array(source_lc[flux_col])
            flux = np.concatenate((np.array([rcat]),
                                   flux))
            flux = np.expand_dims(flux, axis=0)
            source_info = np.array([[rcat,
                                     pc_dist,
                                     np.nanmedian(flux0),
                                     ra, dec]])

    return flux, pd.DataFrame(data=source_info, columns=['runcat',
                                                         pc_col,
                                                         'median {}'.format(flux_col),
                                                         'ra', 'dec'])


def get_corrs(source, other_sources):
    '''
    Calculate the Pearson's correlation coefficient
    
    This is using the formula from 
    https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.stats.pearsonr.html
    
    I chose to write my own function as the scipy
    function doesn't have array-wise functionality,
    which makes it really slow for calculating
    a large amount of coefficients.
    
    Therefore, this function takes one light curve
    and calculates the correlation coefficient
    of that source with an array of light curves.
    
    Args:
    source (array): the light curve of a single source
    other_sources (array): an array of the light curves
                           you want to compare to. Each row
                           is a different light curve
    
    Returns:
    An array containing the correlation coefficients
    '''
    x_mean = np.nanmean(source)
    
    y_mean = np.nanmean(other_sources, axis=1)
    y_mean = np.transpose(np.tile(y_mean, (len(source), 1)))
    
    y_sub = other_sources - y_mean
    y_sub_squared = y_sub ** 2.
    
    x_sub = np.tile((source - x_mean),
                    (len(y_sub), 1))
    x_sub_squared = np.tile(((source - x_mean) ** 2.),
                            (len(y_sub), 1))
    
    top = np.nansum(x_sub*y_sub, axis=1)
    bottom = np.sqrt(np.nansum(x_sub_squared, axis=1) *
                     np.nansum(y_sub_squared, axis=1))
    
    return top/bottom


def compare_all(fluxes, info,
                mjd_col='mjd',
                flux_col='f_int',
                flux_err_col='f_int_err',
                freq_col='freq_eff_int',
                pc_col='dist_to_pc_DEG'):
    '''
    Compare every light curve to every other light curve
    using the Pearson's r correlation coefficient
    
    Args:
    fluxes (array): array of fluxes from make_lc_array
    info (dataframe): pandas dataframe of information
                      from make_lc_array
    max_dpc (float): the maximum allowed distance from
                     phase centre in degrees

    *kwargs:
    mjd_col (str): the label for the MJD column in the
                   lightcurve files
    flux_col (str): the label for the flux density column
                    in the lightcurve files
    flux_err_col (str): the label for the flux density error
                        column in the lightcurve files
    freq_col (str): the label for the frequency column
                    in the lightcurve files
    pc_col (str): the label for the distance to phase centre
                  column in the lightcurve files

    returns:
    a dataframe with the results of the matches
    '''
    # Sort both the flux array and the info
    # dataframe by runcat
    flux_order = np.argsort(fluxes[:, 0])
    fluxes_included = fluxes[flux_order]
    inner_info = info.sort_values(by='runcat')
    
    results = np.ones((1, 11))
    for s1, source1 in enumerate(np.array(inner_info['runcat'])):
        # Set up the source you're going to compare
        # all the other sources to
        s1_loc = np.where(fluxes_included[:, 0] == source1)[0]
        s1_flux = fluxes_included[s1_loc][0]
        s1_flux = s1_flux[1:]
        s1_info = inner_info[inner_info['runcat'] == source1]

        # Delete the source you're comparing to
        # (otherwise you'll get correlations of 1)
        fluxes_included = np.delete(fluxes_included, s1_loc, axis=0)
        runcats_included = np.copy(fluxes_included)[:, 0]

        fluxes_for_corr = np.copy(fluxes_included)
        fluxes_for_corr = fluxes_for_corr[:, 1:]
        info_included = inner_info[inner_info['runcat'].isin(runcats_included)]
        
        # Correlate the source to all of the other sources
        corrs = get_corrs(s1_flux, fluxes_for_corr)
        
        # Add the results to the results array
        corr_results = np.ones((len(corrs), 11))
        
        corr_results[:, 0] = np.repeat(np.array(s1_info['runcat'])[0],
                                       len(corr_results), axis=0)
        corr_results[:, 1] = np.repeat(np.array(s1_info[pc_col])[0],
                                       len(corr_results), axis=0)
        corr_results[:, 2] = np.repeat(np.array(s1_info['ra'])[0],
                                       len(corr_results), axis=0)
        corr_results[:, 3] = np.repeat(np.array(s1_info['dec'])[0],
                                       len(corr_results), axis=0)
        corr_results[:, 4] = np.repeat(np.array(s1_info['median {}'.format(flux_col)])[0],
                                       len(corr_results), axis=0)
        
        corr_results[:, 5] = np.array(info_included['runcat'])
        corr_results[:, 6] = np.array(info_included[pc_col])
        corr_results[:, 7] = np.array(info_included['ra'])
        corr_results[:, 8] = np.array(info_included['dec'])
        corr_results[:, 9] = np.array(info_included['median {}'.format(flux_col)])
        
        corr_results[:, 10] = corrs
        
        results = np.concatenate((results, corr_results))
    results = results[1:]
        
    results_frame = pd.DataFrame(data=results,
                                 columns=['s1_runcat',
                                          's1_{}'.format(pc_col),
                                          's1_ra', 's1_dec',
                                          's1_median_{}'.format(flux_col),
                                          's2_runcat',
                                          's2_{}'.format(pc_col),
                                          's2_ra', 's2_dec',
                                          's2_median_{}'.format(flux_col),
                                          'correlation coefficient'])
    results_frame = results_frame[results_frame['s1_runcat'] !=
                                  results_frame['s2_runcat']]
    
    return results_frame


if __name__ in '__main__':
    # The TraP database name
    db = 'Laura_NWLayers_ReProc'

    # The path to the light curves you want
    # to use
    lightcurve_path = ('/raid/driessen/FlareStars/'
                       'GX339/Source_Light_Curves/'
                       'Average_Scaled/')
    # The path to where files will be saved
    file_path = ('/raid/driessen/FlareStars/'
                 'GX339/Correlation_Investigation/')

    # The dataset ID values (from TraP) and
    # the corresponding central frequencies (MHz)
    DSs = [49, 52, 53, 50, 54, 55, 51, 33]
    freqs = [1658, 1551, 1444, 1337, 1123, 1016, 909, 'MFS']

    # The minimum signal to noise a source
    # needs to have (in at least one epoch)
    # to be included in the analysis
    minimum_sn = 2.

    # The coordinates of the extended sources in
    # the field
    es_coords = np.load(('/raid/driessen/FlareStars/'
                         'GX339/2021.01.18_GX339_KnownExtendedSources.npy'))

    # Make or load the flux array and info dataframe
    for d, ds in enumerate(DSs):
        print('----------------------------------')
        freq = freqs[d]
        print('Working on {0} MHz, ds {1}'.format(freq, ds))

        for e, ext in enumerate(['ES', 'PS']):
            print('******************')
            print('Running {}'.format(ext))

            all_lcs = sorted(glob.glob(('{0}rcat*_'
                                        'ra*_dec*_'
                                        'db{1}_ds{2}_'
                                        '{3}.csv').format(lightcurve_path,
                                                          db,
                                                          ds,
                                                          ext)))
            sn_cut_lcs = filter_sn(all_lcs, minimum_sn)

            for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
                fc_err = '{}_err'.format(fc)
                print('Working on {0} {1}'.format(fc, fc_err))

                flux_filename = ('{0}AllFluxes_'
                                 '{1}MHz_'
                                 '{2}_'
                                 'db{3}_'
                                 'ds{4}_'
                                 '{5}_'
                                 'minSN{6}').format(file_path,
                                                    freq, fc,
                                                    db, ds, ext,
                                                    minimum_sn)
                info_filename = ('{0}AllInfo_'
                                 '{1}MHz_'
                                 '{2}_'
                                 'db{3}_'
                                 'ds{4}_'
                                 '{5}'
                                 'minSN{6}.csv').format(file_path,
                                                        freq, fc,
                                                        db, ds, ext,
                                                        minimum_sn)

                flux, info = make_lc_array(sn_cut_lcs,
                                           flux_col=fc,
                                           flux_err_col=fc_err)
                print('Shape of flux array: ', np.shape(flux))
                np.save(flux_filename, flux)
                info.to_csv(info_filename)

                print('*** {} Done ***'.format(fc))
            print('^^^ {} Done ^^^'.format(ext))
        print('--- {} Done ---'.format(freq))
        print('\n----------------------------------\n')
    print('\n******************************')
    print('Done')
    print('\n******************************')
