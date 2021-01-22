# L. N. Driessen
# Last updated: 2021.01.22

params = {"figure.figsize": (12,9),
          "font.size": 20,
          "font.weight": "normal",
          "xtick.major.size": 9,
          "xtick.minor.size": 4,
          "ytick.major.size": 9,
          "ytick.minor.size": 4,
          "xtick.major.width": 4,
          "xtick.minor.width": 3,
          "ytick.major.width": 4,
          "ytick.minor.width": 3,
          "xtick.major.pad": 8,
          "xtick.minor.pad": 8,
          "ytick.major.pad": 8,
          "ytick.minor.pad": 8,
          "lines.linewidth": 3,
          "lines.markersize": 10,
          "axes.linewidth": 4,
          "legend.loc": "best",
          "text.usetex": False,    
          "xtick.labelsize" : 20,
          "ytick.labelsize" : 20,
          }

import matplotlib
matplotlib.rcParams.update(params)

import numpy as np
import glob
from astropy import units as un
from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as spstats
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec


def madulation_parameter(data, dud_epoch=False):
    '''
    Use the median and MAD to make a modulation-type parameter
    
    This parameter takes a 1-D numpy array and calculates
    the MAD (median absolute deviation) parameter.
    We first calculate the median and mad
    for the data. Then we subtract the mean from each value
    and divide by the MAD. The final MAD parameter is the
    absolute value of the extremus.
    
    Args:
    data (array): a 1-D numpy array of values
    
    kwargs:
    dud_epoch (int/array): an integer or array of integers
                           giving the indices of data values
                           to remove/ignore
                           Default: False (no values to
                           remove/ignore)
                           
    returns:
    (madvalue, madlocation)
    madvalue is the value of the MAD parameter for these
    data. madlocation is the index of the epoch where the
    mad parameter is from.
    '''
    # Remove nans as they affect the calculation
    data_bob = np.copy(data[np.where(~np.isnan(data))[0]])
    me = np.nanmedian(data_bob)
    # Use the unscaled MAD (not the equivalent of the
    # standard deviation)
    ma = spstats.median_absolute_deviation(data_bob,
                                           scale=1.0,
                                           nan_policy='omit')
    
    # Remove dodgey epochs if needed
    data_ = np.copy(data_bob)
    if dud_epoch:
        data_[dud_epoch] = np.nan
    
    # Subtract the median from all values
    shifted_data = np.abs(data_ - me)
    # divide the subtracted values
    # by the MAD
    divided_by_mad = shifted_data/ma
    
    # Find the maximum value
    try:
        madvalue = np.nanmax(divided_by_mad)
        madlocation = np.nanargmax(divided_by_mad)
    except ValueError:
        # just in case all the values are
        # nans in the data
        madvalue = np.nan
        madlocation = np.nan
    
    return madvalue, madlocation


def chi2_parameter(data, uncertainties):
    '''
    Calculate the chi2 variability parameter for a set of points.
    
    A description of the variability parameter
    can be found here:
    https://tkp.readthedocs.io/en/r3.0/devref/
    database/schema.html#appendices
    
    Args:
    data (array): a 1D row array of data points
    uncertainties (array): the 1D array of uncertainties
                           that corresponds to the data array
    
    Returns:
    A float that is the value of the variability
    parameter for the data array
    '''
    weights = 1./(uncertainties**2.)
    
    p1 = len(data[~np.isnan(data)])/(len(data[~np.isnan(data)])-1.)
    p2 = np.nanmean(weights*(data**2.))
    p3 = ((np.nanmean(weights*data))**2.)/(np.nanmean(weights))
    
    return p1 * (p2 - p3)


def modulation_parameter(data):
    '''
    Calculate the modulation parameter for a set of points.
    
    A description of the modulation parameter
    can be found here:
    https://tkp.readthedocs.io/en/r3.0/devref/
    database/schema.html#appendices
    
    Args:
    data (array): a 1D row array of data points
    
    Returns:
    A float that is the value of the modulation
    parameter for the data array
    '''
    p1 = 1./np.nanmean(data)
    p2 = len(data)/(len(data)-1)
    p3 = np.nanmean(data**2.) - (np.nanmean(data))**2.
    
    return p1 * np.sqrt(p2*p3)


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


def get_unique_epochs(lcs,
                      mjd_col='mjd'):
    '''
    Get all of the unique MJDs from all files
    
    This gets the unique MJDs from a list
    of csv files (in pandas format)
    
    Args:
    lcs (list): list of file names (including
                the path) of the pandas
                light curve files
    kwargs:
    mjd_col (str): the name of the column
                   containing the MJDs in the
                   files
                   Default: 'mjd'
    Returns:
    An array containing the unique MJD
    values from all the light curve files
    '''
    all_mjds = []
    for lc in lcs:
        mjd = pd.read_csv(lcs[0])[mjd_col]
        all_mjds += list(mjd)
    mjds = np.unique(np.array(all_mjds))

    return mjds


def scale_fluxes(lcfs,
                 ref_epoch='None',
                 flux_col='f_int',
                 ra_col='ra',
                 dec_col='decl',
                 mjd_col='mjd'):
    '''
    Scale all the fluxes by the flux from one epoch
    
    To find the systematics we need to divide all
    of the fluxes for each source by the flux of the
    reference epoch. This function does that.
    You can either set the reference epoch yourself,
    or it will choose the epoch with the most
    values in it (i.e. where the highest number of
    sources are detected)
    It also ignores sources that don't have any
    values with a signal to noise of greater
    than sig_noise.
    
    Args:
    lcfs (list): a list of strings, where each string is the
                 path to a pandas csv. Each pandas csv has the
                 light curve and information for a single source
    
    kwargs:
    ref_epoch (int): The index of the reference
                     epoch. If 'None' an epoch will
                     be chosen. The selected epoch
                     will have the most detections.
                     Default = 'None'
    flux_col (str): The name of the flux column
                    Default ='f_int'
    ra_col (str): The name of the riht ascension column
                  Default = 'ra'
    dec_col (str): The name of the declination column
                   Default = 'decl'
    mjd_col (str): The name of the mjd column
                   Default = 'mjd'
                      
    Returns:
    An array of scaled fluxes where each row is
    a different sources, and the columns are the epochs
    (in chronological order)
    '''
    # Find out how many epochs you expect
    # per source
    mjds = get_unique_epochs(lcfs)
    num_mjds = len(mjds)
    
    all_fluxes = []
    for l, lcf in enumerate(lcfs):
        # Read the light curve file
        lc = pd.read_csv(lcf)
        # Make sure the information
        # is in chronological order
        lc = lc.sort_values(by=[mjd_col])
        
        # Check if the source has the right number of
        # MJDs (they all should, but it's a useful check)
        if len(lc[flux_col]) == num_mjds:
            all_fluxes.append(list(np.array(lc[flux_col])))
        else:
            print(('Wrong number of mjds? '
                   'Should be {0}, is {1}').format(num_mjds,
                                                   len(lc[flux_col])))
            print('lc with wrong number of mjds:\n{}'.format(lcf))
            pass

    # Turn the list of source fluxes into an array
    all_fluxes = np.array(all_fluxes)
    
    # If no reference epoch is specified, work
    # out which epoch has the highest number
    # of detected sources and use that
    if ref_epoch == 'None':
        num_nans = np.zeros(len(all_fluxes[0]))
        for c, col in enumerate(all_fluxes.T):
            nans = np.where(np.isnan(col))
            num_nans[c] = len(nans[0])
        ref_epoch = np.nanargmin(num_nans)
    print('Reference epoch is epoch {}'.format(ref_epoch))
    
    # Divide the light curves of every source
    # by the flux of the reference epoch for
    # each source (i.e. each source is divided
    # by it's own reference epoch flux)
    bob = all_fluxes[:, ref_epoch]
    bob = np.expand_dims(bob, axis=0)
    bob = bob.T
    
    # Return the array of source flux
    # densities that have been scaled
    # to the reference epoch
    return all_fluxes / bob


def get_epoch_offsets(scaled_fluxes,
                      mjds,
                      image_folder,
                      database,
                      dataset_id,
                      ext,
                      outlier_sigmas=3.0,
                      pc_cut='None',
                      close_all=True):
    '''
    Work out the light curve systematics
    
    This function takes the flux density array
    from scale_fluxes and uses it to find out
    the systematic offset for each epoch.
    Specifically, it takes the distribution
    of scaled fluxes for each epoch and finds
    the mean, standard deviation, median,
    and median absolute deviation and returns
    a pandas dataframe with thse values and the
    corresponding mjds
    
    Args:
    scaled_fluxes (array): the array of scaled fluxes
                           made by the scale_fluxes function
    mjds (array): The mjds for each epoch in an array
    image_folder (str): This function makes and couple
                        of plots, this is a string of the
                        path where you'd like to save
                        those plots
    database (str): The name of the TraP database, this will
                    be used in the plot filenames
    dataset_id (int): the number of the TraP dataset_id, this
                      be used in the plot filenames
    ext (str): added extension at the end of file names
               (for both csvs and pngs)
    
    kwargs:
    outlier_sigmas (float): The number of sigma away source
                            has to be from the mean to be removed
                            from the calculation of the mean
                            and standard deviation. This
                            is the avoid skewing the analysis
                            by real variable/transient sources
                            Default=3.0
    close_all (bool): if True all the plots will be closed
                      Default: True
    Returns:
    A pandas data frame with:
    'mu': mean per epoch, 'std': standard deviation per epoch,
    'median': median per epoch, 'mad': median absolute deviation per epoch,
    'mjd': mjds

    The median absolute deviation (MAD) is produced by
    scipy.stats.median_absolute_deviation which includes
    a default scale of 1.4826 to ensure "consistency with
    the standard deviation for normally distributed data."
    '''
    # Set up an empty array to put the mean, std,
    # median and MAD
    models = np.zeros((len(scaled_fluxes[0]), 4))

    # Go through by epoch
    for column, col in enumerate(scaled_fluxes[0]):
        fluxes = scaled_fluxes[:, column]
        # Remove nans as the fitting won't allow them
        fluxes = fluxes[np.logical_not(np.isnan(scaled_fluxes[:,
                                                              column]))]
        # Get the median scaled flux for this epoch
        median = np.nanmedian(fluxes)
        # Get the MAD for the scaled flux for this epoch
        # Making sure to divide by the sqrt(n) to make it
        # the population MAD
        mad = (spstats.median_absolute_deviation(fluxes,
                                                 nan_policy='omit')/
               np.sqrt(len(fluxes)))

        # Remove initial outliers as these are likely variable
        # and will have a strong affect on the mean and std
        pre_mu, pre_std = spstats.norm.fit(fluxes)
        plus3sig = np.where(fluxes>(pre_mu + outlier_sigmas*pre_std))[0]
        minus3sig = np.where(fluxes<(pre_mu - outlier_sigmas*pre_std))[0]
        outliers = np.concatenate((plus3sig, minus3sig))
        fluxes_prepped = np.delete(fluxes, outliers, axis=0)

        # Check that this epoch has measurements and
        # isn't just nans
        if len(fluxes_prepped) == 0:
            print('***********************************')
            print('No sources for column: {}'.format(column))
            print('***********************************')
            mu = np.nan
            std = np.nan
        else:
            # Get the mean and standard deviation
            mu = np.nanmean(fluxes_prepped)
            std = np.nanstd(fluxes_prepped)
            num_not_nan = len(fluxes_prepped)
            # Make sure you use the population
            # standard deviation (divide by sqrt n)
            pop_std = std / np.sqrt(num_not_nan)

        # Put the mean, std, median, and mad for this epoch
        # in the prepared blank array
        models[column] = [mu, pop_std, median, mad]

        # Plot the Gaussian for the epoch and fit results
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        a = ax.hist(fluxes_prepped,
                    bins=20,
                    density=True,
                    facecolor='LightBlue')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 10)

        ax.plot(np.ones(10)*mu, y, '--',
                c='DarkGrey')

        p = spstats.norm.pdf(x, mu, std)
        ax.plot(x, p, linewidth=2, c='RoyalBlue')

        ax.set_ylabel('Count', fontsize=18)
        ax.set_xlabel('Epoch {}: flux/ref_flux'.format(column),
                      fontsize=18)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        atext = AnchoredText(('Mean: {0:.4f}\n'
                              'StD: {1:.4f}\n'
                              'Median: {2:.4f}\n'
                              'MAD: {3:.4f}\n').format(mu,
                                                       std,
                                                       median,
                                                       mad),
                             loc='upper right',
                             prop=dict(fontsize=16))
        ax.add_artist(atext)

        if column < 10:
            epoch_str = '00{}'.format(column)
        elif column < 100:
            epoch_str = '0{}'.format(column)
        else:
            epoch_str = str(column)

        if pc_cut == 'None':
            fig.savefig(('{0}EpochGaussian_Epoch{1}_'
                         'db{2}_ds{3}_{4}.png').format(image_folder,
                                                   epoch_str,
                                                   database,
                                                   dataset_id, ext),
                        bbox_inches='tight')
        else:
            fig.savefig(('{0}EpochGaussian_Epoch{1}_'
                         'db{2}_ds{3}_'
                         'pccut{4:.3f}deg_{5}.png').format(image_folder,
                                                       epoch_str,
                                                       database,
                                                       dataset_id, pc_cut, ext),
                        bbox_inches='tight')
        if close_all:
            plt.close(fig)
        else:
            if column % 10 != 0:
                plt.close(fig)

    # Plot the means and medians for every epoch
    # These are the models for the
    # systematics
    fig, ax = plt.subplots(2, 1, figsize=(12, 12),
                           sharex=True, sharey=True)

    ax[0].errorbar(mjds, models[:, 0],
                yerr=models[:, 1],
                fmt='o', c='#ff87ab',
                label='Mean and std', elinewidth=5, ms=10)
    ax[1].errorbar(mjds, models[:, 2],
                yerr=models[:, 3],
                fmt='o', c='#6D98BA', label='Median and mad')

    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()

    xs = np.linspace(xmin, xmax, 10)
    for a in range(2):
        ax[a].plot(xs, np.ones(10), '--', c='DarkGrey', zorder=0)

        ax[a].set_xlim(xmin, xmax)
        ax[a].set_ylim(ymin, ymax)
        ax[a].legend()
    ax[0].set_ylabel('Epoch mean', fontsize=18)
    ax[1].set_ylabel('Epoch median', fontsize=18)
    ax[1].set_xlabel('MJD', fontsize=18)

    if pc_cut == 'None':
        fig.savefig(('{0}EpochGaussian_AllEpochs_'
                     'db{2}_ds{3}_{4}.png').format(image_folder,
                                               column,
                                               database,
                                               dataset_id, ext),
                    bbox_inches='tight')
    else:
        fig.savefig(('{0}EpochGaussian_AllEpochs_'
                     'db{2}_ds{3}_'
                     'pccut{4:.3f}deg_{5}.png').format(image_folder,
                                                   column,
                                                   database,
                                                   dataset_id,
                                                   pc_cut, ext),
                    bbox_inches='tight')
    if close_all:
        plt.close(fig)

    # Return the values as a pandas dataframe
    df_data = {'mu': models[:, 0],
               'std': models[:, 1],
               'median': models[:, 2],
               'mad': models[:, 3],
               'mjd': mjds}
    return pd.DataFrame(data=df_data)


def scale_lightcurves(lcfs, scale_models,
                      database, dataset_id,
                      ref_epoch, csv_savepath,
                      ext,
                      flux_col = 'f_int',
                      mjd_col='mjd',
                      ra_col='ra',
                      dec_col='decl',
                      pc_cut='None'):
    '''
    Apply the scale models to all light curves
    
    This function applies the scale models found
    using get_epoch_offsets. It applies both
    the median and mean model, putting them into
    different columns. It also saves the models
    themselves in the tables for each light curve.
    The final light curve files will have
    the new columns:
    'ref scale model mean': the systematics model
                            made from the mean of
                            each epoch
    'ref scale model std': the population standard
                           devation uncertanties
                           for the mean scale model
    'ref scale model median': the systematics model
                              made from the median of
                              each epoch
    'ref scale model mad': the population MAD uncertainties
                           for the median scale model
    flux_col+'_mean_scaled': the source fluxes scaled
                             using the mean model (Jy)
    flux_col+'_median_scaled': the source fluxes scaled
                               using the median model (Jy)
    flux_col+'_mean_scaled_err': the uncertainties on the
                                 source fluxes scaled by
                                 the std (mean model)
    flux_col+'_median_scaled_err': the uncertainties on the
                                   source fluxes scaled by
                                   the MAD (median model)
    'mean_scaled_V': V parameter of the mean model scaled fluxes
    'median_scaled_V': V parameter of the median model scaled fluxes
    'mean_scaled_eta': eta parameter of the mean model scaled fluxes
    'median_scaled_eta': eta parameter of the median model scaled fluxes
    'mean_scaled_madp': MAD parameter of the mean model scaled fluxes
    'median_scaled_madp': MAD parameter of the median model scaled fluxes
    The best way to read and use the light curve files is using pandas.
    
    Args:
    lcfs: a list of the light curve/source info files (including paths)
    scale_models: the scale model pandas date frame from
                  get_epoch_offsets
    database: the name of the TraP database
    dataset_id: the name of the TraP dataset_id
    ref_epoch: the epoch used as the reference epoch
    csv_savepath: where you want to save the final, updated
                  light curve/source info csvs
    close_all (bool): if True all the plots will be closed
                      Default: True
    
    kwargs:
    flux_col (str): The name of the flux column
                    Default ='f_int'
    ra_col (str): The name of the riht ascension column
                  Default = 'ra'
    dec_col (str): The name of the declination column
                   Default = 'decl'
    mjd_col (str): The name of the mjd column
                   Default = 'mjd'
                   
    Returns:
    a list of the new light curves info pandas frames
    (i.e. one panda frame per source)
    '''
    # Make an empty list to put the updated
    # light curve data frames for later use
    new_lcs = []

    # Loop through each source file
    for l, lcf in enumerate(lcfs):
        lc = pd.read_csv(lcf)
        # Make sure the light curve is in
        # chronological order
        lc = lc.sort_values(by=[mjd_col])

        # Add columns to include the models as they are
        lc['ref scale model mean'] = scale_models['mu']
        lc['ref scale model std'] = scale_models['std']
        lc['ref scale model median'] = scale_models['median']
        lc['ref scale model mad'] = scale_models['mad']
        # Scale the flux densities by the scale models
        # and put the values in new columns so that you
        # can also keep the original fluxes
        lc[flux_col+'_mean_scaled'] = lc[flux_col] * (1./scale_models['mu'])
        lc[flux_col+'_median_scaled'] = lc[flux_col] * (1./scale_models['median'])

        # Scale the uncertainties and put those in new columns too
        new_flux_uncertainty_mean = (lc[flux_col+'_mean_scaled'] *
                                ((lc['{}_err'.format(flux_col)]/lc[flux_col]) +
                                 (scale_models['std']/scale_models['mu'])))
        lc[flux_col+'_mean_scaled_err'] = new_flux_uncertainty_mean
        
        new_flux_uncertainty_median = (lc[flux_col+'_median_scaled'] *
                                       ((lc['{}_err'.format(flux_col)]/lc[flux_col]) +
                                        (scale_models['mad']/scale_models['median'])))
        lc[flux_col+'_median_scaled_err'] = new_flux_uncertainty_median

        # When you calculate the new/scaled variability
        # parameters, you have to ditch the reference
        # epoch, because that will skew your variability
        # parameters. So here we make arrays where the
        # reference epoch has been removeed, so we can
        # calculate the new variability parameters
        allowed_fluxes_mean = np.array(lc[flux_col+'_mean_scaled'])
        allowed_fluxes_mean = np.delete(allowed_fluxes_mean,
                                        [int(ref_epoch)], 0)
        allowed_fluxes_mean_errs = np.array(lc[flux_col+'_mean_scaled_err'])
        allowed_fluxes_mean_errs = np.delete(allowed_fluxes_mean_errs,
                                             [int(ref_epoch)], 0)
        
        allowed_fluxes_median = np.array(lc[flux_col+'_median_scaled'])
        allowed_fluxes_median = np.delete(allowed_fluxes_median,
                                          [int(ref_epoch)], 0)
        allowed_fluxes_median_errs = np.array(lc[flux_col+'_median_scaled_err'])
        allowed_fluxes_median_errs = np.delete(allowed_fluxes_median_errs,
                                               [int(ref_epoch)], 0)

        # Now calculate the new variability parameters for
        # both the median and mean scaled fluxes
        Vs_mean = np.repeat(modulation_parameter(allowed_fluxes_mean),
                            len(lc))
        lc['mean_scaled_V'] = Vs_mean                            
        Vs_median = np.repeat(modulation_parameter(allowed_fluxes_median),
                              len(lc))
        lc['median_scaled_V'] = Vs_median

        try:
            etas_mean = np.repeat(chi2_parameter(allowed_fluxes_mean,
                                                 allowed_fluxes_mean_errs),
                                  len(lc))
        except ZeroDivisionError:
            print('source: ', l, 'etas error')
            etas_mean = np.repeat(np.nan, len(lc))
        lc['mean_scaled_eta'] = etas_mean

        try:
            etas_median = np.repeat(chi2_parameter(allowed_fluxes_median,
                                                   allowed_fluxes_median_errs),
                                    len(lc))
        except ZeroDivisionError:
            print('source: ', l, 'etas error')
            etas_median = np.repeat(np.nan, len(lc))
        lc['median_scaled_eta'] = etas_median

        madps_mean = np.repeat(madulation_parameter(allowed_fluxes_mean)[0], len(lc))
        lc['mean_scaled_madp'] = madps_mean
        madps_median = np.repeat(madulation_parameter(allowed_fluxes_median)[0], len(lc))
        lc['median_scaled_madp'] = madps_median

        # Put the light curve into the prepared list,
        # and save the data as a csv
        new_lcs.append(lc)
        if pc_cut == 'None':
            savename = ('{0}rcat{1}_'
                        'ra{2:.5f}_dec{3:.5f}_'
                        'db{4}_ds{5}_'
                        '{6}.csv').format(csv_savepath,
                                         lc['runcat'][0],
                                         np.nanmean(lc[ra_col]),
                                         np.nanmean(lc[dec_col]),
                                         database, dataset_id, ext)
            
        else:
            savename = ('{0}rcat{1}_'
                        'ra{2:.5f}_dec{3:.5f}_'
                        'db{4}_ds{5}_'
                        'pccut{6:.3f}deg_'
                        '{7}.csv').format(csv_savepath,
                                         lc['runcat'][0],
                                         np.nanmean(lc[ra_col]),
                                         np.nanmean(lc[dec_col]),
                                         database, dataset_id,
                                         pc_cut, ext)
        lc.to_csv(savename)

    return new_lcs


def get_variability_parameters(new_lcs, csv_savepath,
                               database,
                               dataset_id,
                               ext,
                               pc_cut='None',
                               min_sn=2):
    '''
    Calculate the new variability parameters
    for the scaled light curves
    
    Get the eta, V and MAD variability parameters
    for the scaled light curves (both median and mean
    scaled)
    
    Args:
    new_lcs (list): list of filenames (including the path)
                    of the scaled light curve files
    database (str): the name of the TraP database
    dataset_id (int): the dataset number from TraP
    
    kwargs:
    pc_cut (float): the distance from the phase centre
                    in degrees within which sources
                    will be included in the analysis
                    Default: 'None' (all sources included)
    min_sn (float): the minimum allowed signal
                    to noise
                    Default: 2
    
    Returns:
    A pandas dataframe with the new and old variability
    parameters for each source
    The dataframe is also saved as a csv
    '''
    variability_parameters = np.zeros((len(new_lcs), 14))

    source_detected_2 = []
    source_detected_3 = []
    for nl, nlc in enumerate(new_lcs):
        flux = nlc['f_int']
        flux_errs = nlc['f_int_err']

        # Work out whether the source is
        # "detected" to include this in the
        # dataframe
        signal_to_noise = flux / flux_errs
        if (np.nanmax(signal_to_noise) > 3):
            source_detected_3.append('Yes')
            source_detected_2.append('Yes')
        elif(np.nanmax(signal_to_noise) > 2):
            source_detected_2.append('Yes')
            source_detected_3.append('No')
        else:
            source_detected_2.append('No')
            source_detected_3.append('No')

        variability_parameters[nl] = [nlc['runcat'][0], nlc['freq_eff_int'][0],
                                      np.nanmedian(nlc['f_int']),
                                      np.nanmedian(nlc['f_int_mean_scaled']),
                                      np.nanmedian(nlc['f_int_median_scaled']),
                                      nlc['V_param'][0],
                                      nlc['mean_scaled_V'][0],
                                      nlc['median_scaled_V'][0],
                                      nlc['eta_param'][0],
                                      nlc['mean_scaled_eta'][0],
                                      nlc['median_scaled_eta'][0],
                                      nlc['mad_param'][0],
                                      nlc['mean_scaled_madp'][0],
                                      nlc['median_scaled_madp'][0]]

    # Set up the data to turn into a dataframe
    data_dict = {'runcat': variability_parameters[:, 0],
                 'freq_eff_int': variability_parameters[:, 1],
                 'f_int_median': variability_parameters[:, 2],
                 'f_int_mean_scaled_median': variability_parameters[:, 3],
                 'f_int_median_scaled_median': variability_parameters[:, 4],
                 'V_param': variability_parameters[:, 5],
                 'mean_scaled_V': variability_parameters[:, 6],
                 'median_scaled_V': variability_parameters[:, 7],
                 'eta_param': variability_parameters[:, 8],
                 'mean_scaled_eta': variability_parameters[:, 9],
                 'median_scaled_eta': variability_parameters[:, 10],
                 'mad_param': variability_parameters[:, 11],
                 'mean_scaled_madp': variability_parameters[:, 12],
                 'median_scaled_madp': variability_parameters[:, 13],
                 'detected SN2':source_detected_2,
                 'detected SN3':source_detected_3}

    # Save the data frame
    var_param_frame = pd.DataFrame(data=data_dict)
    if pc_cut == 'None':
        var_param_frame.to_csv('{0}VariabilityParams_db{1}_'
                               'ds{2}_{3}.csv'.format(csv_savepath,
                                                      database,
                                                      dataset_id, ext))
    else:
        var_param_frame.to_csv(('{0}VariabilityParams_'
                                'db{1}_ds{2}_'
                                'pccut{3:.3f}deg_'
                                '{4}.csv').format(csv_savepath,
                                                  database,
                                                  dataset_id,
                                                  pc_cut, ext))
    return var_param_frame


def plot_var_params(var_param_frame,
                    image_folder, 
                    database,
                    dataset_id,
                    ext,
                    pc_cut='None',
                    close_all=True):
    '''
    Plot the variability parameters
    
    This plots the variability parameters
    before and after correction
    
    Args:
    var_param_frame (pandas dataframe): the pandas dataframe
                                        containing the variability
                                        parameter information
    image_folder (str): the path to where you'd like to save
                        the plot
    database (str): the name of the TraP database
    dataset_id (int): the dataset number from TraP
    
    kwargs:
    pc_cut (float): the distance from the phase centre
                    in degrees within which sources
                    will be included in the analysis
                    Default: 'None' (all sources included)
    '''
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    ax[0].scatter(np.log10(var_param_frame['f_int_median']),
                  np.log10(var_param_frame['eta_param']),
                  marker='.', c='DarkGrey',
                  label='Original')
    ax[0].scatter(np.log10(var_param_frame['f_int_mean_scaled_median']),
                  np.log10(var_param_frame['mean_scaled_eta']),
                  marker='.', c='#FF87AB',
                  label='Scaled', s=200)

    ax[1].scatter(np.log10(var_param_frame['f_int_median']),
                  np.log10(var_param_frame['V_param']),
                  marker='.', c='DarkGrey',
                  label='Original')
    ax[1].scatter(np.log10(var_param_frame['f_int_mean_scaled_median']),
                  np.log10(var_param_frame['mean_scaled_V']),
                  marker='.', c='#FF87AB',
                  label='Scaled', s=200)

    ax[2].scatter(np.log10(var_param_frame['f_int_median']),
                  np.log10(var_param_frame['mad_param']),
                  marker='.', c='DarkGrey',
                  label='Original')
    ax[2].scatter(np.log10(var_param_frame['f_int_mean_scaled_median']),
                  np.log10(var_param_frame['mean_scaled_madp']),
                  marker='.', c='#FF87AB',
                  label='Mean scaled', s=200)

    ax[0].scatter(np.log10(var_param_frame['f_int_median_scaled_median']),
                  np.log10(var_param_frame['median_scaled_eta']),
                  marker='.', c='#6D98BA',
                  label='Median scaled', s=80)

    ax[1].scatter(np.log10(var_param_frame['f_int_median_scaled_median']),
                  np.log10(var_param_frame['median_scaled_V']),
                  marker='.', c='#6D98BA',
                  label='Median scaled', s=80)

    ax[2].scatter(np.log10(var_param_frame['f_int_median_scaled_median']),
                  np.log10(var_param_frame['median_scaled_madp']),
                  marker='.', c='#6D98BA',
                  label='Median scaled', s=80)

    for a in range(3):
        ax[a].set_xlabel(r'log$_{10}\left(\mathrm{integrated\,flux\,[Jy]} \right)$', fontsize=16)
    ax[0].set_ylabel(r'log$_{10}\left(\eta \right)$', fontsize=16)
    ax[1].set_ylabel(r'log$_{10}\left(V \right)$', fontsize=16)
    ax[2].set_ylabel(r'log$_{10}\left(MAD \right)$', fontsize=16)

    ax[2].legend(fontsize=12, loc='best')
    
    if close_all:
        plt.close(fig)

    fig.tight_layout()

    if pc_cut == 'None':
        fig.savefig(('{0}EpochGaussian_VariabilityParams_'
                     '_db{1}_ds{2}_{3}.png').format(image_folder, 
                                                database,
                                                dataset_id, ext),
                    bbox_inches='tight')
    else:
        fig.savefig(('{0}EpochGaussian_VariabilityParams_'
                     '_db{1}_ds{2}_'
                     'pccut{3:.3f}deg_{4}.png').format(image_folder,
                                                   database,
                                                   dataset_id, pc_cut),
                    bbox_inches='tight')
    
    if close_all:
        plt.close(fig)


if __name__ in '__main__':
    # The name of the TraP database
    database = 'Laura_NWLayers_ReProc'

    # Paths to light curve files
    lc_path = ('/raid/driessen/FlareStars/'
               'GX339/Source_Light_Curves/')
    # Path to where to save the plots
    image_folder = ('/raid/driessen/FlareStars/'
                    'GX339/EpochScaling/')
    # Path to where to save the CSV files
    csv_savepath = ('/raid/driessen/FlareStars/'
                    'GX339/Source_Light_Curves/'
                    'Average_Scaled/')

    # Path to the file containing the
    # coordinates of known resolved
    # sources and artefacts
    extended_sources = ('/raid/driessen/FlareStars/'
                        'GX339/2021.01.18_GX339_'
                        'KnownExtendedSources.npy')

    # Column names for light curve files
    flux_col = 'f_int'
    flux_err_col = 'f_int_err'
    mjd_col = 'mjd'
    ra_col = 'ra'
    dec_col = 'decl'
    freq_col = 'freq_eff_int'

    # Minimum required signal
    # to noise
    min_signaltonoise = 3.

    # Dataset IDs for TraP and their corresponding
    # subband frequencies
    dataset_ids = [49, 52, 53, 50, 54, 55, 51, 33]
    freqs = [1658, 1551, 1444, 1337, 1123, 1016, 909, 'MFS']

    # Primary beam sizes
    # (Not used here unless
    # code is edited)
    pbs = []
    for freq in freqs[:-1]:
        pbs.append(np.rad2deg(np.arcsin((1.22*(3e8/(freq*1.0e6)))/13.9))/2.)

    # Coordinates of known variables
    # MKT J170456.2-482100, PSR J1703-4851,
    # and GX 339-4
    known_variables = SkyCoord(np.array([[256.23450507,
                                          -48.35008655],
                                         [255.97732915,
                                          -48.86685808],
                                         [255.70567674,
                                          -48.78963475]]), unit=(un.degree,
                                                                 un.degree))

    for d, dataset_id in enumerate(dataset_ids):
        for cut_extended in [True, False]:
            # Get the light curve files that are from
            # TraP
            lcs_original = glob.glob(('{0}rcat*_'
                                      'ra*_dec*_'
                                      'db{1}_ds{2}_'
                                      '*Source.csv').format(lc_path,
                                                            database,
                                                            dataset_id))

            freq = pd.read_csv(lcs_original[1])[freq_col][0]
            print('*******************')
            print('Freq: {}'.format(freq))
            print('*******************')

            print('Total number of light curves: {}'.format(len(lcs_original)))

            # Choose your reference epoch
            mjds = get_unique_epochs(lcs_original)
            print('Number of epochs: {}'.format(len(mjds)))
            ref_epoch = len(mjds) - 1
            print('Reference epoch: {}'.format(ref_epoch))

            # Remove the extended sources and artefacts
            if cut_extended:
                lcs = filter_extended(lcs_original,
                                      extended_sources)
                ext = 'PS'
                print('*** Point sources only ***', ext)
            else:
                lcs = lcs_original
                ext = 'ES'
                print('*** All sources ***', ext)
            print(('Number of light curves after '
                   'extended cut: {}').format(len(lcs)))

            # Cut sources below a specified signal to noise
            lcs_cut_sn = filter_sn(lcs, min_signaltonoise)
            print(('Number of light curves after '
                   'S/N cut: {}').format(len(lcs_cut_sn)))

            # Remove known variable sources
            lcs_cut_vars = filter_coords(lcs_cut_sn, known_variables)
            print(('Number of light curves after '
                   'known variable cut: {}').format(len(lcs_cut_vars)))

            # Scale the light curves by the flux density of
            # the refence epoch
            scaled_fluxes = scale_fluxes(lcs_cut_vars,
                                         ref_epoch=ref_epoch)
            print('Scaled fluxes: ', np.shape(scaled_fluxes))

            # Determine the scale models
            scale_models = get_epoch_offsets(scaled_fluxes,
                                             mjds,
                                             image_folder,
                                             database,
                                             dataset_id,
                                             ext)

            # Scale each light curve by the models
            new_lcs = scale_lightcurves(lcs, scale_models,
                                        database, dataset_id,
                                        ref_epoch, csv_savepath,
                                        ext)

            # Make some data frames containing the variability
            # parameter information of each source (this is
            # useful for plottin etc.)
            var_param_frame = get_variability_parameters(new_lcs, csv_savepath,
                                                         database,
                                                         dataset_id,
                                                         ext,
                                                         min_sn=2.)

            # Plot the variability parameters
            plot_var_params(var_param_frame,
                            image_folder, 
                            database,
                            dataset_id,
                            ext,
                            close_all=True)

            # Make a plot of the models so that you can
            # check that nothing has gone obviously
            # wrong
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.errorbar(scale_models['mjd'], scale_models['median'],
                        yerr=scale_models['mad'],
                        fmt='.', c='HotPink', label='Median and MAD')
            ax.errorbar(scale_models['mjd'], scale_models['mu'],
                        yerr=scale_models['std'],
                        fmt='.', c='Pink', label='Mean and STD')

            ax.legend()
            ax.set_xlabel('MJD', fontsize=18)
            ax.set_ylabel('Model', fontsize=18)
            ax.set_title('{0}MHz {1}'.format(freq, ext))
            fig.tight_layout()

            print('----------------------------------------------------')
