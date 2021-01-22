# L. N. Driessen
# Last updated: 2021.01.22

import numpy as np
import pandas as pd
import glob

from astropy import units as un
from astropy.coordinates import SkyCoord
import astropy.time
from astropy.io import fits
from astropy import nddata
from astropy.wcs import WCS

import tkp.config
import sqlalchemy
from sqlalchemy import *
from sqlalchemy.orm import relationship
import tkp.db
import logging
logging.basicConfig(level=logging.INFO)
query_loglevel = logging.WARNING  # Set to INFO to see queries, otherwise WARNING
import sys
sys.path.append('../')
sys.path.append('/scratch/ldriessen/TraP_tools')
sys.path.append('/scratch/ldriessen/TraP_tools/databaseTools/')
sys.path.append('/scratch/ldriessen/TraP_tools/tools/')
sys.path.append('/scratch/ldriessen/TraP_tools/exampleScripts/')
sys.path.append('/scratch/ldriessen/TraP_tools/plotting/')
from dblogin import * # This file contains all the variables required to connect to the database
from databaseTools import dbtools
from tools import tools
from plotting import plot_varib_params as pltvp


def madulation_parameter(data):
    '''
    Use the median and mad to make a modulation-type parameter.

    Args:
    data (array): the 1D array
                  of data that you
                  want to find the
                  MAD parameter of
    Returns:
    (madvalue, madlocation)
    The value of the MAD parameter
    and the index of the epoch
    of the MAD parameter
    '''
    me = np.nanmedian(data)
    ma = np.nanmedian(np.absolute(data-np.nanmedian(data)))
    
    data_ = np.copy(data)
    
    shifted_data = np.abs(data_ - me)
    divided_by_mad = shifted_data/ma
    
    try:
        madvalue = np.nanmax(divided_by_mad)
        madlocation = np.nanargmax(divided_by_mad)
    except ValueError:
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


def run_query(transients_query, dbname, dataset_id,
              engine, host, port, user, pword):
    '''
    Function to get information from TraP directly.
    
    This function accesses a TraP database and gets
    the information (in transients query).
    
    Args:
    transients query: This is a weird format thing
                      that contains the keywords for
                      what you would like from TraP.
                      For example:
        transients_query = """SELECT 
        ex.f_peak,
        ex.f_peak_err,
        ex.ra,
        ex.ra_err,
        ex.decl,
        ex.decl_err
        FROM extractedsource ex, image im 
        WHERE ex.id IN ( select xtrsrc FROM assocxtrsource WHERE runcat = {}) 
        AND ex.image = im.id
        ORDER BY ex.det_sigma DESC;
        """.format(runcat)
    dbname (str): the name of the TraP database you'd like to access
    dataset_id (int): the number of the data set in the database that you'd
                      like to access
    engine (str): a TraP thing, going to be 'postgresql'
    host (str): the host url ('vlo.science.uva.nl')
    port (int): the port to connect to (5432)
    user (str): your TraP username
    pword (str): your TraP password
    
    Returns:
    A Pandas table containing the information you requested
    '''
    tkp.db.Database(
        database=dbname, user=user, password=pword,
        engine=engine, host=host, port=port
    )

    cursor = tkp.db.execute(transients_query, (dataset_id,))
    transients = tkp.db.generic.get_db_rows_as_dicts(cursor)
    return transients


def get_lightcurves(phase_centre_ra,
                    phase_centre_dec,
                    min_dpts,
                    engine,
                    host,
                    port,
                    user,
                    password,
                    database,
                    dataset_id,
                    image_values,
                    drop_mjd='None',
                    flux_col='f_int'):
    '''
    Get the light curves from a TraP dataset and add some parameters.
    
    This script accesses the banana database and gets the light curves
    from the database as pandas tables. It makes a list of all the light
    curve tables from the dataset.
    
    Args:
    phase_centre_ra (float): the Right Ascension of the phase centre of
                             the images you've processed with TraP
                             in degrees
    phase_centre_dec (float): the Declination of the phase centre of the
                              the images you've processed with TraP
                              in degrees
    mid_dpts (int): the minimum number of detections of the source
                    by TraP
    dbname (str): the name of the TraP database you'd like to access
    dataset_id (int): the number of the data set in the database that you'd
                      like to access
    engine (str): a TraP thing, going to be 'postgresql'
    host (str): the host url ('vlo.science.uva.nl')
    port (int): the port to connect to (5432)
    user (str): your TraP username
    pword (str): your TraP password
    
    Kwargs:
    drop_mjd (int/float): the MJD of the epoch you want
                          to exclude completely. This is
                          important if you have artificially
                          included a deep image at the start
                          of the TraP run.
                          Default: 'None' (no epoch is removed)
    flux_col (str): the nameof the flux column you want to
                    use. This is important because TraP records
                    the integrated ('f_int') and peak ('f_peak')
                    flux densities for all the sources.
                    Default: 'f_int'
    
    Returns:
    A list of pandas tables, where each pandas table contains
    information about a source from TraP
    '''
    # The coordinates of the phase centre of the image    
    phase_centre = SkyCoord(phase_centre_ra*un.degree,
                            phase_centre_dec*un.degree,
                            frame='icrs')

    # Get the basic values for each source in the dataset
    # by accessing the TraP
    session = dbtools.access(engine,host,port,user,password,database)
    VarParams = dbtools.GetVarParams(session, dataset_id)
    plotdata = [[VarParams[i].Runningcatalog.id,
                 VarParams[i].Varmetric.eta_int,
                 VarParams[i].Varmetric.v_int,
                 VarParams[i].Varmetric.lightcurve_max,
                 VarParams[i].Varmetric.lightcurve_median,
                 (VarParams[i].Varmetric.band.freq_central/1e6),
                 VarParams[i].Runningcatalog.datapoints,
                 VarParams[i].Varmetric.newsource,
                 VarParams[i].Runningcatalog.wm_ra,
                 VarParams[i].Runningcatalog.wm_decl]
                for i in range(len(VarParams))]

    plotdata = pd.DataFrame(data=plotdata,columns=['runcat',
                                                   'eta',
                                                   'V',
                                                   'maxFlx',
                                                   'avgFlx',
                                                   'freq',
                                                   'dpts',
                                                   'newSrc',
                                                   'ra',
                                                   'dec'])
    plotdata = plotdata.fillna('N')
    plotdata = plotdata.loc[(plotdata['dpts']>=min_dpts)]
    num_measurements = np.argmax(np.array(plotdata['dpts']))

    runcats = np.array(plotdata['runcat'])
    Vs = np.array(plotdata['V'])
    etas = np.array(plotdata['eta'])

    del session
    
    # Connect a new session to the TraP database
    db = tkp.db.Database(engine=engine, host=host, port=port,
                         user=user, password=password,
                         database=database)
    db.connect()
    session = db.Session()

    # Set up to make a list of source panda tables
    source_lightcurves = []
    # Get the light curve and other information
    # for each running catalogue (source)
    for r, runcat in enumerate(runcats):
        # These are the values for the source that
        # will be in its pandas table
        transients_query = """SELECT
        ex.f_int,
        ex.f_int_err,
        ex.f_peak,
        ex.f_peak_err,
        ex.ra,
        ex.ra_err,
        ex.decl,
        ex.decl_err,
        im.band,
        im.freq_eff,
        im.taustart_ts
        FROM extractedsource ex, image im 
        WHERE ex.id IN ( select xtrsrc FROM assocxtrsource WHERE runcat = {}) 
        AND ex.image = im.id
        ORDER BY ex.det_sigma DESC;
        """.format(runcat)
        # Get the pandas table
        source_data = pd.DataFrame(run_query(transients_query,
                                             database,
                                             dataset_id, 
                                             engine, host,
                                             port, user,
                                             password))
        # The centre frequency can vary by a couple
        # of kHz depending on whether the data were taken
        # in 1k, 4k, or 32k mode. Using the integer
        # flux density helps keep everything together
        # even if the centre frequency isn't exactly the same
        source_data['freq_eff_int'] = list(np.array(source_data['freq_eff']*1.0e-6).astype(int))

        # Get the epochs of the source and convert them
        # into isot and mjd formats
        taustart = np.array(source_data['taustart_ts']).astype(str)
        isot_times = astropy.time.Time(taustart, format='isot')

        # Add columns to the source pandas table
        # for the MJD, isot time, and frequency
        source_data['mjd'] = isot_times.mjd
        source_data['taustart_ts'] = taustart

        # Set up a new pandas table so that you can
        # sort the table the way you want to
        mjds_ = []
        isots_ = []
        freqs_ = []
        int_freqs_ = []
        for im in image_values:
            if ((source_data['freq_eff_int'] == int(im[0])) &
                (source_data['mjd'] == im[1])).any():
                pass
            else:
                new_row = dict()
                for col in source_data.columns:
                    new_row[col] = np.nan
                    if col == 'mjd':
                        new_row[col] = im[1]
                    if col == 'freq_eff':
                        new_row[col] = im[0]*1.0e6
                    if col == 'freq_eff_int':
                        new_row[col] = int(im[0])
                    if col == 'taustart_ts':
                        new_row[col] = astropy.time.Time(im[1], format='mjd').isot
                source_data = source_data.append(new_row, ignore_index=True)

        len_col = np.ones(len(source_data['ra']))
        all_nans = len_col * np.nan

        # Get the mean RA and DEC of the source
        ras = np.array(source_data['ra'])
        decs = np.array(source_data['decl'])
        c = SkyCoord(ras*un.degree,
                     decs*un.degree,
                     frame='icrs')
        # Find the distance to the phase centre and add it to the table
        dist_to_phase_centre = phase_centre.separation(c)
        source_data['dist_to_pc_DEG'] = dist_to_phase_centre.deg

        # Add the runcat of the source to the table
        source_data['runcat'] = (len_col*runcat).astype(int)
        
        # Sort the final table by MJD and frequency
        source_data = source_data.sort_values(by=['freq_eff_int', 'mjd'])
        source_data = source_data.drop_duplicates(subset=['mjd'])

        # If you included the deep image at the
        # start of the TraP run, remove
        # that epoch completely and re-calculate
        # the variability parameters
        if drop_mjd == 'None':
            # Make columns for the V and eta parameters
            try:
                source_data['V_param'] = Vs[r]
            except TypeError:
                print('V: ', Vs[r], type(Vs[r]))
                source_data['V_param'] = np.nan

            try:
                source_data['eta_param'] = etas[r]
            except TypeError:
                print('eta: ', etas[r], type(etas[r]))
                source_data['eta_param'] = np.nan

            madval = madulation_parameter(np.array(source_data['f_int']))[0]
            try:
                source_data['mad_param'] = madval
            except TypeError:
                print('V: ', Vs[r], type(Vs[r]))
                source_data['mad_param'] = np.nan
        else:
            drop_index = source_data.index[source_data['mjd']<drop_mjd][0]
            source_data = source_data.drop(drop_index)

            Vs = np.repeat(modulation_parameter(source_data[flux_col]),
                           len(source_data[flux_col]))
        
            try:
                etas = np.repeat(chi2_parameter(source_data[flux_col],
                                                source_data['{}_err'.format(flux_col)]),
                                 len(source_data[flux_col]))
            except ZeroDivisionError:
                print('source: ', r, 'etas error')
                etas = np.repeat(np.nan, 
                                 len(source_data[flux_col]))

            madps = np.repeat(madulation_parameter(source_data[flux_col])[0],
                              len(source_data[flux_col]))

            source_data['V_param'] = Vs
            source_data['eta_param'] = etas
            source_data['mad_param'] = madps
        
        source_lightcurves.append(source_data)

    # Close your connection to the database
    db._configured = False
    del db, session
    
    return source_lightcurves


def delete_duplicates(source_lightcurves, min_sep=3./60./60.):
    '''
    Sometimes TraP includes the same source twice.
    This function removes duplicate sources based
    on the right ascension and declination.
    
    TraP sometimes picks up the same source twice
    usually recording two different light curves
    for the same source where one light curve
    includes few epochs than the other. Here we
    check whether any sources are right on top
    of each other using the RA and Dec, and remove
    the source with fewer detections.
    
    Args:
    source_lightcurves (str): List of pandas tables with the
                              information for each source
    kwargs:
    min_sep (float): the minimum allowed separation
                     between two sources in degrees
                     Default: 3./60./60. degrees
    Returns:
    A list of pandas tables excluding duplicate
    sources
    
    '''
    # Get the coordinates of all of
    # the sources
    all_coords = np.zeros((len(source_lightcurves), 2))
    for s, source in enumerate(source_lightcurves):
        ra = np.nanmean(source['ra'])
        dec = np.nanmean(source['decl'])
        all_coords[s] = [ra, dec]
    all_coords = SkyCoord(all_coords, unit=(un.degree, un.degree))

    # Get the indices of sources that are
    # too close together
    delete_indices = []
    for s, source in enumerate(source_lightcurves):
        ra = np.nanmean(source['ra'])
        dec = np.nanmean(source['decl'])
        s_coord = SkyCoord(ra*un.degree, dec*un.degree)

        seps = s_coord.separation(all_coords)
        matches = np.where(seps.deg < min_sep)[0]

        if len(matches) > 1:
            flux_lengths = []
            for m, match in enumerate(matches):
                lc = source_lightcurves[m]
                len_flux = np.where(~np.isnan(lc['f_int']))
                flux_lengths.append(len(len_flux[0]))
            delete_indices += list(matches[np.where(matches!=(matches[np.nanargmax(flux_lengths)]))[0]])
    delete_indices = np.unique(np.array(delete_indices))

    # Remove the sources that are duplicates
    source_lightcurves_nodupes = []
    for s, source in enumerate(source_lightcurves_nocut):
        if s not in delete_indices:
            source_lightcurves_nodupes.append(source)

    print('Number of duplicates removed: {}'.format(len(delete_indices)))

    return source_lightcurves_nodupes


def save_sources(source_list,
                 flux_col,
                 database,
                 dataset_id,
                 savepath='/scratch/ldriessen/Source_Light_Curves/',
                 extended_sources='GX339_KnownExtendedSources.npy'):
    '''
    Save each source as a CSV. Include whether the source
    is an extended/resolved source
    
    Args:
    source_list (str): List of pandas tables with the
                       information for each source
    flux_col (str): The name of the column in the source
                    panda tables with the flux density in
                    it
    database (str): the name of the TraP database
    dataset_id (int): the number of the dataset in the database
    
    Kwargs:
    savepath (str): the name of the folder to save the
                    light curve files
    extended_sources (str): the name of the numpy file
                            with the coordinates of the
                            known extended sources
    
    Returns:
    A list of updated pandas tables for each source
    '''    
    ext_s = np.load(extended_sources)
    ext_s = SkyCoord(ext_s, unit=un.degree)
    
    scaled_sources = []
    for s, source in enumerate(source_list):
        # Get the frequency order for each pandas
        # table to make sure you match everything
        # up correctly by frequency

        rcat = np.array(source['runcat'])[0]

        # Check whether the source is
        # a resolved/extended source
        # and save accordingly
        ra = (source['ra']).mean()
        dec = (source['decl']).mean()
        coords = SkyCoord(ra, dec, unit=un.degree)
        seps = coords.separation(ext_s).deg
        if np.nanmin(seps)*60.*60. < 3.:
            savename = ('{0}rcat{1}_'
                        'ra{2:.2f}_'
                        'dec{3:.2f}_'
                        'db{4}_ds{5}_'
                        'ExtendedSource.csv').format(savepath,
                                                     rcat,
                                                     ra,
                                                     dec,
                                                     database,
                                                     dataset_id)
        else:
            savename = ('{0}rcat{1}_'
                        'ra{2:.2f}_'
                        'dec{3:.2f}_'
                        'db{4}_ds{5}_'
                        'PointSource.csv').format(savepath,
                                                  rcat,
                                                  ra,
                                                  dec,
                                                  database,
                                                  dataset_id)
        source.to_csv(savename)
        scaled_sources.append(source)

    print('Done')

    return scaled_sources


if __name__ in '__main__':
    # TraP database name
    database = 'Laura_NWLayers_ReProc'

    # The values needed to access TraP
    engine = 'postgresql'
    user =  'username' # your username here
    password = 'password'  # your password here
    host =  'vlo.science.uva.nl'
    port = 5432

    # Set up the path to where you want
    # to save the output files
    savepath = '/scratch/ldriessen/Source_Light_Curves/'
    # The file containing the ra and dec of
    # resolved sources and artefacts in
    # the field
    es_file = ('/scratch/ldriessen/'
               '2021.01.18_GX339_KnownExtendedSources.npy')
    # The path to where you want to save the
    # image postage stamps
    im_savepath = '/scratch/ldriessen/NWLayers_Bands/'

    # The MJD of the deep image
    # that you are going to remove
    # from each light curve
    august_2018 = 58331

    # Define your phase centre and and the minimum number
    # of data point required in a light curve.
    phase_centre_ra = 255.706
    phase_centre_dec = -48.790
    min_dpts = 1

    # The dataset IDs of the datasets you want
    # to process
    dataset_ids = [49, 50, 51, 52, 53, 54, 55]

    for d, dataset_id in enumerate(dataset_ids):    
        print('All datasets: ', dataset_ids)
        freq = freqs[d]

        print('***********************************')
        print('Dataset id: {}'.format(dataset_id))
        print('Frequency: {} MHz'.format(freq))

        # Get all the images that were run throuh
        # TraP and get some basic information
        images = glob.glob('/scratch/ldriessen/NWLayers/*{}*.fits'.format(freq))
        images = np.array(sorted(images))

        im_dates = []
        for i, im in enumerate(images):
            bn = im.split('/')[-1]
            d = bn.split('_')[0]
            im_dates.append(d)
        im_dates = np.array(im_dates)
        unique_images = np.unique(im_dates, return_index=True)

        print('Total images: {}'.format(len(images)))
        images = images[unique_images[1]]
        print('Unique images: {}'.format(len(images)))

        # Get the total number of epochs and their mjds,
        # and get the full correct frequency/ies from the
        # fits image headers
        image_values = np.ones((len(images), 2))
        for i, im in enumerate(images):
            with fits.open(im) as f:
                for hdu in f:
                    if hdu.header['CTYPE2'] == 'FREQ':
                        freq = (hdu.header['CRVAL2'])*1e-6
                    elif hdu.header['CTYPE3'] == 'FREQ':
                        freq = (hdu.header['CRVAL3'])*1e-6
                    elif hdu.header['CTYPE4'] == 'FREQ':
                        freq = (hdu.header['CRVAL4'])*1e-6
                    obs_time = astropy.time.Time(hdu.header['DATE-OBS'])
                    obs_isot = obs_time.isot
                    obs_mjd = obs_time.mjd

                    image_values[i] = [freq, obs_mjd]
        all_frequencies = np.sort(np.unique(image_values[:, 0]))
        num_epochs = len(image_values)/len(np.unique(all_frequencies.astype(int)))
        print('Frequencies observed: ', all_frequencies)
        print('Number of images: {}'.format(len(images)))
        print('Number of epochs: {}'.format(num_epochs))

        # Get all of the light curves for this database and
        # dataset. If you've included the deep image first,
        # make sure to drop that mjd, this will remove that
        # epoch completely
        source_lightcurves_nocut = get_lightcurves(phase_centre_ra,
                                                   phase_centre_dec,
                                                   min_dpts,
                                                   engine,
                                                   host,
                                                   port,
                                                   user,
                                                   password,
                                                   database,
                                                   dataset_id,
                                                   image_values,
                                                   drop_mjd=august_2018)

        print('Total number of sources: {}'.format(len(source_lightcurves_nocut)))

        # Sometimes TraP grabs the same source twice. Get rid of any
        # duplicates, removing the liht curve with few detections
        new_source_list = delete_duplicates(source_lightcurves_nocut)

        # Save each source as a csv
        saved_sources = save_sources(new_source_list,
                                     'f_int',
                                     database,
                                     dataset_id,
                                     savepath=savepath,
                                     extended_sources=es_file)
