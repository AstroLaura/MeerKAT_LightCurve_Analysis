# L. N. Driessen
# Last updatedL 2021.01.22

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
import pandas as pd
import time
import os
from datetime import date
import subprocess as sub

import scipy as sp
import scipy.stats as spstats
from scipy.optimize import curve_fit

from astropy import units as un
from astropy.coordinates import SkyCoord
import astropy.time
from astropy.io import fits
from astropy import nddata
from astropy.wcs import WCS
from astropy.timeseries import LombScargle

import matplotlib
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits import mplot3d


def annulus_outer_r(inner_r, area):
    '''
    Get the outer radius of an annulus of specified area
    
    Takes the inner radius of the annulus and
    the required area of the annulus and
    calculates the outer radius corresponding
    to those values
    
    Args:
    inner_r (float): the inner radius of the
                     annulus
    area (float): the area of the annulus
    
    Returns:
    The outer radius of the annulus (float)
    '''
    if inner_r == 0:
        outer_r = np.sqrt(area/np.pi)
    else:
        outer_r = np.sqrt((area/np.pi) + inner_r**2.)
        
    return outer_r


# The path to where files/plots will be saved
file_path = ('/raid/driessen/FlareStars/'
             'GX339/Correlation_Investigation/')

# The TraP database name
db = 'Laura_NWLayers_ReProc'

# The path to the source light curves you want
# to use
lightcurve_path = ('/raid/driessen/FlareStars/'
                   'GX339/Source_Light_Curves/'
                   'Average_Scaled/')

# The dataset ID values (from TraP) and
# the corresponding central frequencies (MHz)
DSs = [49, 52, 53, 50, 54, 55, 51, 33]
freqs = [1658, 1551, 1444, 1337, 1123, 1016, 909, 'MFS']

# The names of the columns you want
# to use in the pandas tables
mjd_col = 'mjd'
flux_col = 'f_int_median_scaled'
flux_err_col = 'f_int_median_scaled_err'
freq_col = 'freq_eff_int'
pc_col = 'dist_to_pc_DEG'

# If you're using the light curve files with
# all extended sources removed, this should
# be 'PS', otherwise it should be 'ES'
ext = 'PS'

#################################################

# Set up the coordinates of the variable sources
fb = SkyCoord(np.array([[256.23450507,
                         -48.35008655]]),
              unit=(un.degree, un.degree))
gx = SkyCoord(np.array([[255.70567674,
                         -48.78963475]]),
              unit=(un.degree, un.degree))
psr = SkyCoord(np.array([[255.97732915,
                          -48.86685808]]),
               unit=(un.degree, un.degree))

fb_rcats = {}
gx_rcats = {}
psr_rcats = {}

# Search all of the light curves to identify the running catalogue
# of each source in each data set
for d, ds in enumerate(DSs):
    all_lcs = all_lcs = glob.glob(('{0}rcat*'
                                       'db{1}_ds{2}_'
                                       '{3}.csv').format(lightcurve_path,
                                                         db,
                                                         ds, ext))
    
    for lcf in all_lcs:
        lc = pd.read_csv(lcf)
        ra = np.nanmean(lc['ra'])
        dec = np.nanmean(lc['decl'])
        lc_coord = SkyCoord(np.array([[ra,
                                       dec]]),
                            unit=(un.degree,
                                  un.degree))
        
        fb_sep = lc_coord.separation(fb)
        gx_sep = lc_coord.separation(gx)
        psr_sep = lc_coord.separation(psr)
        
        if fb_sep.deg < 2./60./60.:
            fb_rcats[ds] = np.array(lc['runcat'])[0]
        elif gx_sep.deg < 2./60./60.:
            gx_rcats[ds] = np.array(lc['runcat'])[0]
        elif psr_sep.deg < 2./60./60.:
            psr_rcats[ds] = np.array(lc['runcat'])[0]

print(fb_rcats)
print(gx_rcats)
print(psr_rcats)


# Set up some useful plotting variables
fb_col = '#E0777D'
gx_col = '#46B1C9'
psr_col = '#645DD7'

fb_mark = 'v'
gx_mark = 'd'
psr_mark = '^'

pointsize=5
fb_frame = mlines.Line2D([], [], color='None', marker=fb_mark,
                       markerfacecolor='None',
                       markeredgecolor=fb_col,
                       markersize=pointsize+5,
                       label='MKT J170456.2-482100')
gx_frame = mlines.Line2D([], [], color='None', marker=gx_mark,
                       markerfacecolor='None',
                       markeredgecolor=gx_col,
                       markersize=pointsize+5,
                       label='GX 339-4')
psr_frame = mlines.Line2D([], [], color='None', marker=psr_mark,
                       markerfacecolor='None',
                       markeredgecolor=psr_col,
                       markersize=pointsize+5,
                       label='PSR J1703-4851')

#################################################

fig, ax = plt.subplots(8, 2, figsize=(16, 24), sharex=True, sharey='row')
cols = ['Black', 'Grey']
fmts = ['o', '.']
for d, ds in enumerate(DSs):
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    for e, ext in enumerate(['ES', 'PS']):
        all_lcs = all_lcs = glob.glob(('{0}rcat*'
                                       'db{1}_ds{2}_'
                                       '{3}.csv').format(lightcurve_path,
                                                         db,
                                                         ds,
                                                         ext))
        lc = pd.read_csv(all_lcs[0])

        median_model = lc['ref scale model median']
        median_model_err = lc['ref scale model mad']

        mean_model = lc['ref scale model mean']
        mean_model_err = lc['ref scale model std']

        mjd = lc[mjd_col]

        ax[d, 0].errorbar(mjd, median_model,
                          yerr=median_model_err,
                          fmt=fmts[e], c=cols[e])
        ax[d, 1].errorbar(mjd, mean_model,
                          yerr=mean_model_err,
                          fmt=fmts[e], c=cols[e])

    ax_text = AnchoredText(freq_str,
                           loc='upper left',
                           prop=dict(size=18, color='Grey'),
                           borderpad=0.1,
                           frameon=False)
    ax[d, 0].add_artist(ax_text)

    xmin, xmax = ax[d, 0].get_xlim()
    ax[d, 0].plot(np.linspace(xmin, xmax, 10), np.ones(10),
                  '--', c='DarkGrey', lw=3, alpha=0.5)
    ax[d, 0].set_xlim(xmin, xmax)
    xmin, xmax = ax[d, 1].get_xlim()
    ax[d, 1].plot(np.linspace(xmin, xmax, 10), np.ones(10),
                  '--', c='DarkGrey', lw=3, alpha=0.5)
    ax[d, 1].set_xlim(xmin, xmax)
    
    ax[d, 0].set_ylabel('Fractional offset', fontsize=14)

    ax[d, 1].tick_params(axis='y', direction='in')
    ax[d, 1].tick_params(labelleft=False)
    if d < 7:
        for col in range(2):
            ax[d, col].tick_params(axis='x', direction='in')
            ax[d, col].tick_params(labelbottom=False)
    
ax[0, 0].set_title('Median systematic model', fontsize=18)
ax[0, 1].set_title('Mean systematic model', fontsize=18)

ax[7, 0].set_xlabel('MJD', fontsize=14)
ax[7, 1].set_xlabel('MJD', fontsize=14)

pointsize = 5
es_dot = mlines.Line2D([], [], color='None', marker='o',
                       markerfacecolor='Black',
                       markersize=pointsize+5,
                       label='Including resolved/artefact sources')
ps_dot = mlines.Line2D([], [], color='None', marker='o',
                       markerfacecolor='DarkGrey',
                       markeredgecolor='DarkGrey',
                       markersize=pointsize+5,
                       label='Point sources only')
one_line = mlines.Line2D([], [], color='DarkGrey', lw=3, alpha=0.7,
                         marker='None', linestyle='--',
                         label='No change to measured fluxes')

leg0 = ax[7, 0].legend(handles=[es_dot, ps_dot, one_line],
                       fontsize=15, frameon=True,
                       loc='lower left', ncol=3,
                       bbox_to_anchor=(0.0, -0.7),
                       borderaxespad=0,
                       edgecolor='Black')

fig.subplots_adjust(hspace=0, wspace=0)

figure_name = ('{0}PSvsES_Models.png').format(file_path)
fig.savefig(figure_name, bbox_inches='tight')

fig.close(fig)

#################################################

for ext in ['ES', 'PS']:
    for flux_col in ['f_int_median',
                     'f_int_median_scaled_median']:
        if flux_col == 'f_int_median':
            eta_col = 'eta_param'
            V_col = 'V_param'
            mad_col = 'mad_param'
        elif flux_col == 'f_int_median_scaled_median':
            eta_col = 'median_scaled_eta'
            V_col = 'median_scaled_V'
            mad_col = 'median_scaled_madp'

        fig, ax = plt.subplots(8, 3, figsize=(16, 24), sharex=True, sharey='col')
        for d, ds in enumerate(DSs):
            freq = freqs[d]
            if freq == 'MFS':
                freq_str = 'MFS'
            else:
                freq_str = '{}MHz'.format(freq)
            var_params = glob.glob(('{0}VariabilityParams_'
                                    'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                                 db,
                                                                 ds, ext)))[0]
            vp_all = pd.read_csv(var_params)
            qry = '`detected SN2` == \'No\''
            vp_no = vp_all.query(qry)

            qry = '`detected SN2` == \'Yes\' and `detected SN3` == \'No\''
            vp_2 = vp_all.query(qry)

            qry = '`detected SN3` == \'Yes\''
            vp_3 = vp_all.query(qry)

            dataframes = [vp_no, vp_2, vp_3]
            vp_cols = ['#CED2DE', '#5A6587', '#08090C']

            for df, dataframe in enumerate(dataframes):
                flux = np.log10(dataframe[flux_col])
                ax[d, 0].scatter(flux, np.log10(dataframe[eta_col]),
                                 marker='.', c=vp_cols[df], s=35)
                ax[d, 1].scatter(flux, np.log10(dataframe[V_col]),
                                 marker='.', c=vp_cols[df], s=35)
                ax[d, 2].scatter(flux, np.log10(dataframe[mad_col]),
                                 marker='.', c=vp_cols[df], s=35)

            fb_info = vp_all[vp_all['runcat'] == fb_rcats[ds]]
            gx_info = vp_all[vp_all['runcat'] == gx_rcats[ds]]
            psr_info = vp_all[vp_all['runcat'] == psr_rcats[ds]]
            for df, dataframe in enumerate([fb_info,
                                            gx_info,
                                            psr_info]):
                cola = [fb_col, gx_col, psr_col][df]
                marka = [fb_mark, gx_mark, psr_mark][df]

                ax[d, 0].scatter(np.log10(dataframe[flux_col]),
                                 np.log10(dataframe[eta_col]),
                                 marker=marka, c='none',
                                 edgecolor=cola)

                ax[d, 1].scatter(np.log10(dataframe[flux_col]),
                                     np.log10(dataframe[V_col]),
                                     marker=marka, c='none',
                                 edgecolor=cola)

                ax[d, 2].scatter(np.log10(dataframe[flux_col]),
                                 np.log10(dataframe[mad_col]),
                                 marker=marka, c='none',
                                 edgecolor=cola)

            ax[d, 0].set_ylabel(r'log$_{10}\eta$', fontsize=14)
            ax[d, 1].set_ylabel(r'log$_{10}$V', fontsize=14)
            ax[d, 2].set_ylabel(r'log$_{10}\xi$', fontsize=14)

            ax_text = AnchoredText(freq_str,
                                   loc='upper left',
                                   prop=dict(size=18, color='Grey'),
                                   borderpad=0.1,
                                   frameon=False)
            ax[d, 0].add_artist(ax_text)

            if d == 7:
                for col in range(3):
                    ax[d, col].set_xlabel(r'log$_{10}$Flux [Jy]',
                                          fontsize=14)
            else:
                for col in range(3):
                    ax[d, col].tick_params(axis='x', direction='in')
                    ax[d, col].tick_params(labelbottom=False)


        pointsize = 5
        undetected = mlines.Line2D([], [], color='None', marker='.',
                                   markerfacecolor=vp_cols[0],
                                   markeredgecolor='None',
                                   markersize=pointsize+5,
                                   label='Light curves with S/N < 2')
        less3 = mlines.Line2D([], [], color='None', marker='.',
                              markerfacecolor=vp_cols[1],
                              markeredgecolor='None',
                              markersize=pointsize+5,
                              label='Light curves with 2 < S/N < 3')
        great3 = mlines.Line2D([], [], color='None', marker='.',
                               markerfacecolor=vp_cols[2],
                               markeredgecolor='None',
                               markersize=pointsize+5,
                               label='Light curves with S/N > 3')

        leg0 = ax[7, 0].legend(handles=[undetected, less3, great3,
                                        fb_frame, gx_frame, psr_frame],
                               fontsize=15, frameon=True, loc='lower left', ncol=4,
                               bbox_to_anchor= (0.0, -0.7), borderaxespad=0,
                               edgecolor='Black')

        fig.subplots_adjust(hspace=0, wspace=0.3)

        figure_name = ('{0}VarParams_{1}_SNall_{2}.png').format(file_path,
                                                                ext,
                                                                flux_col)
        fig.savefig(figure_name, bbox_inches='tight')
        
        plt.close(fig)

#################################################

signoises = [2, 3]
ext = 'PS'
flux_col = 'f_int_median'
eta_col = 'eta_param'
V_col = 'V_param'
mad_col = 'mad_param'

for signoise in signoises:
    fig, ax = plt.subplots(8, 3, figsize=(16, 24), sharex=True, sharey='col')
    for d, ds in enumerate(DSs):
        freq = freqs[d]
        if freq == 'MFS':
            freq_str = 'MFS'
        else:
            freq_str = '{}MHz'.format(freq)
        var_params = glob.glob(('{0}VariabilityParams_'
                                'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                             db,
                                                             ds, ext)))[0]
        vp_all = pd.read_csv(var_params)
        vp = vp_all[vp_all['detected SN{}'.format(signoise)] == 'Yes']
        vp_no = vp_all[vp_all['detected SN{}'.format(signoise)] == 'No']
        fb_info = vp[vp['runcat'] == fb_rcats[ds]]
        gx_info = vp[vp['runcat'] == gx_rcats[ds]]
        psr_info = vp[vp['runcat'] == psr_rcats[ds]]

        flux = np.log10(vp_no[flux_col])
        ax[d, 0].scatter(flux, np.log10(vp_no[eta_col]),
                         marker='.', c='DarkGrey', s=35)
        ax[d, 1].scatter(flux, np.log10(vp_no[V_col]),
                         marker='.', c='DarkGrey', s=35)
        ax[d, 2].scatter(flux, np.log10(vp_no[mad_col]),
                         marker='.', c='DarkGrey', s=35)

        flux = np.log10(vp[flux_col])
        ax[d, 0].scatter(flux, np.log10(vp[eta_col]),
                         marker='.', c='Black', s=35)
        ax[d, 1].scatter(flux, np.log10(vp[V_col]),
                         marker='.', c='Black', s=35)
        ax[d, 2].scatter(flux, np.log10(vp[mad_col]),
                         marker='.', c='Black', s=35)

        for df, dataframe in enumerate([fb_info,
                                        gx_info,
                                        psr_info]):
            cola = [fb_col, gx_col, psr_col][df]
            marka = [fb_mark, gx_mark, psr_mark][df]

            ax[d, 0].scatter(np.log10(dataframe[flux_col]),
                             np.log10(dataframe[eta_col]),
                             marker=marka, c='none',
                             edgecolor=cola)

            ax[d, 1].scatter(np.log10(dataframe[flux_col]),
                                 np.log10(dataframe[V_col]),
                                 marker=marka, c='none',
                             edgecolor=cola)

            ax[d, 2].scatter(np.log10(dataframe[flux_col]),
                             np.log10(dataframe[mad_col]),
                             marker=marka, c='none',
                             edgecolor=cola)

        ax[d, 0].set_ylabel(r'log$_{10}\eta$', fontsize=14)
        ax[d, 1].set_ylabel(r'log$_{10}$V', fontsize=14)
        ax[d, 2].set_ylabel(r'log$_{10}\xi$', fontsize=14)

        ax_text = AnchoredText(freq_str,
                               loc='upper left',
                               prop=dict(size=18, color='Grey'),
                               borderpad=0.1,
                               frameon=False)
        ax[d, 0].add_artist(ax_text)

        if d == 7:
            for col in range(3):
                ax[d, col].set_xlabel(r'log$_{10}$Flux [Jy]',
                                      fontsize=14)
        else:
            for col in range(3):
                ax[d, col].tick_params(axis='x', direction='in')
                ax[d, col].tick_params(labelbottom=False)


    pointsize = 5
    corr_dot = mlines.Line2D([], [], color='None', marker='.',
                           markerfacecolor='Black',
                           markersize=pointsize+5,
                           label='Light curves with S/N > {}'.format(signoise))
    uncorr_dot = mlines.Line2D([], [], color='None', marker='.',
                           markerfacecolor='DarkGrey',
                           markeredgecolor='DarkGrey',
                           markersize=pointsize+5,
                           label='Light curves with S/N < {}'.format(signoise))

    leg0 = ax[7, 0].legend(handles=[corr_dot, uncorr_dot,
                                    fb_frame, gx_frame, psr_frame],
                           fontsize=15, frameon=True, loc='lower left', ncol=4,
                           bbox_to_anchor= (0.0, -0.7), borderaxespad=0,
                           edgecolor='Black')

    fig.subplots_adjust(hspace=0, wspace=0.3)

    figure_name = ('{0}VarParams_{1}_SN{2}_{3}.png').format(file_path, ext,
                                                            signoise, flux_col)
    fig.savefig(figure_name, bbox_inches='tight')
    
    plt.close(fig)

#################################################

flux_col = 'f_int_median'
eta_col = 'eta_param'
V_col = 'V_param'
mad_col = 'mad_param'
exts = ['ES', 'PS']
ext_cols = ['DarkGrey', 'Black']

fig, ax = plt.subplots(8, 3, figsize=(16, 24), sharex=True, sharey='col')
for d, ds in enumerate(DSs):
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)

    for e, ext in enumerate(exts):
        ecol = ext_cols[e]
        var_params = glob.glob(('{0}VariabilityParams_'
                                'db{1}_'
                                'ds{2}_'
                                '{3}.csv'.format(lightcurve_path,
                                                 db,
                                                 ds, ext)))[0]
        vp_all = pd.read_csv(var_params)
        vp = vp_all[vp_all['detected SN2'] == 'Yes']

        flux = np.log10(vp[flux_col])
        ax[d, 0].scatter(flux, np.log10(vp[eta_col]),
                         marker='.', c=ecol, s=35)
        ax[d, 1].scatter(flux, np.log10(vp[V_col]),
                         marker='.', c=ecol, s=35)
        ax[d, 2].scatter(flux, np.log10(vp[mad_col]),
                         marker='.', c=ecol, s=35)

    
    fb_info = vp_all[vp_all['runcat'] == fb_rcats[ds]]
    gx_info = vp_all[vp_all['runcat'] == gx_rcats[ds]]
    psr_info = vp_all[vp_all['runcat'] == psr_rcats[ds]]
    for df, dataframe in enumerate([fb_info,
                                    gx_info,
                                    psr_info]):
        cola = [fb_col, gx_col, psr_col][df]
        marka = [fb_mark, gx_mark, psr_mark][df]

        ax[d, 0].scatter(np.log10(dataframe[flux_col]),
                         np.log10(dataframe[eta_col]),
                         marker=marka, c='none',
                         edgecolor=cola)

        ax[d, 1].scatter(np.log10(dataframe[flux_col]),
                         np.log10(dataframe[V_col]),
                         marker=marka, c='none',
                         edgecolor=cola)

        ax[d, 2].scatter(np.log10(dataframe[flux_col]),
                         np.log10(dataframe[mad_col]),
                         marker=marka, c='none',
                         edgecolor=cola)

    ax[d, 0].set_ylabel(r'log$_{10}\eta$', fontsize=14)
    ax[d, 1].set_ylabel(r'log$_{10}$V', fontsize=14)
    ax[d, 2].set_ylabel(r'log$_{10}\xi$', fontsize=14)

    ax_text = AnchoredText(freq_str,
                           loc='upper left',
                           prop=dict(size=18, color='Grey'),
                           borderpad=0.1,
                           frameon=False)
    ax[d, 0].add_artist(ax_text)
    
    if d == 7:
        for col in range(3):
            ax[d, col].set_xlabel(r'log$_{10}$Flux [Jy]',
                                  fontsize=14)
    else:
        for col in range(3):
            ax[d, col].tick_params(axis='x', direction='in')
            ax[d, col].tick_params(labelbottom=False)

    print(freq_str)
    print('----------')
    
    
pointsize = 5
ps_dot = mlines.Line2D([], [], color='None', marker='.',
                       markerfacecolor='Black',
                       markersize=pointsize+5,
                       label='Point sources only')
es_dot = mlines.Line2D([], [], color='None', marker='.',
                       markerfacecolor='DarkGrey',
                       markeredgecolor='DarkGrey',
                       markersize=pointsize+5,
                       label='All sources')

leg0 = ax[7, 0].legend(handles=[es_dot,
                                ps_dot,
                                fb_frame,
                                gx_frame,
                                psr_frame],
                       fontsize=15, frameon=True,
                       loc='lower left', ncol=4,
                       bbox_to_anchor= (0.0, -0.7),
                       borderaxespad=0,
                       edgecolor='Black')

fig.subplots_adjust(hspace=0, wspace=0.3)

figure_name = ('{0}VarParams_'
               'ESvsPS_SN2.png').format(file_path,
                                        ext)
fig.savefig(figure_name, bbox_inches='tight')

plt.close(fig)

#################################################

ext = 'PS'
fig, ax = plt.subplots(8, 3, figsize=(16, 24), sharex=True, sharey='col')
for d, ds in enumerate(DSs):
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    var_params = glob.glob(('{0}VariabilityParams_'
                            'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                         db,
                                                         ds, ext)))[0]
    vp_all = pd.read_csv(var_params)
    vp = vp_all[vp_all['detected SN2'] == 'Yes']
    fb_info = vp[vp['runcat'] == fb_rcats[ds]]
    gx_info = vp[vp['runcat'] == gx_rcats[ds]]
    psr_info = vp[vp['runcat'] == psr_rcats[ds]]
    
    flux = np.log10(vp['f_int_median'])
    ax[d, 0].scatter(flux, np.log10(vp['eta_param']),
                     marker='.', c='Grey')
    ax[d, 1].scatter(flux, np.log10(vp['V_param']),
                     marker='.', c='Grey')
    ax[d, 2].scatter(flux, np.log10(vp['mad_param']),
                     marker='.', c='Grey')

    flux = np.log10(vp['f_int_median_scaled_median'])
    ax[d, 0].scatter(flux, np.log10(vp['median_scaled_eta']),
                     marker='.', c='Black', s=35)
    ax[d, 1].scatter(flux, np.log10(vp['median_scaled_V']),
                     marker='.', c='Black', s=35)
    ax[d, 2].scatter(flux, np.log10(vp['median_scaled_madp']),
                     marker='.', c='Black', s=35)

    for f, flux_col in enumerate(['f_int_median',
                                  'f_int_median_scaled_median']):
        eta_col = ['eta_param', 'median_scaled_eta'][f]
        V_col = ['V_param', 'median_scaled_V'][f]
        mad_col = ['mad_param', 'median_scaled_madp'][f]
        for df, dataframe in enumerate([fb_info,
                                        gx_info,
                                        psr_info]):
            cola = [fb_col, gx_col, psr_col][df]
            marka = [fb_mark, gx_mark, psr_mark][df]

            ax[d, 0].scatter(np.log10(dataframe[flux_col]),
                             np.log10(dataframe[eta_col]),
                             marker=marka, c='none',
                             edgecolor=cola)

            ax[d, 1].scatter(np.log10(dataframe[flux_col]),
                                 np.log10(dataframe[V_col]),
                                 marker=marka, c='none',
                             edgecolor=cola)

            ax[d, 2].scatter(np.log10(dataframe[flux_col]),
                             np.log10(dataframe[mad_col]),
                             marker=marka, c='none',
                             edgecolor=cola)

    ax[d, 0].set_ylabel(r'log$_{10}\eta$', fontsize=14)
    ax[d, 1].set_ylabel(r'log$_{10}$V', fontsize=14)
    ax[d, 2].set_ylabel(r'log$_{10}\xi$', fontsize=14)

    ax_text = AnchoredText(freq_str,
                           loc='upper left',
                           prop=dict(size=18, color='Grey'),
                           borderpad=0.1,
                           frameon=False)
    ax[d, 0].add_artist(ax_text)
    
    if d == 7:
        for col in range(3):
            ax[d, col].set_xlabel(r'log$_{10}$Flux [Jy]',
                                  fontsize=14)
    else:
        for col in range(3):
            ax[d, col].tick_params(axis='x', direction='in')
            ax[d, col].tick_params(labelbottom=False)
    
    
pointsize = 5
corr_dot = mlines.Line2D([], [], color='None', marker='.',
                       markerfacecolor='Black',
                       markersize=pointsize+5,
                       label='Corrected light curves')
uncorr_dot = mlines.Line2D([], [], color='None', marker='.',
                       markerfacecolor='DarkGrey',
                       markeredgecolor='DarkGrey',
                       markersize=pointsize+5,
                       label='Uncorrected light curves')

leg0 = ax[7, 0].legend(handles=[uncorr_dot, corr_dot,
                                fb_frame, gx_frame, psr_frame],
                       fontsize=15, frameon=True, loc='lower left', ncol=3,
                       bbox_to_anchor= (0.0, -0.7), borderaxespad=0,
                       edgecolor='Black')

fig.subplots_adjust(hspace=0, wspace=0.3)

figure_name = ('{0}UncorrvsCorr_VarParams.png').format(file_path)
fig.savefig(figure_name, bbox_inches='tight')

plt.close(fig)

#################################################

for signoise in [2, 3]:
    for ext in ['ES', 'PS']:

        for flux_col in ['f_int_median', 'f_int_median_scaled_median']:
            if flux_col == 'f_int_median':
                eta_col = 'eta_param'
                V_col = 'V_param'
                mad_col = 'mad_param'
            elif flux_col == 'f_int_median_scaled_median':
                eta_col = 'median_scaled_eta'
                V_col = 'median_scaled_V'
                mad_col = 'median_scaled_madp'


            fig, ax = plt.subplots(4, 2, figsize=(10, 14), sharey=True)
            for d, ds in enumerate(DSs):
                row = d // 2
                col = d % 2

                freq = freqs[d]
                if freq == 'MFS':
                    freq_str = 'MFS'
                else:
                    freq_str = '{}MHz'.format(freq)
                var_params = glob.glob(('{0}VariabilityParams_'
                                        'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                                     db,
                                                                     ds, ext)))[0]
                vp = pd.read_csv(var_params)
                vp = vp[vp['detected SN{}'.format(signoise)]=='Yes']

                flux = np.log10(vp[flux_col])
                ax[row, col].scatter(flux, np.log10(vp[eta_col]),
                                     marker='.', c='Black')

                fb_info = vp[vp['runcat'] == fb_rcats[ds]]
                gx_info = vp[vp['runcat'] == gx_rcats[ds]]
                psr_info = vp[vp['runcat'] == psr_rcats[ds]]
                for df, dataframe in enumerate([fb_info,
                                                gx_info,
                                                psr_info]):
                    cola = [fb_col, gx_col, psr_col][df]
                    marka = [fb_mark, gx_mark, psr_mark][df]

                    ax[row, col].scatter(np.log10(dataframe[flux_col]),
                                     np.log10(dataframe[eta_col]),
                                     marker=marka, c='none',
                                     edgecolor=cola)

                ax[row, col].set_xlim(-7.7, 0.2)

                if col == 0:
                    ax[row, col].set_ylabel(r'log$_{10}\eta$', fontsize=14)
                else:
                    ax[row, col].tick_params(axis='y', direction='in')

                if row == 3:
                    for bob in range(2):
                        ax[row, bob].set_xlabel(r'log$_{10}$Flux [Jy]',
                                              fontsize=14)
                else:
                    ax[row, col].tick_params(axis='x', direction='in')

                ax_text = AnchoredText(freq_str,
                                       loc='upper left',
                                       prop=dict(size=18, color='Grey'),
                                       borderpad=0.1,
                                       frameon=False)
                ax[row, col].add_artist(ax_text)

            pointsize = 5
            fb_dot = mlines.Line2D([], [], color='None', marker=fb_mark,
                                   markerfacecolor='None',
                                   markeredgecolor=fb_col,
                                   markersize=pointsize+5,
                                   label='MKT J170456.2-482100')
            gx_dot = mlines.Line2D([], [], color='None', marker=gx_mark,
                                   markerfacecolor='None',
                                   markeredgecolor=gx_col,
                                   markersize=pointsize+5,
                                   label='GX 339-4')
            psr_dot = mlines.Line2D([], [], color='None', marker=psr_mark,
                                   markerfacecolor='None',
                                   markeredgecolor=psr_col,
                                   markersize=pointsize+5,
                                   label='PSR J1703-4851')

            leg0 = ax[3, 0].legend(handles=[fb_dot, gx_dot, psr_dot],
                                   fontsize=15, frameon=True, loc='lower left', ncol=2,
                                   bbox_to_anchor= (0.0, -0.6), borderaxespad=0,
                                   edgecolor='Black')

            fig.subplots_adjust(hspace=0, wspace=0)

            figure_name = ('{0}Uncorr_EtaParams_{1}_{2}_SN{3}.png').format(file_path,
                                                                           ext,
                                                                           flux_col,
                                                                           signoise)
            fig.savefig(figure_name, bbox_inches='tight')

            plt.close(fig)

#################################################

for signoise in [2, 3]:
    for ext in ['ES', 'PS']:

        for flux_col in ['f_int_median',
                         'f_int_median_scaled_median']:
            if flux_col == 'f_int_median':
                eta_col = 'eta_param'
                V_col = 'V_param'
                mad_col = 'mad_param'
            elif flux_col == 'f_int_median_scaled_median':
                eta_col = 'median_scaled_eta'
                V_col = 'median_scaled_V'
                mad_col = 'median_scaled_madp'


            fig, ax = plt.subplots(4, 2, figsize=(10, 14), sharey=True)
            for d, ds in enumerate(DSs):
                row = d // 2
                col = d % 2

                freq = freqs[d]
                if freq == 'MFS':
                    freq_str = 'MFS'
                else:
                    freq_str = '{}MHz'.format(freq)
                var_params = glob.glob(('{0}VariabilityParams_'
                                        'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                                     db,
                                                                     ds, ext)))[0]
                vp = pd.read_csv(var_params)
                vp = vp[vp['detected SN{}'.format(signoise)]=='Yes']

                ax[row, col].scatter(np.log10(vp[eta_col]),
                                     np.log10(vp[V_col]),
                                     marker='.', c='Black')

                fb_info = vp[vp['runcat'] == fb_rcats[ds]]
                gx_info = vp[vp['runcat'] == gx_rcats[ds]]
                psr_info = vp[vp['runcat'] == psr_rcats[ds]]
                for df, dataframe in enumerate([fb_info,
                                                gx_info,
                                                psr_info]):
                    cola = [fb_col, gx_col, psr_col][df]
                    marka = [fb_mark, gx_mark, psr_mark][df]

                    ax[row, col].scatter(np.log10(dataframe[eta_col]),
                                         np.log10(dataframe[V_col]),
                                         marker=marka, c='none',
                                         edgecolor=cola)

                if col == 0:
                    ax[row, col].set_ylabel(r'log$_{10}$V', fontsize=14)
                else:
                    ax[row, col].tick_params(axis='y', direction='in')

                if row == 3:
                    for bob in range(2):
                        ax[row, bob].set_xlabel(r'log$_{10}\eta$',
                                              fontsize=14)
                else:
                    ax[row, col].tick_params(axis='x', direction='in')

                ax_text = AnchoredText(freq_str,
                                       loc='upper right',
                                       prop=dict(size=18, color='Grey'),
                                       borderpad=0.1,
                                       frameon=False)
                ax[row, col].add_artist(ax_text)

            leg0 = ax[3, 0].legend(handles=[fb_frame, gx_frame, psr_frame],
                                   fontsize=15, frameon=True, loc='lower left', ncol=2,
                                   bbox_to_anchor= (0.0, -0.6), borderaxespad=0,
                                   edgecolor='Black')

            fig.subplots_adjust(hspace=0, wspace=0)

            figure_name = ('{0}Uncorr_EtaVsV_{1}_{2}_SN{3}.png').format(file_path,
                                                                      ext,
                                                                      flux_col, signoise)
            fig.savefig(figure_name, bbox_inches='tight')

            plt.close(fig)

#################################################

# The minimum signal to noise a source
# needs to have (in at least one epoch)
# to be included in the analysis
minimum_sn = 2.

f = 0
fc = ['f_int', 'f_int_median_scaled'][f]
fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]

fig, ax = plt.subplots(8, 2, figsize=(15, 25), sharey=True)

for d, ds in enumerate(DSs):
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    
    print('Working on {0} MHz, ds {1}'.format(freq, ds))
    print('Working on {0} {1}'.format(fc, fc_err))
    
    for e, ext in enumerate(['ES', 'PS']):
        col = e
        row = d
        
        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)
        
        if ext == 'PS':
            label_string = 'Resolved sources removed'
        elif ext == 'ES':
            label_string = 'All detected sources'
        ax0_text = AnchoredText('{0}, '
                                '{1}'.format(label_string, freq_str),
                                loc='lower right',
                                prop=dict(size=14, color='Grey'),
                                borderpad=0.1,
                                frameon=False)
        
        all_corrs = pd.read_csv(correlation_filename)

        ax[row, col].scatter(all_corrs['s1_{}'.format(pc_col)],
                             all_corrs['correlation coefficient'],
                             marker='.', s=10, c='Black')
        ax[row, col].set_xlim(0, 1.27)
        
        ax[row, col].add_artist(ax0_text)
        
        
        
        if col == 0:
            ax[row, col].set_xticklabels([0, 0.25, 0.5, 0.75, 1.0])
            ax[row, col].set_ylabel('Correlation\ncoefficient', fontsize=18)
        else:
            ax[row, col].tick_params(axis='y', direction='in')
            ax[row, col].tick_params(labelleft=False)
        if row == len(DSs) - 1:
            ax[row, col].set_xlabel('Distance to phase centre (deg)', fontsize=18)
        else:
            ax[row, col].tick_params(axis='x', direction='in')
            ax[row, col].tick_params(labelbottom=False)

fig.subplots_adjust(hspace=0, wspace=0)

figure_name = ('{0}AllCorrelations_'
               'ESvPS_'
               '{1}_'
               'db{2}_'
               'ds{3}_'
               'SN{4}.png').format(file_path,
                                 fc, 
                                 db, ds,
                                 minimum_sn)
fig.savefig(figure_name,
            bbox_inches='tight')

plt.close(fig)

#################################################

fig, ax = plt.subplots(8, 2, figsize=(15, 25), sharey=True)
ext = 'PS'

for d, ds in enumerate(DSs):
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    
    print('Working on {0} MHz, ds {1}'.format(freq, ds))
    print('Working on {0} {1}'.format(fc, fc_err))
    
    for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]

        col = f
        row = d
        
        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)
        
        if fc == 'f_int':
            label_string = 'Uncorrected light curves'
        elif fc == 'f_int_median_scaled':
            label_string = 'Corrected light curves'
        ax0_text = AnchoredText('{0}, '
                                '{1}'.format(label_string, freq_str),
                                loc='lower right',
                                prop=dict(size=14, color='Grey'),
                                borderpad=0.1,
                                frameon=False)
        
        all_corrs = pd.read_csv(correlation_filename)

        ax[row, col].scatter(all_corrs['s1_{}'.format(pc_col)],
                             all_corrs['correlation coefficient'],
                             marker='.', s=10, c='Black')
        ax[row, col].set_xlim(0, 1.27)
        
        ax[row, col].add_artist(ax0_text)
        
        if col == 0:
            ax[row, col].set_xticklabels([0, 0.25, 0.5, 0.75, 1.0])
            ax[row, col].set_ylabel('Correlation\ncoefficient', fontsize=18)
        else:
            ax[row, col].tick_params(axis='y', direction='in')
            ax[row, col].tick_params(labelleft=False)
        if row == len(DSs) - 1:
            ax[row, col].set_xlabel('Distance to phase centre (deg)', fontsize=18)
        else:
            ax[row, col].tick_params(axis='x', direction='in')
            ax[row, col].tick_params(labelbottom=False)

fig.subplots_adjust(hspace=0, wspace=0)


figure_name = ('{0}AllCorrelations_'
               '{1}_'
               'UncorrVsCorr_'
               '{2}_'
               'db{3}_'
               'ds{4}_'
               'SN{5}.png').format(file_path, ext,
                                   fc, 
                                   db, ds,
                                   minimum_sn)
fig.savefig(figure_name,
            bbox_inches='tight')

plt.close(fig)

#################################################

max_dpc = 2.0

use_area = True
step = 0.05
area_step = 0.1

ext = 'PS'

ymins = []
ymaxes = []

fig, ax = plt.subplots(8, 2, figsize=(15, 25), sharey=True)

if use_area:
    figure_name = ('{0}Torus_'
                   'UncorrVsCorr_'
                   'area{1}_'
                   '{2}_'
                   'SN{3}.png').format(file_path,
                                       area_step,
                                       ext,
                                       minimum_sn)
else:
    figure_name = ('{0}Torus_'
                   'UncorrVsCorr_'
                   'radius{1}_'
                   '{2}_'
                   'SN{3}.png').format(file_path,
                                       step,
                                       ext,
                                       minimum_sn)

for d, ds in enumerate(DSs):
    
    
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    print('Working on {0} MHz, ds {1}'.format(freq, ds))
    for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]
        print('Working on {0} {1}'.format(fc, fc_err))

        r1 = 0.

        col = f
        row = d

        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)

        
        if fc == 'f_int':
            label_string = 'Uncorrected light curves'
        elif fc == 'f_int_median_scaled':
            label_string = 'Corrected light curves'
        ax0_text = AnchoredText('{0}, '
                                '{1}'.format(label_string, freq_str),
                                loc='lower right',
                                prop=dict(size=14, color='Grey'),
                                borderpad=0.1,
                                frameon=False)

        all_corrs = pd.read_csv(correlation_filename)
        r1s = []

        count = 0
        while r1 < max_dpc:
            r1s.append(r1)
            if use_area:
                r2 = annulus_outer_r(r1, area_step)
            else:
                r2 = r1 + step

            qry = '{0} < s1_{1} <= {2} and {0} < s2_{1} <= {2}'.format(r1, pc_col, r2)
            annulus = all_corrs.query(qry)

            ax[row, col].scatter(annulus['s1_{}'.format(pc_col)],
                                 annulus['correlation coefficient'],
                                 marker='.', s=12)

            r1 = r2
            count += 1

        ax[row, col].set_xlim(0, 1.3)
        
        ax[row, col].add_artist(ax0_text)
        
        if col == 0:
            ax[row, col].set_xticklabels([0, 0.25, 0.5, 0.75, 1.0])
            ax[row, col].set_ylabel('Correlation\ncoefficient', fontsize=18)
        else:
            ax[row, col].tick_params(axis='y', direction='in')
            ax[row, col].tick_params(labelleft=False)
        if row == len(DSs) - 1:
            ax[row, col].set_xlabel('Distance to phase centre (deg)', fontsize=18)
        else:
            ax[row, col].tick_params(axis='x', direction='in')
            ax[row, col].tick_params(labelbottom=False)

        ymin, ymax = ax[row, col].get_ylim()
        ymins.append(ymin)
        ymaxes.append(ymax)

    ymin = np.min(np.array(ymins)) - 0.02
    ymax = np.max(np.array(ymaxes)) + 0.02
    for a in range(2):        
        for r in r1s:
            ax[row, a].plot(np.ones(10)*r,
                            np.linspace(ymin-50, ymax+50, 10),
                            '--', alpha=0.2,
                            c='Grey', zorder=0, lw=2)

        ax[row, a].set_ylim(ymin, ymax)    

    fig.subplots_adjust(hspace=0, wspace=0)
    
fig.savefig(figure_name,
            bbox_inches='tight')

plt.close(fig)

#################################################

ext = 'PS'
for d, ds in enumerate(DSs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    print('Working on {0} MHz, ds {1}'.format(freq, ds))
    
    figure_name = ('{0}QQPlot_'
                   '{1}MHz_'
                   'db{2}_'
                   'ds{3}_'
                   '{4}_'
                   'SN{5}.png').format(file_path,
                                       freq,
                                       db, ds,
                                       ext,
                                       minimum_sn)

    for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]
        print('Working on {0} {1}'.format(fc, fc_err))

        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)
        
        all_corrs = pd.read_csv(correlation_filename)
        
        if fc == 'f_int':
            label_string = 'Uncorrected light curves, {0}'.format(freq_str)
        elif fc == 'f_int_median_scaled':
            label_string = 'Corrected light curves, {0}'.format(freq_str)
        ax0_text = AnchoredText(label_string,
                                loc='lower right',
                                prop=dict(size=16, color='Grey'),
                                borderpad=0.1,
                                frameon=False)
        
        ax[f].add_artist(ax0_text)
        
        res = spstats.probplot(all_corrs['correlation coefficient'], plot=ax[f])
    fig.tight_layout()
    
    fig.savefig(figure_name,
                bbox_inches='tight')
    
    plt.close(fig)

#################################################

ext = 'PS'
fig, ax = plt.subplots(8, 2, figsize=(14, 26), sharex=True, sharey=True)
figure_name = ('{0}FluxPlot_'
               '{1}_'
               'SN{2}.png').format(file_path,
                                   ext,
                                   minimum_sn)

for d, ds in enumerate(DSs):
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    print('Working on {0} MHz, ds {1}'.format(freq, ds))
    for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]
        print('Working on {0} {1}'.format(fc, fc_err))

        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)
        
        all_corrs = pd.read_csv(correlation_filename)
        
        all_corrs = all_corrs.sort_values(by=['s1_median_{}'.format(fc),
                                              's2_median_{}'.format(fc)])
        leg_5 = all_corrs.query('`correlation coefficient` <= 0.5')

        ax[d, f].scatter(np.log10(leg_5['s1_median_{}'.format(fc)]),
                      np.log10(leg_5['s2_median_{}'.format(fc)]),
                      c='Black', marker='.')
        
        geq_5 = all_corrs.query(('`correlation coefficient` > 0.5 and '
                                 '`correlation coefficient` <= 0.6'))
        geq_6 = all_corrs.query(('`correlation coefficient` > 0.6 and '
                                 '`correlation coefficient` <= 0.7'))
        geq_7 = all_corrs.query(('`correlation coefficient` > 0.7 and '
                                 '`correlation coefficient` <= 0.8'))
        geq_8 = all_corrs.query(('`correlation coefficient` > 0.8'))
        
        ax[d, f].scatter(np.log10(geq_5['s1_median_{}'.format(fc)]),
                      np.log10(geq_5['s2_median_{}'.format(fc)]),
                      c='DarkGrey', marker='.')
        ax[d, f].scatter(np.log10(geq_6['s1_median_{}'.format(fc)]),
                      np.log10(geq_6['s2_median_{}'.format(fc)]),
                      c='Yellow', marker='.')
        ax[d, f].scatter(np.log10(geq_7['s1_median_{}'.format(fc)]),
                      np.log10(geq_7['s2_median_{}'.format(fc)]),
                      c='Orange', marker='.')
        ax[d, f].scatter(np.log10(geq_8['s1_median_{}'.format(fc)]),
                      np.log10(geq_8['s2_median_{}'.format(fc)]),
                      c='Red', marker='.')
        
        if fc == 'f_int':
            label_string = 'Uncorrected light curves'
        elif fc == 'f_int_median_scaled':
            label_string = 'Corrected light curves'
        ax0_text = AnchoredText('{0}, '
                                '{1}'.format(label_string, freq_str),
                                loc='lower right',
                                prop=dict(size=14, color='Grey'),
                                borderpad=0.1,
                                frameon=False)
        
        if f == 0:
            ax[d, f].set_ylabel(r'log$_{10}$Flux density\n[Jy]', fontsize=16)
        else:
            ax[d, f].tick_params(axis='y', direction='in')
            ax[d, f].tick_params(labelleft=False)
        if d == 7:
            ax[d, f].set_xlabel(r'log$_{10}$Flux density [Jy]', fontsize=16)
        else:
            ax[d, f].tick_params(axis='x', direction='in')
            ax[d, f].tick_params(labelbottom=False)
        
        ax[d, f].add_artist(ax0_text)
            
pointsize = 5
black_dot = mlines.Line2D([], [], color='None',
                          marker='o',
                          markerfacecolor='Black',
                          markeredgecolor='Black',
                          markersize=pointsize+5,
                          label='Correlation coefficient < 0.5')
grey_dot = mlines.Line2D([], [], color='None',
                          marker='o',
                          markerfacecolor='DarkGrey',
                          markeredgecolor='DarkGrey',
                          markersize=pointsize+5,
                          label='0.5 < Correlation coefficient < 0.6')
yellow_dot = mlines.Line2D([], [], color='None',
                          marker='o',
                          markerfacecolor='Yellow',
                          markeredgecolor='Yellow',
                          markersize=pointsize+5,
                          label='0.6 < Correlation coefficient < 0.7')
orange_dot = mlines.Line2D([], [], color='None',
                          marker='o',
                          markerfacecolor='Orange',
                          markeredgecolor='Orange',
                          markersize=pointsize+5,
                          label='0.7 < Correlation coefficient < 0.8')
red_dot = mlines.Line2D([], [], color='None',
                          marker='o',
                          markerfacecolor='Red',
                          markeredgecolor='Red',
                          markersize=pointsize+5,
                          label='0.8 < Correlation coefficient')

leg0 = ax[7, 0].legend(handles=[black_dot, grey_dot, yellow_dot, orange_dot, red_dot],
                       fontsize=15, frameon=True, loc='lower left', ncol=2,
                       bbox_to_anchor= (0.0, -0.75), borderaxespad=0,
                       edgecolor='Black')

fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(figure_name, bbox_inches='tight')

plt.close(fig)

#################################################

ds = 33
ext = 'PS'
fb_file = glob.glob(('{0}rcat{1}_'
                     'ra*_dec*_'
                     'db{2}_ds{3}_'
                     '{4}.csv').format(lightcurve_path,
                                       int(fb_rcats[ds]),
                                       db,
                                       ds,
                                       ext))[0]
fb_info = pd.read_csv(fb_file)
gx_file = glob.glob(('{0}rcat{1}_'
                     'ra*_dec*_'
                     'db{2}_ds{3}_'
                     '{4}.csv').format(lightcurve_path,
                                       int(gx_rcats[ds]),
                                       db,
                                       ds,
                                       ext))[0]
gx_info = pd.read_csv(gx_file)
psr_file = glob.glob(('{0}rcat{1}_'
                      'ra*_dec*_'
                      'db{2}_ds{3}_'
                      '{4}.csv').format(lightcurve_path,
                                        int(psr_rcats[ds]),
                                        db,
                                        ds,
                                        ext))[0]
psr_info = pd.read_csv(psr_file)

flux_col = 'f_int'
flux_err = 'f_int_err'
mjd_col = 'mjd'

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for df, dataframe in enumerate([fb_info,
                            gx_info,
                            psr_info]):
    for flux_col in ['f_int', 'f_int_median_scaled']:
        if flux_col == 'f_int':
            cola = [fb_col, gx_col, psr_col][df]
            ax[df].errorbar(dataframe[mjd_col], dataframe[flux_col]*1e3,
                            yerr=dataframe[flux_err]*1e3, fmt='o', c=cola,
                            markersize=8,
                            label='Uncorrected light curve')
        else:
            cola = ['#C82D35', '#256E7E', '#1C1862'][df]
            ax[df].errorbar(dataframe[mjd_col], dataframe[flux_col]*1e3,
                            yerr=dataframe[flux_err]*1e3, fmt='o', c=cola,
                            markersize=6,
                            label='Corrected light curve')

    name = ['(a) MKT J170456.2-482100',
            '(b) GX 339-4',
            '(c) PSR J1703-4851'][df]
    legend = ax[df].legend(loc='upper right', fontsize=12)
    legend.set_title(name, prop={'size':14})
    legend._legend_box.align = "left"

for a in range(2):
    ax[a].tick_params(axis='x', direction='in')
    ax[a].tick_params(labelbottom=False)
for a in range(3):
    ax[a].set_ylabel('Flux density [mJy]', fontsize=20)
ax[2].set_xlabel('MJD', fontsize=22)

fig.subplots_adjust(hspace=0)

fig.savefig('{0}KnownVariables_LightCurves.png'.format(file_path),
            bbox_inches='tight')

plt.close(fig)

#################################################


ext = 'ES'
for d, ds in enumerate(DSs[6:7], 6):
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    print('Working on {0} MHz, ds {1}'.format(freq, ds))

    for f, fc in enumerate(['f_int', 'f_int_median_scaled'][:1]):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]
        print('Working on {0} {1}'.format(fc, fc_err))

        correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                '{2}_'
                                'db{3}_'
                                'ds{4}_'
                                '{5}_'
                                'minSN{6}.csv').format(file_path,
                                                       freq, fc, 
                                                       db, ds,
                                                       ext,
                                                       minimum_sn)
        
        all_corrs = pd.read_csv(correlation_filename)

qry = '`correlation coefficient` > 0.85'
bob = all_corrs.query(qry)
unique_rcats = pd.concat([bob['s1_runcat'],
                          bob['s2_runcat']]).unique()

bob_rcats = []
for r, row in bob.iterrows():
    bob_rcats.append([row['s1_runcat'],
                      row['s2_runcat'],
                      row['correlation coefficient']])
bob_rcats = np.array(bob_rcats)
unique_s1_index = np.unique(bob_rcats[:, 0], return_index=True)
bob_rcats = bob_rcats[unique_s1_index[1]]

ext = 'ES'
fig, ax = plt.subplots(8, 1, figsize=(10, 20), sharex=True)
col = ['Black', 'Grey']
mss = [30, 20]
count = 0
for r, row in enumerate(bob_rcats):
    medians = []
    for a in range(2):
        filename = glob.glob(('{0}rcat{1}_'
                              'ra*_dec*_'
                              'db{2}_ds{3}_'
                              '{4}.csv').format(lightcurve_path,
                                                int(row[a]),
                                                db,
                                                ds,
                                                ext))[0]
        source_info = pd.read_csv(filename)
        medians.append(np.nanmedian(source_info['f_int']))

    if ((medians[0]/medians[1]>0.5) and
        (medians[0]/medians[1]<1.5) and
        count<8):
        for a in range(2):
            filename = glob.glob(('{0}rcat{1}_'
                                  'ra*_dec*_'
                                  'db{2}_ds{3}_'
                                  '{4}.csv').format(lightcurve_path,
                                                    int(row[a]),
                                                    db,
                                                    ds,
                                                    ext))[0]
            source_info = pd.read_csv(filename)
            ax[count].scatter(source_info['mjd'], source_info['f_int']*1e3, s=mss[a], c=col[a])

        ax0_text = AnchoredText('r = {:.3f}'.format(row[2]),
                                loc='upper left',
                                prop=dict(size=14, color='Grey'),
                                borderpad=0.05,
                                frameon=True)
        ax[count].add_artist(ax0_text)
        count += 1

for a in range(7):
    ax[a].tick_params(axis='x', direction='in')
    ax[a].tick_params(labelbottom=False)
for a in range(8):
    ax[a].set_ylabel('Flux density\n[mJy]', fontsize=18)
ax[7].set_xlabel('MJD', fontsize=20)

fig.subplots_adjust(hspace=0)

fig.savefig('{0}CorrelationDemo_LightCurves.png'.format(file_path),
            bbox_inches='tight')

plt.close(fig)

#################################################

ext = 'PS'
for d, ds in enumerate(DSs[6:7], 6):
    print('----------------------------------')
    print('----------------------------------')
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = 'MFS'
    else:
        freq_str = '{}MHz'.format(freq)
    print('Working on {0} MHz, ds {1}'.format(freq, ds))

    for f, fc in enumerate(['f_int', 'f_int_median_scaled'][:1]):
        fc_err = ['f_int_err', 'f_int_median_scaled_err'][f]
        print('Working on {0} {1}'.format(fc, fc_err))

        var_params = glob.glob(('{0}VariabilityParams_'
                                'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                             db,
                                                             ds, ext)))[0]
        vp_all = pd.read_csv(var_params)
        vp_sn2 = vp_all[vp_all['detected SN2']=='Yes']

qry = 'eta_param > 0.8 and median_scaled_eta < 0.8'
bob = vp_sn2.query(qry)

bob_rcats = []
for r, row in bob.iterrows():
    bob_rcats.append(row['runcat'])
bob_rcats = np.array(bob_rcats)

ext = 'PS'
fig, ax = plt.subplots(4, 2, figsize=(16, 12), sharex=True, sharey='row')
col = ['Black', 'Grey']
for r, row in enumerate([153385, 152744, 148925, 148377]):
    filename = glob.glob(('{0}rcat{1}_'
                          'ra*_dec*_'
                          'db{2}_ds{3}_'
                          '{4}.csv').format(lightcurve_path,
                                            int(row),
                                            db,
                                            ds,
                                            ext))[0]
    source_info = pd.read_csv(filename)
    ax[r, 0].errorbar(source_info['mjd'],
                       source_info['f_int']*1e3,
                       yerr=source_info['f_int_err']*1e3,
                       fmt='o', c='Black')
    ax[r, 1].errorbar(source_info['mjd'],
                       source_info['f_int_median_scaled']*1e3,
                       yerr=source_info['f_int_median_scaled_err']*1e3,
                       c='Black', fmt='o')
    
    a = 0
    xmin, xmax = ax[r, a].get_xlim()
    ax[r, a].plot(np.linspace(xmin, xmax, 10),
                  np.ones(10)*np.nanmedian(source_info['f_int']*1e3),
                  '--', c='Black', alpha=0.4, lw=3)
    ax[r, a].set_xlim(xmin, xmax)
    a = 1
    xmin, xmax = ax[r, a].get_xlim()
    ax[r, a].plot(np.linspace(xmin, xmax, 10),
                  np.ones(10)*np.nanmedian(source_info['f_int_median_scaled']*1e3),
                  '--', c='Black', alpha=0.4, lw=3)
    ax[r, a].set_xlim(xmin, xmax)
    
    c1_text = (r'$\eta$={0:.4f} $V$={1:.4f} $\xi$={2:.4f}').format(np.nanmedian(source_info['eta_param']),
                                       np.nanmedian(source_info['V_param']),
                                       np.nanmedian(source_info['mad_param']))
    c2_text = r'$\eta$={0:.4f} $V$={1:.4f} $\xi$={2:.4f}'.format(np.nanmedian(source_info['median_scaled_eta']),
                                                     np.nanmedian(source_info['median_scaled_V']),
                                                     np.nanmedian(source_info['median_scaled_madp']))
    
    ax1_text = AnchoredText(c1_text,
                            loc='upper left',
                            prop=dict(size=16, color='Grey'),
                            borderpad=0.05,
                            frameon=False)
    ax[r, 0].add_artist(ax1_text)
    ax2_text = AnchoredText(c2_text,
                            loc='upper left',
                            prop=dict(size=16, color='Grey'),
                            borderpad=0.03,
                            frameon=False)
    ax[r, 1].add_artist(ax2_text)

    ax00_text = AnchoredText('Uncorrected light curves',
                            loc='lower right',
                            prop=dict(size=14, color='Grey'),
                            borderpad=0.1,
                            frameon=False)
    ax11_text = AnchoredText('Corrected light curves',
                            loc='lower right',
                            prop=dict(size=14, color='Grey'),
                            borderpad=0.1,
                            frameon=False)
    ax[r, 0].add_artist(ax00_text)
    ax[r, 1].add_artist(ax11_text)

for a in range(3):
    ax[a, 0].tick_params(axis='x', direction='in')
    ax[a, 0].tick_params(labelbottom=False)
    ax[a, 1].tick_params(axis='x', direction='in')
    ax[a, 1].tick_params(labelbottom=False)
for a in range(4):
    ax[a, 0].set_ylabel('Flux density\n[mJy]', fontsize=18)
    ax[a, 1].tick_params(axis='y', direction='in')
    ax[a, 1].tick_params(labelleft=False)
ax[3, 0].set_xlabel('MJD', fontsize=20)
ax[3, 1].set_xlabel('MJD', fontsize=20)

fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig('{0}CorrectionDemo_LightCurves.png'.format(file_path), bbox_inches='tight')

plt.close(fig)

#################################################

ext = 'PS'
for d, ds in enumerate(DSs):
    freq = freqs[d]
    if freq == 'MFS':
        freq_str = '1283\,MHz (MFS)'
    else:
        freq_str = '{}\,MHz'.format(freq)
    var_params = glob.glob(('{0}VariabilityParams_'
                            'db{1}_ds{2}_{3}.csv'.format(lightcurve_path,
                                                         db,
                                                         ds, ext)))[0]
    vp_all = pd.read_csv(var_params)
    
    vp_2 = vp_all[vp_all['detected SN2'] == 'Yes']
    vp_3 = vp_all[vp_all['detected SN3'] == 'Yes']
    
    print('{0} & {1} & {2} & {3} \\\\'.format(freq_str,
                                              len(vp_all.index),
                                              len(vp_2.index),
                                              len(vp_3.index)))

#################################################

for d, ds in enumerate(DSs):
    ext = 'PS'
    freq = freqs[d]
    try:
        row = [int(freq)]
    except ValueError:
        row = [freq]

    for num_std in [3, 4, 5]:
        for f, fc in enumerate(['f_int', 'f_int_median_scaled']):
            fc_err = '{}_err'.format(fc)
            correlation_filename = ('{0}AllCorrelations_{1}MHz_'
                                    '{2}_'
                                    'db{3}_'
                                    'ds{4}_'
                                    '{5}_'
                                    'minSN{6}.csv').format(file_path,
                                                           freq, fc, 
                                                           db, ds,
                                                           ext,
                                                           minimum_sn)
            all_corrs = pd.read_csv(correlation_filename)
            corr_coeffs = all_corrs['correlation coefficient']
            
            mean_ = np.nanmean(corr_coeffs)
            std_ = np.nanstd(corr_coeffs)
            total_length = len(np.array(corr_coeffs))
            
            rv = spstats.norm(mean_, std_)
            
            theory = 1. - (rv.cdf(mean_+num_std*std_) - rv.cdf(mean_-num_std*std_))
            num_outliers = (len(np.where(np.logical_or(corr_coeffs<mean_-num_std*std_,
                                                       corr_coeffs>mean_+num_std*std_))[0]))
            
            row.append(num_outliers)
            row.append(theory * total_length)
    print(('{0} & '
           '{1} & {2:.2f} & '
           '{3} & {4:.2f} & '
           '{5} & {6:.2f} & '
           '{7} & {8:.2f} & '
           '{9} & {10:.2f} & '
           '{11} & {12:.2f} \\\\').format(row[0],
                                          row[1], row[2], row[3], row[4],
                                          row[5], row[6], row[7], row[8],
                                          row[9], row[10], row[11], row[12]))

#################################################
