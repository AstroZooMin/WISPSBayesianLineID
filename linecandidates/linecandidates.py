import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from utils.match_cats import match_cats
from make_plots import *


def prep_column(data, shape, inds, ones=True):
    """Prep a column to be added to the catalog."""
    if ones:
        arr = np.ones(shape, dtype=int)*-99
    else:
        arr = np.zeros(shape, dtype=float)

    arr[inds] = data
    return arr


def mad_based_outlier(data, thresh=3.5, maxprob=5):
    """Find outliers using the median absolute deviation.

    The median absolute deviation is median(abs(data - median(data)))

    Args:
        data (float): array, here of probabilities
        thresh (Optional[float]): the modified z-score to use as a 
            threshold. Data values with a modified z_score (based
            on the MAD) greater than this value will be classified 
            as outliers
        maxprob (Optional[float]): max probability threshold to use 
            if the MAD is zero. This occurs in a few cases, usually 
            when there is only a single peak in the probability 
            distribution. All source with MAD=0 and a maximum 
            probability > maxprob will be classified as an outlier
            
    Returns:
        flag (bool): boolean array, True if source is identified as
            an outlier, False if source is not identified as an outlier
    """
    if len(data.shape) == 1:
        data = data[:,None]
    median = np.median(data, axis=0)
    diff = np.sum((data - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    # for a few objects
    if med_abs_deviation != 0.:
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh
    elif (med_abs_deviation == 0.) & (np.max(data) > maxprob):
        # a handful of sources have a single probability peak.
        return np.ones(data.shape[0], dtype=bool)
    elif (med_abs_deviation == 0.) & (np.max(data) < maxprob):
        return np.zeros(data.shape[0], dtype=bool)


def find_outliers(t):
    """Identify sources with peaked probability distributions.

    A peaked probability distribution indicates there is a high 
    probability that the source is in a narrow redshift range. 
    Such a probability distribution may indicate that there is 
    an emission line in the spectrum. We identify such emission
    line candidates if the 'best-fit' template redshift represents
    an outlier in the probability distribution.

    Args:
        t (table): data table

    Returns:
        j (bool): boolean array, True if the max probability is an 
            outlier, False if the max probability is not an outlier.
    """
    z = np.arange(0,3.31,0.01)
    j = np.zeros(t['obj'].shape[0], dtype=bool)
    for i in range(t['obj'].shape[0]):
        prob = t['prob'][i]
        prob[np.isnan(prob)] = 0
        zfit = t['z'][i]
        # skip objects with probability distributions of all zero
        if not np.any(prob):
            continue
        # find all outliers in the probability distribution
        o = mad_based_outlier(prob)
        if zfit in z[o]:
            j[i] = True
    return j


def flag_edge(t):
    """Flag source with emission lines on the right edge of the chip.

    We do not have 0th order information along the right edge, and so 
    we cannot rule out contamination in the spectra of sources in this 
    region. We therefore reject sources that are 

    Use the wavelength of the emission line and the bounding box 
    coordinates in the header to determine the x-position of the 
    emission line in the FLT. The x-cutoffs are: 
        G102: x=760
        G141: x=820
    and are calculated using the config files for the most common 
    filter combination (G102-F110 and G141-F160) and assuming a 
    y-position in the middle of the chip. 
    
    The emission line is assumed to be [OII], [OIII], or Ha. The 
    corresponding wavelengths of each emission line are checked. 
    The source is considered 'good' if ANY of the three wavelengths
    places the emission line to the left of the x-cutouff. The source
    is considered 'bad' if ALL if the wavelengths place the emission 
    line to the right of the x-cutoff. 

    Args:
        t (table): data table

    Returns:
        waveflag (bool): boolean array, True if any emission line falls 
            to the left of the x-cutoff position, False if all emission 
            lines fall to the right of the x-cutoff position.
    """
    # find wavelength of emission lines at this redshift
    wave = np.zeros((t['obj'].shape[0],3), dtype=float)
    wave[:,0] = (t['z']+1.) * 6563.
    wave[:,1] = (t['z']+1.) * 5007.
    wave[:,2] = (t['z']+1.) * 3727.

    baseg102 = 'Par302/G102_DRIZZLE/aXeWFC3_G102_mef_ID'
    baseg141 = 'Par302/G141_DRIZZLE/aXeWFC3_G141_mef_ID'

    waveflags = np.zeros(wave.shape, dtype=bool)
    for i in range(t['obj'].shape[0]):
        # only need to check wavelength for objects with x >/~ 1000
        if t['x'][i] < 1000:
            waveflags[i,0] = True
            continue

        try:
            g102hdr = fits.getheader('%s%i.fits'%(baseg102,t['obj'][i]), 'SCI')
        except IOError:
            g102hdr = None
        try:
            g141hdr = fits.getheader('%s%i.fits'%(baseg141,t['obj'][i]), 'SCI')
        except IOError:
            g141hdr = None

        # check for each value
        for j,w in enumerate(wave[i]):
            if (w >= 8000.) & (w < 11500.):
                if g102hdr is None:
                    continue
                hdr = g102hdr
                maxx = 760
            elif (w >= 11500.) & (w <= 16500.):
                if g141hdr is None:
                    continue
                hdr = g141hdr
                maxx = 820
            else:
                continue
            # convert wavelength of emission to pixels
            xpos = (w - hdr['CRVAL1']) / hdr['CDELT1'] + hdr['CRPIX1'] +\
                         hdr['BB0X']
            if xpos < maxx:
                waveflags[i,j] = True
    return np.any(waveflags, axis=1)
            

def combine_catalogs(templatefits, par='Par302'):
    """Match catalog of template fits to WISP catalog and line list.

    Add columns to the template catalog, including 
        - RA and Dec; 
        - x and y positions in the drizzled direct image; 
        - J magnitude; 
        - the redshift, redshift error, flux, flux error, observed 
            wavelength and quality flag of [OII], [OIII] and Halpha 
            for sources matched to the line list; 
        - the cleanliness measured in each grism; 
        - whether the 'best-fit' template redshift is an outlier in 
            the probability distribution; 
        - whether the emission line falls in the region of the chip with 
            0th order information available.
    
    Args:
        templatefits (str): catalog of template fits in main repo
        par (Optional[str]): Name of field, default is 'Par302'

    Outputs:
        [base]_wcs.fits: a fits table including all columns, saved in 
            the working directory
    """
    f = fits.getdata(templatefits)
    # WISP catalog
    d = np.genfromtxt('%s/DATA/DIRECT_GRISM/fin_F110.cat'%par)
    # full line list, organized by field
    ll = fits.getdata('full_linelist.fits')
    ll = ll[ll['par'] == 302] 
    # contamination measures
    g102 = fits.getdata('g102_withCleanliness.fits')
    g141 = fits.getdata('g141_withCleanliness.fits')

    # Add in x,y RA,Dec, Jmag,Hmag from WISP catalog
    # Assumption is that the objs in template cat match the WISP cat exactly
    if f.shape[0] != d[:,0].shape[0]:
        print 'Catalog sizes do not match.\n\t%i sources in template catalog\n\t%i sources in WISP catalog' % (f.shape[0], d[:,0].shape[0])
        exit()

    # create table
    t = Table([f['target'], d[:,2], d[:,3], d[:,7], d[:,8], d[:,12],
               f['redshift'], f['redshift_err'], f['model'],
               f['probability'], f['normalization'], f['has_line']],
               names=('obj', 'x', 'y', 'ra', 'dec', 'jmag', 'z', 'zerr', 
                      'model', 'prob', 'norm', 'has_line'))

    # match catalog to the line list
    idx,sep = match_cats(ll['ra'], ll['dec'], d[:,7], d[:,8])
    mat = (sep.value*3600. <= 0.2)
    # from line lists, keep IDs and for each of OII, OIII, Ha:
    #   z, zerr
    #   flux, fluxerr
    #   observed wavelength
    #   quality flag
    shape = d[:,0].shape[0]
    ids = prep_column(ll['obj'][mat], shape, idx[mat])
    lines = ['OII', 'OIII', 'H_alpha']
    cols = ['z', 'z_err', 'flux', 'flux_err', 'lambda_obs']
    for line in lines:
        for col in cols:
            c = prep_column(ll['%s_%s'%(line,col)][mat], shape, idx[mat], 
                            ones=False)
            t.add_column(Column(data=c, name='%s_%s'%(line,col)), 10)
    for line in lines:
        c = prep_column(ll['%s_quality_flag'%line][mat], shape, idx[mat])
        t.add_column(Column(data=c, name='%s_quality_flag'%line))

    # add in cleanliness measures
    w102 = np.in1d(d[:,1], g102['object'])
    c102 = prep_column(g102['cleanliness'], shape, w102, ones=False)
    c102[np.isnan(c102)] = 0.
    w141 = np.in1d(d[:,1], g141['object'])
    c141 = prep_column(g141['cleanliness'], shape, w141, ones=False)
    c141[np.isnan(c141)] = 0.
    t.add_columns([Column(data=c102, name='c102'),
                   Column(data=c141, name='c141')])

    # find objects with potential peaks in probability distribution
    peaked = find_outliers(t)
    t.add_column(Column(data=peaked, name='peaked'))

    # flag objects that are too close to the right edge - no 0th orders
    edge = flag_edge(t)
    t.add_column(Column(data=edge, name='edge'))

    t.write(os.path.basename(templatefits.replace('.fits','_wcs.fits')), 
            format='fits')


def main():
    catalogs = ['Par302_estimates.fits','Par302_no_added_lines_estimates.fits']
    directories = ['addedlines', 'noaddedlines']

    for c,d in zip(catalogs,directories):
        cat = os.path.basename(c.replace('.fits', '_wcs.fits'))
        if not os.path.exists(cat):
            combine_catalogs(os.path.join('..',c), par='Par302')

        # check the emission line candidates identified by selection
        check_candidates(cat, d)
        # plot the x,y positions on the drizzled, direct image
        plot_positions(cat, d)
        # plot the frequency / distribution of 'best-fit' models
        plot_models(cat, d)       

    # plot a comparison of the 'best-fit' redshifts
    compare_redshift(catalogs[0].replace('.fits','_wcs.fits'), 
                     catalogs[1].replace('.fits','_wcs.fits'))


if __name__ == '__main__':
    main()


