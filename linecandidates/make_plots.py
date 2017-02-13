import os
from collections import Counter
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def plot_models(catalog, directory, show=False):
    """Make plots of the 'best-fit' template models for each source.

    Creates a figure with two plots:
        1. a histogram of the number of sources fit by each model
        2. a scatterplot of the maximum probability and redshift of 
            each source, color-coded by the best-fit model

    In both plots, groups of models are assigned like colors. A list 
    of the template numbers in plot #1 and the corresponding model names 
    is saved as 'model_legend.dat'

    Args:
        catalog (str): filename of template fits catalog
        directory (str): directory in which to save output plot
        show (Optional[bool]): Set to True to show plot before saving

    Outputs:
        model_fits.pdf, saved in [directory] 
        model_legend.dat, saved in [directory] 
    """
    print catalog, directory
    t = fits.getdata(catalog)

    # histogram of models
    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    models = sorted(np.unique(t['model']))
    m = Counter(t['model'])
    xval = np.arange(len(models))
    f = open(os.path.join(directory,'models_legend.dat'), 'w')
    f.write('# index\tmodel\n')
    for i in range(xval.shape[0]):
        f.write('%i\t%s\n' % (xval[i], models[i]))
    f.close()

    mod = [m[i] for i in models]
    nn = [m[i] for i in models if i == '-99']
    con = [m[i] for i in models if i[0:3] == 'CON']
    cww = [m[i] for i in models if i[0:3] == 'CWW']
    kin = [m[i] for i in models if i[0:3] == 'KIN']
    ssp = [m[i] for i in models if i[0:3] == 'SSP']
    tau = [m[i] for i in models if i[0:3] == 'TAU']
    xnn = np.arange(len(nn))
    xcon = np.arange(len(con))
    xcww = np.arange(len(cww))
    xkin = np.arange(len(kin))
    xssp = np.arange(len(ssp))
    xtau = np.arange(len(tau))

    ax1.bar([0], nn, align='center', width=1, linewidth=0,
             color='k', alpha=0.5, label='-99')
    xval = xcon + 1
    ax1.bar(xval, con, align='center', width=1, linewidth=0,
             color='r', alpha=0.5, label='CON')
    xval = xcww + np.max(xval) + 1
    ax1.bar(xval, cww, align='center', width=1, linewidth=0,
            color='m', alpha=0.5, label='CWW')
    xval = xkin + np.max(xval) + 1
    ax1.bar(xval, kin, align='center', width=1, linewidth=0,
             color='g', alpha=0.5, label='KIN')
    xval = xssp + np.max(xval) + 1
    ax1.bar(xval, ssp, align='center', width=1, linewidth=0,
             color='c', alpha=0.5, label='SSP')
    xval = xtau + np.max(xval) + 1
    ax1.bar(xval, tau, align='center', width=1, linewidth=0,
             color='b', alpha=0.5, label='TAU')
    ax1.legend()
    ax1.set_xlabel('Template', fontsize=20)
    ax1.set_ylabel('n objects', fontsize=20)

    # scatter plot of models
    maxprobs = np.max(t['prob'], axis=1)
    maxprobs[np.isnan(maxprobs)] = 0.
    z = t['z']

    base = np.array([t['model'][i][:3] for i in range(t['obj'].shape[0])])
    wnn = np.where(base == '-99')
    wcon = np.where(base == 'CON')
    wcww = np.where(base == 'CWW')
    wkin = np.where(base == 'KIN')
    wssp = np.where(base == 'SSP')
    wtau = np.where(base == 'TAU')
    ax2.scatter(z[wnn], maxprobs[wnn], edgecolor='none', color='k', 
                s=50, alpha=0.6)
    ax2.scatter(z[wcon], maxprobs[wcon], edgecolor='none', color='r', 
                s=50, alpha=0.6)
    ax2.scatter(z[wcww], maxprobs[wcww], edgecolor='none', color='m', 
                s=50, alpha=0.6)
    ax2.scatter(z[wkin], maxprobs[wkin], edgecolor='none', color='g', 
                s=50, alpha=0.6)
    ax2.scatter(z[wssp], maxprobs[wssp], edgecolor='none', color='c', 
                s=50, alpha=0.6)
    ax2.scatter(z[wtau], maxprobs[wtau], edgecolor='none', color='b', 
                s=50, alpha=0.6)
    ax2.set_xlabel('template redshift', fontsize=20)
    ax2.set_ylabel('max probability', fontsize=20)
    ax2.set_yscale('log')
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(1.e-1, 2.e2)

    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(os.path.join(directory,'model_fits.pdf'))


def plot_linelist(ax, t, xparam, yval, cand, miss=None):
    """Identify sources from the line list in a plot.

    Sources are plotted by their flag value in the line list.
    The flags are:
       +1 - potential flux issues
       +2 - only multi-line for close, S/N < 5.0 lines
       +4 - single line sources
       +8 - only identified by single reviewer
      +16 - significant redshift disagreements

    Args:
        ax (matplotlib AxesSubplot): instance of axis for plotting
        t (table): data table
        xparam (str): column name in data table of data to be plotted 
            as the independent variable 
        yval (float,int array): full array of values from the catalog to 
            be plotted as the dependent variable
        cand (bool): boolean array identifying sources that are 
            emission line candidates
        miss (bool): boolean array identifying sources that are  
            missed by selection criteria

    Returns:
        (tuple): tuple containing:
            n4 (int): number of sources with flag < 4
            n8 (int): number of sources with 4 <= flag < 8
            n16 (int): number of sources with 8 <= flag < 16
            n24 (int): number of sources with flag >= 16
    """
    # use the Halpha quality flag as the generic flag for all objects.
    # very few sources have OIII quality flags but not Halpha
    # and these are almost all <4 anyway
    flag = 'H_alpha_quality_flag'
    w4 = cand & (t[flag] != -99) & (t[flag] < 4)
    w8 = cand & (t[flag] >= 4) & (t[flag] < 8)
    w16 = cand & (t[flag] >= 8) & (t[flag] < 16)
    w24 = cand & (t[flag] >= 16)

    ax.scatter(t[xparam][w4], yval[w4], marker='s',
               edgecolor='none', color='#00b200', label='flag < 4', s=100)
    ax.scatter(t[xparam][w8], yval[w8], marker='^',
               edgecolor='none', color='b', label='4 <= flag < 8', s=100)
    ax.scatter(t[xparam][w16], yval[w16], marker='^',
               edgecolor='none', color='#e59400', label='8 <= flag < 16', s=100)
    ax.scatter(t[xparam][w24], yval[w24], marker='^',
               edgecolor='none', color='r', label='flag >= 16', s=100)

    n4 = t['z'][w4].shape[0]
    n8 = t['z'][w8].shape[0]
    n16 = t['z'][w16].shape[0]
    n24 = t['z'][w24].shape[0]

    # add sources that are missed by selection criteria, but 
    # exist in the line list
    if miss is not None:
        w4 = miss & (t[flag] != -99) & (t[flag] < 4)
        w8 = miss & (t[flag] >= 4) & (t[flag] < 8)
        w16 = miss & (t[flag] >= 8) & (t[flag] < 16)
        w24 = miss & (t[flag] >= 16)

        ax.scatter(t[xparam][w4], yval[w4], marker='s',
                   edgecolor='#00b200', color='none', s=100)
        ax.scatter(t[xparam][w8], yval[w8], marker='^',
                   edgecolor='b', color='none', s=100)
        ax.scatter(t[xparam][w16], yval[w16], marker='^', edgecolor='#e59400', 
                   color='none', s=100)
        ax.scatter(t[xparam][w24], yval[w24], marker='^',
                   edgecolor='r', color='none', s=100)
    
    return (n4,n8,n16,n24)


def check_candidates(catalog, directory, show=False):
    """Compare the line candidates that result from each selection criterion.
    
    Selection criteria:
        1. the 'best-fit' template includes emission lines ('has_line'=True)
        2. the probability distribution is peaked at the best-fit redshift, 
            identified using outlier detection
        3. objects too close to the right edge are removed (no 0th order 
            information available)
        4. objects with cleanliness measures in both grisms of <= 0.95

    Creates a figure with four plots showing the:
        1. max probability as a fn of template redshift of emission 
            line candidates selected via outlier detection only
        2. max probability as a fn of template redshift of emission 
            line candidates from plot #1, but rejecting sources too 
            close to the right edge
        3. max probability as a fn of template redshift of emission 
            line candidates from plot #2, but with cleanliness>=0.95
            in both grisms
        4. max probability as a fn of G102 cleanliness of emission 
            line candidates from plot #3
        
    In each figure, open red circles indicate sources that were fit with 
    emission lines but did not make the selection cut as outliers 
    in the probability distribution. 

    Args:
        catalog (str): filename of template fits catalog
        directory (str): directory in which to save output plot
        show (Optional[bool]): Set to True to show plot before saving

    Outputs:
        candidates.pdf, saved in [directory] 
    """
    print catalog, directory

    t = fits.getdata(catalog)
    hl = t['has_line']
    p = t['peaked']
    e = t['edge']
    # probably needs a more sophisticaed cut based on wavelength
    c = (t['c102'] <= 0.95) & (t['c141'] <= 0.95)

    maxprobs = np.max(t['prob'], axis=1)
    maxprobs[np.isnan(maxprobs)] = 0.

    if directory == 'addedlines':
        title = 'additional lines added to templates'
    elif directory == 'noaddedlines':
        title = 'no additional lines added to templates'

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,15))

    ### Candidate selection: outlier detection
    cand = hl & p
    miss = hl & ~p
    ax1.scatter(t['z'][cand], maxprobs[cand], edgecolor='none', alpha=0.7,
                color='k', s=50, label='line candidates')
    ax1.scatter(t['z'][miss], maxprobs[miss], edgecolor='r', #alpha=0.7,
                color='none', s=50, label='missed candidates')
    # add in line list
    n4,n8,n16,n24 = plot_linelist(ax1, t, 'z', maxprobs, cand, miss)
    ax1.text(0.5, 0.92, '%i line candidates to review'%t['z'][cand].shape[0], 
             ha='center', transform=ax1.transAxes, fontsize=15)
    ax1.text(0.5, 0.85, '%i objs recovered from line lists'%(n4+n8+n16+n24),
             ha='center', transform=ax1.transAxes, fontsize=15)
    
    ax1.set_xlabel('template redshift', fontsize=20)
    ax1.set_ylabel('max probability', fontsize=20)
    ax1.set_yscale('log')
    ax1.set_title('Candidate selection: outlier detection')

    ### Candidate selection: outlier detection and edge rejection
    cand = hl & p & e
    miss = hl & ~p & e
    ax2.scatter(t['z'][cand], maxprobs[cand], edgecolor='none', alpha=0.7,
                color='k', s=50, label='line candidates')
    ax2.scatter(t['z'][miss], maxprobs[miss], edgecolor='r', #alpha=0.7,
                color='none', s=50, label='missed candidates')
    # add in line list
    n4,n8,n16,n24 = plot_linelist(ax2, t, 'z', maxprobs, cand, miss)
    ax2.text(0.5, 0.92, '%i line candidates to review'%t['z'][cand].shape[0], 
             ha='center', transform=ax2.transAxes, fontsize=15)
    ax2.text(0.5, 0.85, '%i objs recovered from line lists'%(n4+n8+n16+n24),
             ha='center', transform=ax2.transAxes, fontsize=15)
    
    ax2.set_xlabel('template redshift', fontsize=20)
    ax2.set_ylabel('max probability', fontsize=20)
    ax2.set_yscale('log')
    ax2.set_title('Candidate selection: outlier detection, edge rejection')

    ### Candidate selection: outlier detection, edge and contam rejection
    cand = hl & p & e & c
    miss = hl & ~p & e & c
    ax3.scatter(t['z'][cand], maxprobs[cand], edgecolor='none', alpha=0.7,
                color='k', s=50, label='line candidates')
    ax3.scatter(t['z'][miss], maxprobs[miss], edgecolor='r', #alpha=0.7,
                color='none', s=50, label='missed candidates')
    # add in line list
    n4,n8,n16,n24 = plot_linelist(ax3, t, 'z', maxprobs, cand, miss)
    ax3.text(0.5, 0.92, '%i line candidates to review'%t['z'][cand].shape[0], 
             ha='center', transform=ax3.transAxes, fontsize=15)
    ax3.text(0.5, 0.85, '%i objs recovered from line lists'%(n4+n8+n16+n24),
             ha='center', transform=ax3.transAxes, fontsize=15)
    
    ax3.set_xlabel('template redshift', fontsize=20)
    ax3.set_ylabel('max probability', fontsize=20)
    ax3.set_yscale('log')
    ax3.set_title('Candidate selection: outliers, edge rejection, clean<0.95')

    ### Plot cleanliness for G102
    ax4.scatter(t['c102'][cand], maxprobs[cand], edgecolor='none', alpha=0.7,
                color='k', s=50, label='line candidates')
    ax4.scatter(t['c102'][miss], maxprobs[miss], edgecolor='r', #alpha=0.7,
                color='none', s=50, label='missed candidates')
    # add in line list
    n4,n8,n16,n24 = plot_linelist(ax4, t, 'c141', maxprobs, cand, miss)

    ax4.set_ylim(0.15, 1.5e3)    
    ax4.legend(scatterpoints=1, ncol=2)
    ax4.set_xlabel('G102 cleanliness', fontsize=20)
    ax4.set_ylabel('max probability', fontsize=20)
    ax4.set_yscale('log')
    ax4.set_title('Candidate selection: outliers, edge rejection, clean<0.95')

    fig.suptitle(title, fontsize=20)

#    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(os.path.join(directory, 'candidates.pdf'))
    

def plot_positions(catalog, directory, show=False):
    """Plot the pixel positions of line candidates in the direct image.

    Args:
        catalog (str): filename of template fits catalog
        directory (str): directory in which to save output plot
        show (Optional[bool]): Set to True to show plot before saving

    Outputs:
        xy.pdf, saved in [directory] 
    """
    print catalog, directory 

    t = fits.getdata(catalog)
    hl = t['has_line']
    p = t['peaked']
    e = t['edge']
    cand = hl & p

    fig,ax = plt.subplots(1,1, figsize=(12,12))
    ax.scatter(t['x'][cand], t['y'][cand], edgecolor='none', alpha=0.7,
               color='k', s=50, label='line candidates')
    n4,n8,n16,n24 = plot_linelist(ax, t, 'x', t['y'], cand, miss=None)

    cand = hl & p & ~e
    ax.plot(t['x'][cand], t['y'][cand], 'x', mew=1.5,
               color='r', markersize=10, label='off-edge')

    ax.legend(scatterpoints=1, ncol=3)
    ax.set_xlim(0,2000)
    ax.set_ylim(0,2000)
    ax.set_title('F110W pixel positions', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    if show:
        plt.show()
    fig.savefig(os.path.join(directory, 'xy.pdf'))


def compare_redshift(catalog1, catalog2, show=False):
    """Compare the redshift fits from two sets of template fits.

    Args:
        catalog1 (str): filename of first set of template fits
        catalog2 (str): filename of second set of template fits
        show (Optional[bool]): Set to True to show plot before saving

    Outputs:
        redshift_comparison.pdf
    """
    t1 = fits.getdata(catalog1)
    t2 = fits.getdata(catalog2)
    # use the candidates 
    hl = t1['has_line']
    p = t1['peaked']
    e = t1['edge']
    cand = hl & p

    fig,ax = plt.subplots(1,1, figsize=(12,12))
    ax.scatter(t1['z'], t2['z'], color= 'k', edgecolor='none', alpha=0.5,
               label='All sources')
    ax.scatter(t1['z'][cand], t2['z'][cand], edgecolor='none', alpha=0.7,
               color='k', s=50, label='line candidates*')
    n4,n8,n16,n24 = plot_linelist(ax, t1, 'z', t2['z'], cand, miss=None)

    ax.set_xlim(-0.1, 3.5)
    ax.set_ylim(-0.1, 4)
    ax.set_xlabel('Redshift from %s'%catalog1.replace('_wcs',''), fontsize=15)
    ax.set_ylabel('Redshift from %s'%catalog2.replace('_wcs',''), fontsize=15)
    
    ax.legend(scatterpoints=1, ncol=3,
              title='Line list matched to %s'%catalog1.replace('_wcs',''))
    ax.get_legend().get_title().set_fontsize('20')
    ax.set_title('*Line candidates include those along right edge')
    if show:
        plt.show()
    fig.savefig('redshift_comparison.pdf')



