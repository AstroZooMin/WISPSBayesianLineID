import os
import numpy as np
import matplotlib.pyplot as plt

from get_config import *

def get_waves(x,hdr):
    return (x - hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']

def calc_mag( flux, filt_name ):

    flux = 10**flux
    c = 3e18
    pivot_l = {'F110':11534.00, 'F140':13922.979, 'F160':15369.161}

    filt_waves, filt_sens = np.genfromtxt('config/HST_WFC3_IR.%sW.dat'%filt_name,unpack=True)
    pivot = pivot_l[filt_name]
    flux = scipy.integrate.simps(flux*filt_sens*filt_waves, filt_waves) / scipy.integrate.simps(filt_sens*filt_waves, filt_waves)
    flux = (pivot**2/c) * flux
    mag = -2.5*np.log10(flux) - 48.6
    return mag

def plot_1d(par,obj):

    zp = {'F110':26.8223,'F160':25.9463}
    conf = {'G102':get_config_G102_F110(),'G141':get_config_G141_F160()}
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.08,right=0.95,bottom=0.07,top=0.95,wspace=0,hspace=0)
    
    spec = np.genfromtxt(os.path.join(parent_dir,"Par%i/Spectra/Par%i_BEAM_%iA.dat"%(par,par,obj)))
    ax1.plot(spec[:,0],spec[:,1],c='k',lw=1.5)
    
    rsp_G102 = conf['G102']['SENS_A']
    grs_G102 = fitsio.open(os.path.join(parent_dir,'Par%i/G102_DRIZZLE/aXeWFC3_G102_mef_ID%i.fits'%(par,obj)))
    hdr_G102 = grs_G102['SCI'].header
    sci_G102 = np.ma.masked_array(grs_G102['SCI'].data,mask=(~np.isfinite(grs_G102['SCI'].data)))[:,1:]
    wht_G102 = np.ma.masked_array(grs_G102['WHT'].data,mask=(grs_G102['WHT'].data==0))[:,1:]
    spc_G102 = np.genfromtxt(os.path.join(parent_dir,"Par%i/Spectra/Par%i_G102_BEAM_%iA.dat"%(par,par,obj)))
    cat_G102 = np.genfromtxt(os.path.join(parent_dir,"Par%i/DATA/DIRECT_GRISM/fin_F110.cat"%par),
                    dtype=[('ID',int),('X',float),('Y',float),('A',float),('B',float),('THETA',float),('MAG',float)],usecols=[1,2,3,4,5,6,12])
    ent_G102 = cat_G102[cat_G102['ID']==obj][0]
    wav_G102 = get_waves(np.arange(sci_G102.shape[1])+2,hdr_G102)
    flx_G102 = np.ma.sum(sci_G102*wht_G102,axis=0)/np.ma.sum(wht_G102,axis=0)/rsp_G102(wav_G102)
    ftr_G102 = np.ma.sum(wht_G102,axis=0) / 24.
    flx_G102 = flx_G102 * ftr_G102
    cnd_G102 = (spec[0,0] <= wav_G102) & (wav_G102 < 11500)
    wav_G102,flx_G102,spc_G102 = wav_G102[cnd_G102],flx_G102[cnd_G102],spc_G102[cnd_G102]

    flx_F110 = 10**scipy.optimize.brentq(lambda x: calc_mag(x,'F110') - ent_G102['MAG'],-20,-12)
    ax1.hlines(flx_F110,8000,11500,color='b',lw=1)

    #ax1.plot(spc_G102[:,0],spc_G102[:,1],c='k',lw=1.5)
    ax1.plot(wav_G102,flx_G102,c='b',lw=1.5)
    ax2.plot(wav_G102,flx_G102/spc_G102[:,1],c='b',lw=2)
    ax2.axhline(np.median(flx_G102/spc_G102[:,1]),c='b',lw=2,ls='--',label="%.2f"%np.median(flx_G102/spc_G102[:,1]))

    rsp_G141 = conf['G141']['SENS_A']
    grs_G141 = fitsio.open(os.path.join(parent_dir,'Par%i/G141_DRIZZLE/aXeWFC3_G141_mef_ID%i.fits'%(par,obj)))
    hdr_G141 = grs_G141['SCI'].header
    sci_G141 = np.ma.masked_array(grs_G141['SCI'].data,mask=(~np.isfinite(grs_G141['SCI'].data)))[:,1:]
    wht_G141 = np.ma.masked_array(grs_G141['WHT'].data,mask=(grs_G141['WHT'].data==0))[:,1:]
    spc_G141 = np.genfromtxt(os.path.join(parent_dir,"Par%i/Spectra/Par%i_G141_BEAM_%iA.dat"%(par,par,obj)))
    cat_G141 = np.genfromtxt(os.path.join(parent_dir,"Par%i/DATA/DIRECT_GRISM/fin_F160.cat"%par),
                    dtype=[('ID',int),('X',float),('Y',float),('A',float),('B',float),('THETA',float),('MAG',float)],usecols=[1,2,3,4,5,6,12])
    ent_G141 = cat_G141[cat_G141['ID']==obj][0]
    wav_G141 = get_waves(np.arange(sci_G141.shape[1])+2,hdr_G141)
    flx_G141 = np.ma.sum(sci_G141*wht_G141,axis=0)/np.ma.sum(wht_G141,axis=0)/rsp_G141(wav_G141)
    ftr_G141 = np.ma.sum(wht_G141,axis=0) / 46.5
    flx_G141 = flx_G141 * ftr_G141
    cnd_G141 = (11000 < wav_G141) & (wav_G141 < spec[-1,0])
    wav_G141,flx_G141,spc_G141 = wav_G141[cnd_G141],flx_G141[cnd_G141],spc_G141[cnd_G141]

    flx_F160 = 10**scipy.optimize.brentq(lambda x: calc_mag(x,'F160') - ent_G141['MAG'],-20,-12)
    ax1.hlines(flx_F160,11500,17000,color='r',lw=1)

    #ax1.plot(spc_G141[:,0],spc_G141[:,1],c='k',lw=1.5)
    ax1.plot(wav_G141,flx_G141,c='r',lw=1.5)
    ax2.plot(wav_G141,flx_G141/spc_G141[:,1],c='r',lw=2)
    ax2.axhline(np.median(flx_G141/spc_G141[:,1]),c='r',lw=2,ls='--',label="%.2f"%np.median(flx_G141/spc_G141[:,1]))

    ax1.set_ylabel('Flux')
    ax2.set_ylabel('Manual/aXe')
    ax2.set_xlabel('Observed Wavelength')
    ax2.legend(loc='best')

if __name__ == '__main__':
    
    parent_dir = "/data/highzgal/PUBLICACCESS/WISPS/data/V6.2/"
    plot_1d(96,13)
    plt.show()