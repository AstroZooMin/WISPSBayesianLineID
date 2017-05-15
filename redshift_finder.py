import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import itertools
from astropy.coordinates import Distance

def extract_spectrum((Par,target,spectrum_folder,method)):
    '''Returns a 3xN array of observed frame wavelength, flux, and flux error for a given target
    '''
    if 'Par'+str(Par)+'_G102_BEAM_'+str(target)+'A.dat' in os.listdir(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra')) and 'Par'+str(Par)+'_G141_BEAM_'+str(target)+'A.dat' in os.listdir(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra')):
        G102 = np.loadtxt(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra','Par'+str(Par)+'_G102_BEAM_'+str(target)+'A.dat'))
        G102_d = fits.getdata(os.path.join(spectrum_folder,'Par'+str(Par),'G102_DRIZZLE','aXeWFC3_G102_mef_ID'+str(target)+'.fits'))
        m = G102_d[G102_d.shape[0]/2,:]
        marske = m>0
        G102_wl = G102.T[0]
        G102_h = fits.open(os.path.join(spectrum_folder,'Par'+str(Par),'G102_DRIZZLE','aXeWFC3_G102_mef_ID'+str(target)+'.fits'))[1].header
        x = np.arange(G102_d.shape[1]) + 1
        wave = G102_h['crval1'] + (x-G102_h['crpix1']) * G102_h['cdelt1']
        i,j = 0,0
        while wave[j] > G102_wl[i]:
            i+=1
        while wave[j] < G102_wl[i]:
            j+=1
        k,l = -1,-1
        while wave[k] > G102_wl[l]:
            k-=1
        while wave[k] < G102_wl[l]:
            l-=1
        change = G102_wl[i:l]<11355
        G102_wl = G102_wl[i:l][change]
        G102_flux = (G102.T[1][i:l]*marske[j:k])[change]
        G102_flux_err = G102.T[2][i:l][change]
        G102_contam = (G102.T[3][i:l]*marske[j:k])[change]
        G102_zo = G102.T[4][i:l][change]
        G141 = np.loadtxt(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra','Par'+str(Par)+'_G141_BEAM_'+str(target)+'A.dat'))
        G141_d = fits.getdata(os.path.join(spectrum_folder,'Par'+str(Par),'G141_DRIZZLE','aXeWFC3_G141_mef_ID'+str(target)+'.fits'))
        m = G141_d[G141_d.shape[0]/2,:]
        marske = m>0
        G141_wl = G141.T[0]
        G141_h = fits.open(os.path.join(spectrum_folder,'Par'+str(Par),'G141_DRIZZLE','aXeWFC3_G141_mef_ID'+str(target)+'.fits'))[1].header
        x = np.arange(G141_d.shape[1]) + 1
        wave = G141_h['crval1'] + (x-G141_h['crpix1']) * G141_h['cdelt1']
        i,j = 0,0
        while wave[j] > G141_wl[i]:
            i+=1
        while wave[j] < G141_wl[i]:
            j+=1
        k,l = -1,-1
        while wave[k] > G141_wl[l]:
            k-=1
        while wave[k] < G141_wl[l]:
            l-=1
        change = G141_wl[i:l]>11355
        G141_wl = G141_wl[i:l][change]
        G141_flux = (G141.T[1][i:l]*marske[j:k])[change]
        G141_flux_err = G141.T[2][i:l][change]
        G141_contam = (G141.T[3][i:l]*marske[j:k])[change]
        G141_zo = G141.T[4][i:l][change]
        plt.plot(G102_wl,G102_flux)
        plt.plot(G141_wl,G141_flux)
        plt.ylim(0,1e-17)
        G141_flux = np.ma.masked_array(G141_flux,np.array(G141_flux) == 0)
        G102_flux = np.ma.masked_array(G102_flux,np.array(G102_flux) == 0)
    elif 'Par'+str(Par)+'_G102_BEAM_'+str(target)+'.fits' in os.listdir(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra')):
        G102 = np.loadtxt(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra','Par'+str(Par)+'_G102_BEAM_'+str(target)+'A.dat'))
        G102_d = fits.getdata(os.path.join(spectrum_folder,'Par'+str(Par),'G102_DRIZZLE','aXeWFC3_G102_mef_ID'+str(target)+'.fits'))
        m = G102_d[G102_d.shape[0]/2,:]
        marske = m>0
        G102_wl = G102.T[0]
        G102_h = fits.open(os.path.join(spectrum_folder,'Par'+str(Par),'G102_DRIZZLE','aXeWFC3_G102_mef_ID'+str(target)+'.fits'))[1].header
        x = np.arange(G102_d.shape[1]) + 1
        wave = G102_h['crval1'] + (x-G102_h['crpix1']) * G102_h['cdelt1']
        i,j = 0,0
        while wave[j] > G102_wl[i]:
            i+=1
        while wave[j] < G102_wl[i]:
            j+=1
        k,l = -1,-1
        while wave[k] > G102_wl[l]:
            k-=1
        while wave[k] < G102_wl[l]:
            l-=1
        G102_wl = G102_wl[i:l]
        G102_flux = G102.T[1][i:l]*marske[j:k]
        G102_flux_err = G102.T[2][i:l]
        G102_contam = G102.T[3][i:l]*marske[j:k]
        G102_zo = G102.T[4][i:l]
        G141_wl=[]
        G102_flux = np.ma.masked_array(G102_flux,np.array(G102_flux) == 0)
    elif 'Par'+str(Par)+'_G141_BEAM_'+str(target)+'.fits' in os.listdir(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra')):
        G141 = np.loadtxt(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra','Par'+str(Par)+'_G141_BEAM_'+str(target)+'A.dat'))
        G141_d = fits.getdata(os.path.join(spectrum_folder,'Par'+str(Par),'G141_DRIZZLE','aXeWFC3_G141_mef_ID'+str(target)+'.fits'))
        m = G141_d[G141_d.shape[0]/2,:]
        marske = m>0
        G141_wl = G141.T[0]
        G141_h = fits.open(os.path.join(spectrum_folder,'Par'+str(Par),'G141_DRIZZLE','aXeWFC3_G141_mef_ID'+str(target)+'.fits'))[1].header
        x = np.arange(G141_d.shape[1]) + 1
        wave = G141_h['crval1'] + (x-G141_h['crpix1']) * G141_h['cdelt1']
        i,j = 0,0
        while wave[j] > G141_wl[i]:
            i+=1
        while wave[j] < G141_wl[i]:
            j+=1
        k,l = -1,-1
        while wave[k] > G141_wl[l]:
            k-=1
        while wave[k] < G141_wl[l]:
            l-=1
        G141_wl = G141_wl[i:l]
        G141_flux = G141.T[1][i:l]*marske[j:k]
        G141_flux_err = G141.T[2][i:l]
        G141_contam = G141.T[3][i:l]*marske[j:k]
        G141_zo = G141.T[4][i:l]
        G102_wl = []
        G141_flux = np.ma.masked_array(G102_flux,np.array(G102_flux) == 0)
    else:
        G141 = np.loadtxt(os.path.join(spectrum_folder,'Par'+str(Par),'Spectra','Par'+str(Par)+'_BEAM_'+str(target)+'A.dat'))
        wl = G141.T[0]
        flux = G141.T[1]
        flux_err = G141.T[2]
        contam = G141.T[3]
        zo = G141.T[4]
        put102 = np.array([lam < 11355 for lam in wl])
        put141 = np.array([lam > 11355 for lam in wl])
        G102_wl = wl[put102]
        G102_flux = flux[put102]
        G102_flux_err= flux_err[put102]
        G102_contam = contam[put102]
        G102_zo = zo[put102]
        G141_wl = wl[put141]
        G141_flux = flux[put141]
        G141_flux_err= flux_err[put141]
        G141_contam = contam[put141]
        G141_zo = zo[put141]
    #Smoothing and clipping
    if len(G141_wl) > 0  and len(G102_wl) > 0:  #if both grisms are present
        g141_lambda = G141_wl
        g141_flux = G141_flux
        g141_flux_error = G141_flux_err
        g141_contam = G141_contam
        g141_mask = G141_zo > 2.5
        g141_flux = np.ma.masked_array(g141_flux,g141_mask)
        g102_lambda = G102_wl
        g102_flux = G102_flux
        g102_flux_error = G102_flux_err
        g102_contam = G102_contam
        g102_mask = G102_zo > 2.5
        g102_flux = np.ma.masked_array(g102_flux,g102_mask)
        ang_per_pixel_g141=(max(g141_lambda)-min(g141_lambda))/len(g141_lambda)
        num_of_g102_pixels=(max(g102_lambda)-min(g102_lambda))/ang_per_pixel_g141
        if method == 'interp':
            new_g102_lambda = np.linspace(max(g102_lambda)-ang_per_pixel_g141*num_of_g102_pixels,max(g102_lambda),num_of_g102_pixels)
            new_g102_flux = np.interp(new_g102_lambda,g102_lambda,g102_flux)
            new_g102_flux_error = np.interp(new_g102_lambda,g102_lambda,g102_flux_error)
            new_g102_contam = np.interp(new_g102_lambda,g102_lambda,g102_contam)
        elif method == 'tophat':
            new_g102_lambda = np.linspace(max(g102_lambda)-ang_per_pixel_g141*num_of_g102_pixels,max(g102_lambda),num_of_g102_pixels)
            new_g102_flux = [np.mean(g102_flux[np.array(np.abs(new_g102_lambda[i]-g102_lambda)<ang_per_pixel_g141/2.)]) for i in range(len(new_g102_lambda)) ]
            new_g102_flux_error = [np.mean(g102_flux_error[np.abs(np.array(new_g102_lambda[i]-g102_lambda)<ang_per_pixel_g141/2.)]) for i in range(len(new_g102_lambda)) ]
            new_g102_contam = [np.mean(g102_contam[np.array(np.abs(new_g102_lambda[i]-g102_lambda)<ang_per_pixel_g141/2.)]) for i in range(len(new_g102_lambda)) ]
        else:
            new_g102_lambda = g102_lambda
            new_g102_flux = g102_flux
            new_g102_flux_error = g102_flux_error
            new_g102_contam = g102_contam
        wl = np.array(list(new_g102_lambda)+list(g141_lambda))
        flux = np.array(list(new_g102_flux)+list(g141_flux))
        flux_error = np.array(list(new_g102_flux_error)+list(g141_flux_error))
        contam = np.array(list(new_g102_contam)+list(g141_contam))
        valid = np.array([l<16500 and l>8700 for l in wl])
        wl = wl[valid]
        flux = flux[valid]
        flux_error=flux_error[valid]
        for i in range(len(flux)-2):
            if flux[i] == 0 and flux[i+2] == 0:
                flux[i+1] =0
        #flux_error = np.array([min(flux[i],flux_error[i]) for i in range(len(flux_error))])
        #flux_error = np.array([1e-18 if flux[i] ==0 else flux_error[i] for i in range(len(flux_error))])
        #flux_error = np.array([np.median(flux_error)]*len(flux_error))
        #flux_error = np.array([2*flux_error[i] if wl[i] < 9000 else flux_error[i] for i in range(len(wl))])
        contam = contam[valid]
        change = list(abs(wl-11355)).index(min(abs(wl-11355)))
        if flux[change+2]/flux[change-2] >1.4 and flux[change-2]!=0:
            print "Possible flux mismatch on grisms in Par {0} Target {1}".format(Par,target)
        #if np.median(flux/flux_error)<4:
        #wl = np.linspace(wl[0],wl[-1],len(wl))
        #flux = [np.mean(flux[np.array(np.abs(wl[i]-wl)<ang_per_pixel_g141*.4)]) for i in range(len(wl)) ]
        #flux_error = [np.mean(flux_error[np.array(np.abs(wl[i]-wl)<ang_per_pixel_g141*.4)]) for i in range(len(wl)) ]
    else: #if only one grism is present
        if len(G102_wl) ==0:
            g102_lambda = G141_wl
            g102_flux = G141_flux
            g102_flux_error = G141_flux_err
            g102_contam = G141_contam
            g102_mask = G141_zo < 2.5
            g102_flux = g102_flux*g102_mask
        else:
            g102_lambda = G102_wl
            g102_flux = G102_flux
            g102_flux_error = G102_flux_err
            g102_contam = G102_contam
            g102_mask = G102_zo < 2.5
            g102_flux = g102_flux*g102_mask
        wl = np.array(list(g102_lambda))
        flux = np.array(list(g102_flux))
        flux_error = np.array(list(g102_flux_error))
        contam = np.array(list(g102_contam))
        valid = np.array([l<16500 and l>8700 for l in wl])
        wl = wl[valid]
        flux = flux[valid]
        contam = contam[valid]
        flux_error=flux_error[valid]
        flux_error = np.array([min(1,fl) for fl in flux_error])
        flux_error = np.array([1 if (flux[i] ==0) else flux_error[i] for i in range(len(flux_error))])
    flux = np.ma.masked_invalid(flux)
    return np.array([wl,flux,flux_error,contam])

def fit_spectrum((wl,flux,flux_error,contam,bc03_folder,Par,target,spectrum_folder,ML_ratio,plot,z_max,z_bins,EW_max,EW_bins,return_spectrum,method)):
    '''given an input list of wavelengths and fluxes, finds the best fit BC03 model and returns redshift, Ha EW, and mass.
    Optionally will return the whole spectrum, if return_spectrum is set to "yes"
    Setting return_spectrum to "continuum" will return the continuunm fit, minus any emission lines

    Note -- to fit the continuum only, set EW_bins to 1
    '''
    flux1 = np.array(flux)
    wl1 = np.array(wl)
    if len(wl1) != 168:
        print "mismatch in wavelength space"
        return -99,'None',np.zeros(331),-99
    contam1 = np.array(contam)
    z_array = np.linspace(0,z_max,z_bins)
    wl  = wl1
    flux = np.ma.masked_array(flux,contam1/flux1 > .5)
    flux = np.ma.masked_invalid(flux)
    min_error = np.ma.median(flux)/6
    flux_error = np.array([1 if np.isnan(flux[i]) else flux_error[i] for i in range(len(flux_error))])
    flux_error = np.array([max(flux_error[i],min_error) for i in range(len(flux))])
    files_in_folder = os.listdir(bc03_folder)
    spectra_files = np.load(os.path.join(bc03_folder,'spectra_files.npy'))
    spec = []
    for i in range(len(spectra_files)):
        spec.append(np.loadtxt(os.path.join(bc03_folder,spectra_files[i]),skiprows = 4))
    models = np.array(np.load(os.path.join(bc03_folder,'models_normed.npy')))
    models_med = np.array(np.load(os.path.join(bc03_folder,'models_med.npy')))
    ff = np.array([[flux for j in range(len(spec))] for z in z_array ])
    fe = np.array([[flux_error  for j in range(len(spec))] for z in z_array ])
    ff_n = np.array([[np.ma.median(np.ma.masked_invalid(ff[z][j])) for j in range(len(spec))] for z in range(len(z_array))])
    flux_normed = np.array([[ff[z][j]/ff_n[z][j] for j in range(len(spec))] for z in range(len(z_array))])
    flux_err_normed = np.array([[fe[z][j]/ff_n[z][j] for j in range(len(spec))] for z in range(len(z_array))])
    g = -((flux_normed-np.array(models))**2)/(2*flux_err_normed**2)
    gma = np.ma.masked_invalid(g)
    hh = np.array([[np.ma.sum(np.ma.masked_array(gma[z][j],contam1/flux1 > .5)) for j in range(len(spec))] for z in range(len(z_array))])
    probs = np.exp(hh)
    a,b = np.unravel_index(np.argmax(probs),probs.shape)
    z_pdf = np.sum(probs,axis = 1)
    noram = np.trapz(z_array,z_pdf)
    z_pdf = np.abs(z_pdf/noram)
    if np.isnan(z_pdf).any():
        print "NaN present in pdf"
        return -99,'None',np.zeros(331),np.ma.median([models_med[a][b]/ff_n[a][b]])
    zed = z_array[list(z_pdf).index(max(z_pdf))]
    print target,zed
    return zed,spectra_files[b],z_pdf,np.ma.median([models_med[a][b]/ff_n[a][b]])
