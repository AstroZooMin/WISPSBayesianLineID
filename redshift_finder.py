import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import itertools
from astropy.coordinates import Distance


def extract_spectrum((Par, target, spectrum_folder, method)):
    '''Returns a 3xN array of observed frame wavelength, flux, and flux error for a given target
    '''
    # ARE DATA FOR BOTH GRISMS PRESENT? IF SO...
    haveG102 = 'Par' + str(Par) + '_G102_BEAM_' + str(target) + 'A.dat' in os.listdir(os.path.join(spectrum_folder, 'Par' + str(Par), 'Spectra'))
    haveG141 = 'Par' + str(Par) + '_G141_BEAM_' + str(target) + 'A.dat' in os.listdir(os.path.join(spectrum_folder, 'Par' + str(Par), 'Spectra'))
    if haveG102 and haveG141:
        # Process G102 first.
        # Load 1D spectral data. The resultant format is an array of row vectors.
        # A transpose is required to obtain the data for all rows of a particular column.
        G102 = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(Par),
                                       'Spectra', 'Par' + str(Par) + '_G102_BEAM_' + str(target) + 'A.dat'))
        # Load the data segment of the FIRST DATA extension of the 2D FITS stamp file
        # FIXME: should really specify `ext='SCI'`, in case ordering of HDUs is not consistent.
        G102_d = fits.getdata(os.path.join(spectrum_folder, 'Par' + str(Par),
                                           'G102_DRIZZLE', 'aXeWFC3_G102_mef_ID' + str(target) + '.fits'))
        # Isolate the middle row of data segment of the PRIMARY extension of the 2D FITS stamp file.
        m = G102_d[G102_d.shape[0] / 2, :]
        # Define a boolean array which is true if the middle row of data segment of the SCI extension exceeds 0
        # FIXME: Is this designed to find things like the deathstar or just empty segments of data?
        # Fluxes could be negative and valid.
        marske = m > 0
        # Extract the 1D spectrum wavelength coordinates
        G102_wl = G102.T[0]
        # Load the header of the first DATA HDU.
        G102_h = fits.open(os.path.join(spectrum_folder, 'Par' + str(Par),
                                        'G102_DRIZZLE', 'aXeWFC3_G102_mef_ID' + str(target) + '.fits'))[1].header
        # Next two lines define an array containing the wavelength coordinates of the
        # pixel boundaries of the first DATA HDU.
        x = np.arange(G102_d.shape[1]) + 1
        wave = G102_h['crval1'] + (x - G102_h['crpix1']) * G102_h['cdelt1']
        # Identify overlap and mismatch between 1D and 2D wavelength sampling
        i, j = 0, 0
        # NOTE: The following algorithm assumes that the 2D pixel wavelength coordinate
        # array always encompasses the 1D spectrum wavelength array. This is probably
        # a reasonable assumption since the latter is derived from the former!
        #
        # Scan forwards in the array of 1D spectrum wavelengths to find the last
        # element (i) whose value is NOT LESS THAN the zeroth element of the 2D pixel
        # boundary wavelength coordinate array.
        # FIXME: j is always zero in the following statement
        while wave[j] > G102_wl[i]:
            i += 1
        # Scan forwards in the array of 2D pixel boundary wavelength coordinates to find
        # the first element (j) whose value is NOT less than the element of the 1D spectrum
        # wavelength array computed in the previous loop.
        while wave[j] < G102_wl[i]:
            j += 1
        # NOTE: Now, wave[0] <= G102_wl[i] <= wave[j]
        # i.e. G102_wl[i] is the FIRST 1D spectrum wavelength NOT less than the
        # MINIMUM pixel boundary wavelength coordinate and wave[j] is the FIRST
        # wavelength in the pixel boundary wavelength coodinate array that exceeds
        # G102_wl[i].

        k, l = -1, -1
        # Scan backwards in the array of 2D pixel boundary wavelength coordinates to find
        # the largest element (-|k|) that DOES NOT EXCEED the last element of 1D spectrum
        # wavelength array.
        # FIXME: l is always -1 in the following statement
        while wave[k] > G102_wl[l]:
            k -= 1
        # Scan backwards in the array of 1D spectrum wavelengths to find the largest element
        # (-|l|) that DOES NOT EXCEED the element of the 2D pixel boundary wavelength coordinate
        # array computed in the previous loop.
        while wave[k] < G102_wl[l]:
            l -= 1
        # NOTE: Now, G102_wl[-|l|] <= wave[-|k|] <= G102_wl[-1]
        # i.e. wave[-|k|] is the LAST pixel boundary wavelength coordinate that
        # DOES NOT EXCEED the MAXIMUM 1D spectrum wavelength coordinate and
        # G102_wl[-|l|] is the LAST 1D spectrum wavelength coordinate that DOES
        # NOT EXCEED wave[-|k|].

        #           0   ?    j(>=0)                 -|k(>=1)|
        # wave[]    |  |||   |                        |
        # =====================================================> WAVELENGTH
        # G102_wl[]        |                  |   |||   |
        #                  i                -|l|   ?   -1

        # i and -|l| defines A range of elements of the 1D spectrum array that are encompassed
        # by the 2D pixel wavelength boundaries

        # j and -|k| define A range of elements of the 2D pixel wavelength boundaries array
        # that completely encompassed by the 1D spectrum wavelength coordinates.

        # Define a boolean index that SELECTS elements of the 1D spectrum that
        # fall within the 2D pixel wavelength boundaries AND DO NOT EXCEED 11355Å
        change = G102_wl[i:l] < 11355
        # Extract the elements of the 1D spectrum wavelength array that fall within
        # the 2D pixel wavelength boundaries AND DO NOT EXCEED 11355Å. Do the same
        # for the flux error and zeroth order arrays.
        G102_wl = G102_wl[i:l][change]
        G102_flux_err = G102.T[2][i:l][change]
        G102_zo = G102.T[4][i:l][change]

        # NOTE: The following two statements appear to assume that the ranges i, -|l|
        # and j, -|k| are of equal length. Moreover, they seem to assume correspondence
        # between the elements in the range, i.e. each element pair's wavelength ranges
        # overlap substantially. If this is guaranteed, then much of the above logic
        # seems superfluous.

        # For the flux and contamination arrays, extract the elements of the 1D spectrum
        # wavelength array that fall within the 2D pixel wavelength boundaries AND DO NOT
        # EXCEED 11355Å, and ALSO set to zero any element for which the corresponding pixel
        # in the middle row of the SCI data segment is not positive.
        G102_flux = (G102.T[1][i:l] * marske[j:k])[change]
        G102_contam = (G102.T[3][i:l] * marske[j:k])[change]

        # NOW DO THE SAME FOR G141 - Almost literal code replication except that
        # the hard wavelength cut at 11355Å is now from below.
        G141 = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(Par),
                                       'Spectra', 'Par' + str(Par) + '_G141_BEAM_' + str(target) + 'A.dat'))
        G141_d = fits.getdata(os.path.join(spectrum_folder, 'Par' + str(Par),
                                           'G141_DRIZZLE', 'aXeWFC3_G141_mef_ID' + str(target) + '.fits'))
        m = G141_d[G141_d.shape[0] / 2, :]
        marske = m > 0
        G141_wl = G141.T[0]
        G141_h = fits.open(os.path.join(spectrum_folder, 'Par' + str(Par),
                                        'G141_DRIZZLE', 'aXeWFC3_G141_mef_ID' + str(target) + '.fits'))[1].header
        x = np.arange(G141_d.shape[1]) + 1
        wave = G141_h['crval1'] + (x - G141_h['crpix1']) * G141_h['cdelt1']
        i, j = 0, 0
        while wave[j] > G141_wl[i]:
            i += 1
        while wave[j] < G141_wl[i]:
            j += 1
        k, l = -1, -1
        while wave[k] > G141_wl[l]:
            k -= 1
        while wave[k] < G141_wl[l]:
            l -= 1
        change = G141_wl[i:l] > 11355
        G141_wl = G141_wl[i:l][change]
        G141_flux = (G141.T[1][i:l] * marske[j:k])[change]
        G141_flux_err = G141.T[2][i:l][change]
        G141_contam = (G141.T[3][i:l] * marske[j:k])[change]
        G141_zo = G141.T[4][i:l][change]

        # AFTER BOTH GRISMS HAVE BEEN EXTRACTED...
        # Plot extracted data
        plt.plot(G102_wl, G102_flux)
        plt.plot(G141_wl, G141_flux)
        plt.ylim(0, 1e-17)
        # Convert the flux arrays to masked_array structures with masks corresponding
        # to zero-valued elements of the original flux arrays.
        G141_flux = np.ma.masked_array(G141_flux, np.array(G141_flux) == 0)
        G102_flux = np.ma.masked_array(G102_flux, np.array(G102_flux) == 0)
    # OTHERWISE, IF ONLY G102 IS PRESENT...
    elif 'Par' + str(Par) + '_G102_BEAM_' + str(target) + '.fits' in os.listdir(os.path.join(spectrum_folder, 'Par' + str(Par), 'Spectra')):
        G102 = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(Par),
                                       'Spectra', 'Par' + str(Par) + '_G102_BEAM_' + str(target) + 'A.dat'))
        G102_d = fits.getdata(os.path.join(spectrum_folder, 'Par' + str(Par),
                                           'G102_DRIZZLE', 'aXeWFC3_G102_mef_ID' + str(target) + '.fits'))
        m = G102_d[G102_d.shape[0] / 2, :]
        marske = m > 0
        G102_wl = G102.T[0]
        G102_h = fits.open(os.path.join(spectrum_folder, 'Par' + str(Par),
                                        'G102_DRIZZLE', 'aXeWFC3_G102_mef_ID' + str(target) + '.fits'))[1].header
        x = np.arange(G102_d.shape[1]) + 1
        wave = G102_h['crval1'] + (x - G102_h['crpix1']) * G102_h['cdelt1']
        i, j = 0, 0
        while wave[j] > G102_wl[i]:
            i += 1
        while wave[j] < G102_wl[i]:
            j += 1
        k, l = -1, -1
        while wave[k] > G102_wl[l]:
            k -= 1
        while wave[k] < G102_wl[l]:
            l -= 1
        G102_wl = G102_wl[i:l]
        G102_flux = G102.T[1][i:l] * marske[j:k]
        G102_flux_err = G102.T[2][i:l]
        G102_contam = G102.T[3][i:l] * marske[j:k]
        G102_zo = G102.T[4][i:l]
        G141_wl = []
        G102_flux = np.ma.masked_array(G102_flux, np.array(G102_flux) == 0)
    # NOW, IF ONLY G141 IS PRESENT
    elif 'Par' + str(Par) + '_G141_BEAM_' + str(target) + '.fits' in os.listdir(os.path.join(spectrum_folder, 'Par' + str(Par), 'Spectra')):
        G141 = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(Par),
                                       'Spectra', 'Par' + str(Par) + '_G141_BEAM_' + str(target) + 'A.dat'))
        G141_d = fits.getdata(os.path.join(spectrum_folder, 'Par' + str(Par),
                                           'G141_DRIZZLE', 'aXeWFC3_G141_mef_ID' + str(target) + '.fits'))
        m = G141_d[G141_d.shape[0] / 2, :]
        marske = m > 0
        G141_wl = G141.T[0]
        G141_h = fits.open(os.path.join(spectrum_folder, 'Par' + str(Par),
                                        'G141_DRIZZLE', 'aXeWFC3_G141_mef_ID' + str(target) + '.fits'))[1].header
        x = np.arange(G141_d.shape[1]) + 1
        wave = G141_h['crval1'] + (x - G141_h['crpix1']) * G141_h['cdelt1']
        i, j = 0, 0
        while wave[j] > G141_wl[i]:
            i += 1
        while wave[j] < G141_wl[i]:
            j += 1
        k, l = -1, -1
        while wave[k] > G141_wl[l]:
            k -= 1
        while wave[k] < G141_wl[l]:
            l -= 1
        G141_wl = G141_wl[i:l]
        G141_flux = G141.T[1][i:l] * marske[j:k]
        G141_flux_err = G141.T[2][i:l]
        G141_contam = G141.T[3][i:l] * marske[j:k]
        G141_zo = G141.T[4][i:l]
        G102_wl = []
        G141_flux = np.ma.masked_array(G102_flux, np.array(G102_flux) == 0)
    # FINALLY, ASSUME THAT ONLY A UNIFIED 1D SPECTRUM IS AVAILABLE
    else:
        G141 = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(Par),
                                       'Spectra', 'Par' + str(Par) + '_BEAM_' + str(target) + 'A.dat'))
        wl = G141.T[0]
        flux = G141.T[1]
        flux_err = G141.T[2]
        contam = G141.T[3]
        zo = G141.T[4]
        # Split the spectrum into assumed G102 and G141 segments based on a cutoff
        # wavelength of 11355Å
        put102 = np.array([lam < 11355 for lam in wl])
        put141 = np.array([lam > 11355 for lam in wl])
        G102_wl = wl[put102]
        G102_flux = flux[put102]
        G102_flux_err = flux_err[put102]
        G102_contam = contam[put102]
        G102_zo = zo[put102]
        G141_wl = wl[put141]
        G141_flux = flux[put141]
        G141_flux_err = flux_err[put141]
        G141_contam = contam[put141]
        G141_zo = zo[put141]
    # Smoothing and clipping
    if len(G141_wl) > 0 and len(G102_wl) > 0:  # if both grisms are present
        # MAINLY SIMPLE COPYING OF DATA EXCEPT WHERE OTHERWISE NOTED...
        g141_lambda = G141_wl
        g141_flux = G141_flux
        g141_flux_error = G141_flux_err
        g141_contam = G141_contam
        # Add additional mask for medium and severe zeroth orders
        g141_mask = G141_zo > 2.5
        g141_flux = np.ma.masked_array(g141_flux, g141_mask)
        g102_lambda = G102_wl
        g102_flux = G102_flux
        g102_flux_error = G102_flux_err
        g102_contam = G102_contam
        # Add additional mask for medium and severe zeroth orders
        g102_mask = G102_zo > 2.5
        g102_flux = np.ma.masked_array(g102_flux, g102_mask)

        # REBIN DATA ONTO A UNIFIED PIXEL GRID CONSISTENT WITH THE LOWEST PIXEL
        # RESOLUTION GRISM (G141)

        # Compute the wavelength increment in Ångström corresponding to 1 pixel for
        # the G141 grism...
        ang_per_pixel_g141 = (
            max(g141_lambda) - min(g141_lambda)) / len(g141_lambda)
        # ... then compute the number of such increments required to span the range
        # of the G102 grism.
        num_of_g102_pixels = (
            max(g102_lambda) - min(g102_lambda)) / ang_per_pixel_g141
        # DIFFERENT REBINNING METHODS. FIRST INTERPOLATION...
        if method == 'interp':
            # create an array of wavelengths that sample the range of the G102 grism in
            # wavelength increments corresponding to 1 pixel for the G141 grism.
            new_g102_lambda = np.linspace(max(
                g102_lambda) - ang_per_pixel_g141 * num_of_g102_pixels, max(g102_lambda), num_of_g102_pixels)
            # Interpolate the G102 flux, flux error and contamination values onto the new
            # wavelength grid.
            new_g102_flux = np.interp(new_g102_lambda, g102_lambda, g102_flux)
            new_g102_flux_error = np.interp(
                new_g102_lambda, g102_lambda, g102_flux_error)
            new_g102_contam = np.interp(
                new_g102_lambda, g102_lambda, g102_contam)
        # NOW, USING THE MEAN OF BIN VALUES FOR BINS WITH COORDINATES NOT MORE THAN HALF
        # THE WAVELENGTH INCREMENT CORRESPONDING TO 1 PIXEL FOR THE G141 GRISM FROM THE
        # EACH WAVELENGTH COORDINATE IN A NEW WAVELENGTH GRID: A.K.A. A TOPHAT FILTER!
        elif method == 'tophat':
            # create an array of wavelengths that sample the range of the G102 grism in
            # wavelength increments corresponding to 1 pixel for the G141 grism.
            new_g102_lambda = np.linspace(max(
                g102_lambda) - ang_per_pixel_g141 * num_of_g102_pixels, max(g102_lambda), num_of_g102_pixels)
            # At each value of the new wavelength grid, compute the mean of elements in the
            # ORIGINAL G102 flux array that correspond to wavelengths not more than half
            # the wavelength increment corresponding to 1 pixel for the G141 grism from that
            # value.
            new_g102_flux = [np.mean(g102_flux[np.array(np.abs(new_g102_lambda[
                                     i] - g102_lambda) < ang_per_pixel_g141 / 2.)]) for i in range(len(new_g102_lambda))]
            # Do the same for the flux error and contamination arrays.
            new_g102_flux_error = [np.mean(g102_flux_error[np.abs(np.array(new_g102_lambda[
                                           i] - g102_lambda) < ang_per_pixel_g141 / 2.)]) for i in range(len(new_g102_lambda))]
            new_g102_contam = [np.mean(g102_contam[np.array(np.abs(new_g102_lambda[
                                       i] - g102_lambda) < ang_per_pixel_g141 / 2.)]) for i in range(len(new_g102_lambda))]
        # NO REBINNING
        else:
            new_g102_lambda = g102_lambda
            new_g102_flux = g102_flux
            new_g102_flux_error = g102_flux_error
            new_g102_contam = g102_contam
        # CREATE UNIFIED SPECTRUM BY CONCATENATING CORRESPONDING G102 AND G141
        # ARRAYS...
        wl = np.array(list(new_g102_lambda) + list(g141_lambda))
        flux = np.array(list(new_g102_flux) + list(g141_flux))
        flux_error = np.array(
            list(new_g102_flux_error) + list(g141_flux_error))
        contam = np.array(list(new_g102_contam) + list(g141_contam))
        # APPLY FINAL HARD WAVELENGTH LIMITS FOR UNIFIED SPECTRUM.
        valid = np.array([l < 16500 and l > 8700 for l in wl])
        wl = wl[valid]
        flux = flux[valid]
        flux_error = flux_error[valid]
        # SET SINGLE NON-ZERO PIXELS THAT ARE BRACKETED BY ZERO-VALUED PIXELS TO
        # ZERO
        for i in range(len(flux) - 2):
            if flux[i] == 0 and flux[i + 2] == 0:
                flux[i + 1] = 0
        #flux_error = np.array([min(flux[i],flux_error[i]) for i in range(len(flux_error))])
        #flux_error = np.array([1e-18 if flux[i] ==0 else flux_error[i] for i in range(len(flux_error))])
        #flux_error = np.array([np.median(flux_error)]*len(flux_error))
        #flux_error = np.array([2*flux_error[i] if wl[i] < 9000 else flux_error[i] for i in range(len(wl))])
        # APPLY FINAL HARD WAVELENGTH LIMITS FOR UNIFIED CONTAMINATION DATA
        contam = contam[valid]
        # CHECK FOR FLUX NORMALIZATION MISMATCH BETWEEN G102 AND G141 SPECTRA
        # Locate the index in the unified spectrum at which the data transition from
        # G102 to G141 i.e. the index with wavelength closest to 11355Å
        change = list(abs(wl - 11355)).index(min(abs(wl - 11355)))
        # Compute the ratio of fluxes at a 2 element separation from the boundary
        # and report if the ratio exceeds a threshold of 1.4
        if flux[change + 2] / flux[change - 2] > 1.4 and flux[change - 2] != 0:
            print "Possible flux mismatch on grisms in Par {0} Target {1}".format(Par, target)
        # if np.median(flux/flux_error)<4:
        #wl = np.linspace(wl[0],wl[-1],len(wl))
        #flux = [np.mean(flux[np.array(np.abs(wl[i]-wl)<ang_per_pixel_g141*.4)]) for i in range(len(wl)) ]
        #flux_error = [np.mean(flux_error[np.array(np.abs(wl[i]-wl)<ang_per_pixel_g141*.4)]) for i in range(len(wl)) ]
    else:  # if only one grism is present
        if len(G102_wl) == 0:
            g102_lambda = G141_wl
            g102_flux = G141_flux
            g102_flux_error = G141_flux_err
            g102_contam = G141_contam
            g102_mask = G141_zo < 2.5
            g102_flux = g102_flux * g102_mask
        else:
            g102_lambda = G102_wl
            g102_flux = G102_flux
            g102_flux_error = G102_flux_err
            g102_contam = G102_contam
            g102_mask = G102_zo < 2.5
            g102_flux = g102_flux * g102_mask
        wl = np.array(list(g102_lambda))
        flux = np.array(list(g102_flux))
        flux_error = np.array(list(g102_flux_error))
        contam = np.array(list(g102_contam))
        valid = np.array([l < 16500 and l > 8700 for l in wl])
        wl = wl[valid]
        flux = flux[valid]
        contam = contam[valid]
        flux_error = flux_error[valid]
        flux_error = np.array([min(1, fl) for fl in flux_error])
        flux_error = np.array([1 if (flux[i] == 0) else flux_error[
                              i] for i in range(len(flux_error))])
    # TRANSFORM THE UNIFIED FLUX ARRAY INTO A MASKED ARRAY WITH A MASK DEFINED BY
    # NaN OR INF VALUES.
    flux = np.ma.masked_invalid(flux)
    return np.array([wl, flux, flux_error, contam])


def fit_spectrum((wl, flux, flux_error, contam, bc03_folder, Par, target, spectrum_folder, ML_ratio, plot, z_max, z_bins, EW_max, EW_bins, return_spectrum, method)):
    '''given an input list of wavelengths and fluxes, finds the best fit BC03 model and returns redshift, Ha EW, and mass.
    Optionally will return the whole spectrum, if return_spectrum is set to "yes"
    Setting return_spectrum to "continuum" will return the continuunm fit, minus any emission lines

    Note -- to fit the continuum only, set EW_bins to 1
    '''
    # COPY INPUT DATA AND COERCE INTO NUMPY ARRAYS
    flux1 = np.array(flux)
    wl1 = np.array(wl)
    # QUESTION: Model spectra to be fitted comprose 168 wavelength values?
    # QUESTION: Assumes that if the number of elements are correct then the data
    # and model spectra must have been prepared to correspond in wavelength space?
    if len(wl1) != 168:
        print "mismatch in wavelength space"
        return -99, 'None', np.zeros(331), -99
    contam1 = np.array(contam)
    # DEFINE ARRAY OF TRIAL REDSHIFT VALUES
    z_array = np.linspace(0, z_max, z_bins)
    # Simple reassignment - no masking required.
    wl = wl1
    # Mask flux array if estimated contaminating flux exceeds 50% of the measured
    # total flux or the value of the masked element is NaN or Inf.
    flux = np.ma.masked_array(flux, contam1 / flux1 > .5)
    flux = np.ma.masked_invalid(flux)
    # Define the minimum "believable" point error as 1/6 of the median of
    # valid flux values.
    min_error = np.ma.median(flux) / 6
    # Replace any error values for which the correspondin flux value is NaN
    # with a unit value that will result in an unweighted contribution to the
    # chi-square computation.
    flux_error = np.array([1 if np.isnan(flux[i]) else flux_error[
                          i] for i in range(len(flux_error))])
    # Replace any "unbelieveably" low error meaurements with the minimum "believable"
    # value.
    flux_error = np.array([max(flux_error[i], min_error)
                           for i in range(len(flux))])

    # LOAD PRECOMPUTED GRID OF RAW AND NORMALIZED MODEL SPECTRA
    files_in_folder = os.listdir(bc03_folder)
    spectra_files = np.load(os.path.join(bc03_folder, 'spectra_files.npy'))
    spec = []
    for i in range(len(spectra_files)):
        spec.append(np.loadtxt(os.path.join(
            bc03_folder, spectra_files[i]), skiprows=4))
    models = np.array(np.load(os.path.join(bc03_folder, 'models_normed.npy')))
    models_med = np.array(np.load(os.path.join(bc03_folder, 'models_med.npy')))
    # Make GIGANTIC arrays with a copy of the data flux and flux error array
    # for every model spectrum for every trial redshift!
    ff = np.array([[flux for j in range(len(spec))] for z in z_array])
    fe = np.array([[flux_error for j in range(len(spec))] for z in z_array])
    # Create an array containing the a copy of the value of the median of
    # valid values of the input flux array for every model spectrum for every
    # trial redshift!
    median_flux = np.ma.median(np.ma.masked_invalid(ff[0][0]))
    ff_n = np.full_like(a=ff[:, :, 0], fill_value=median_flux)
    # Create an array containing the a copy of the median-normalized data flux
    # array for every model spectrum for every trial redshift!
    normed_flux_array = ff[0][0] / median_flux
    flux_normed = np.full_like(a=ff[:, :, 0], fill_value=normed_flux_array)
    # Create an array containing the a copy of the data flux ERROR array normalized
    # by the median of the data FLUX array for every model spectrum for every trial
    # redshift!
    normed_flux_err_array = fe[0][0] / ff_n[0][0]
    flux_err_normed = np.full_like(a=ff[:, :, 0], fill_value=normed_flux_err_array)
    # COMPUTE THE LOG-LIKELIHOOD FOR ALL MODELS GIVEN THE INPUT DATA.
    # Compute the element-wise contribution to the log likelihood (-0.5*chi-square)
    g = -((flux_normed - np.array(models))**2) / (2 * flux_err_normed**2)
    # Mask any NaN or Inf values before searching for maximum likelihood.
    gma = np.ma.masked_invalid(g)
    # Remove contributions from contaminated data elements and sum those remaining
    # for every model for every trial redshift.
    hh = np.array([[np.ma.sum(np.ma.masked_array(gma[z][j], contam1 / flux1 > .5))
                    for j in range(len(spec))] for z in range(len(z_array))])
    # The probability is optained by exponentiating (-0.5*chi-square)
    # i.e. chi-square = -2ln(P)
    probs = np.exp(hh)
    # Locate the two dimensional coordinates of the maximum probability element
    # over all model spectra and model redshifts
    a, b = np.unravel_index(np.argmax(probs), probs.shape)
    # COMPUTE THE MARGINAL PROBABILITY OVER ALL MODELS FOR EACH TRIAL REDSHIFTS
    # Begin by simply projecting onto the trial redshift axis
    z_pdf = np.sum(probs, axis=1)
    # Integrate the resulting projection using the trapezium rule
    noram = np.trapz(z_array, z_pdf)
    # Normalize the projected array to have unit integral.
    z_pdf = np.abs(z_pdf / noram)
    # If the computed PDF contains any NaN values then abandon redshift constraint
    # and return values to indicate failure.
    if np.isnan(z_pdf).any():
        print "NaN present in pdf"
        return -99, 'None', np.zeros(331), np.ma.median([models_med[a][b] / ff_n[a][b]])
    # Locate the index of the element in the trial redshift array that corresponds to the
    # maximum value in the redshift probability density function.
    zed = z_array[list(z_pdf).index(max(z_pdf))]
    print target, zed
    # RETURN:
    # 1: The index of the highest probabilty redshift
    # 2: The best fitting model file name
    # 3: The marginal redshift PDF (can be used with the first return value to get the
    # probability density for the highest marginal probability redshift.)
    # 4: Applied model normalization factor.
    return zed, spectra_files[b], z_pdf, np.ma.median([models_med[a][b] / ff_n[a][b]])
