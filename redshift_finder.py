import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.io import fits
import itertools
from astropy.coordinates import Distance


def extract_spectrum(Par, beam, spectrum_folder):
    '''Returns a 3xN array of observed frame wavelength, flux, and flux error for a given target
    '''
    try:
        real_spectrum = np.loadtxt(os.path.join(spectrum_folder, 'Par' + str(
            Par) + '/Spectra/Par' + str(Par) + '_BEAM_' + str(beam) + 'A.dat'))
    except IOError:
        print "Par {0} Target {1}: File does not exist".format(Par, beam)
        return [[-99], [-99], [-99]]
    # Smoothing and clipping
    if len(real_spectrum.T[0]) > 150:  # if both grisms are present
        g141_lambda = real_spectrum.T[0][134:]
        g141_flux = real_spectrum.T[1][134:]
        g141_flux_error = real_spectrum.T[2][134:]
        g102_lambda = real_spectrum.T[0][:134]
        g102_flux = real_spectrum.T[1][:134]
        g102_flux_error = real_spectrum.T[2][:134]
        # Resample (downsample) g102 spectrum onto g141 pixel scale.
        ang_per_pixel_g141 = (real_spectrum.T[0][
                              134:][-1] - real_spectrum.T[0][134:][0]) / len(real_spectrum.T[0][134:])
        num_of_g102_pixels = (
            real_spectrum.T[0][134] - real_spectrum.T[0][0]) / ang_per_pixel_g141
        new_g102_lambda = np.linspace(real_spectrum.T[0][0], real_spectrum.T[
                                      0][134], num_of_g102_pixels)
        new_g102_flux = np.interp(new_g102_lambda, g102_lambda, g102_flux)
        # FIXME: Is this approach correct? Fluxes are being combined so can the
        # error be simply interpolated like this?
        new_g102_flux_error = np.interp(
            new_g102_lambda, g102_lambda, g102_flux_error)
        # Combine downsampled g102 spectrum with original g141 spectrum
        wl = np.array(list(new_g102_lambda) + list(g141_lambda))
        flux = np.array(list(new_g102_flux) + list(g141_flux))
        flux_error = np.array(
            list(new_g102_flux_error) + list(g141_flux_error))
        # define valid wavelength range...
        valid = np.array([l < 16500 and l > 8700 for l in wl])
        # ...and discard invalid data.
        wl = wl[valid]
        flux = flux[valid]
        flux_error = flux_error[valid]
        # Clamp spurious negative fluxes at zero and define the minimum
        # possible error to be 10^{-5}
        flux = np.array([max(0, fl) for fl in flux])
        # FIXME: Is this a relative error? If not does it make sense to define
        # a minimum absolute error?
        flux_error = np.array([min(1e5, fl) for fl in flux_error])
        # check for comparable fluxes on either side of the grism wavelength
        # boundary
        change = list(abs(wl - 11355)).index(min(abs(wl - 11355)))
        if flux[change + 2] / flux[change - 2] > 1.4 and flux[change - 2] != 0:
            print "Possible flux mismatch on grisms in Par {0} Target {1}".format(Par, beam)
    else:  # if only one grism is present
        g102_lambda = real_spectrum.T[0][:134]
        g102_flux = real_spectrum.T[1][:134]
        g102_flux_error = real_spectrum.T[2][:134]
        wl = np.array(list(g102_lambda))
        flux = np.array(list(g102_flux))
        flux_error = np.array(list(g102_flux_error))
        valid = np.array([l < 16500 and l > 8700 for l in wl])
        wl = wl[valid]
        flux = flux[valid]
        flux_error = flux_error[valid]
        flux = np.array([max(0, fl) for fl in flux])
        flux_error = np.array([min(1e5, fl) for fl in flux_error])
    return np.array([wl, flux, flux_error])


def fit_spectrum(wl, flux, flux_error, bc03_folder, ML_ratio=1, plot=False, z_max=3, z_bins=101, EW_max=120, EW_bins=4, return_spectrum=False):
    '''given an input list of wavelengths and fluxes, finds the best fit BC03 model and returns redshift, Ha EW, and mass.
    Optionally will return the whole spectrum.

    Note -- to fit the continuum only, set EW_bins to 1
    '''
    wl = np.array(wl)
    flux = np.array(flux)
    flux_error = np.array(flux_error)
    files_in_folder = os.listdir(bc03_folder)
    # Load B&C model spectra
    spectra_files = []
    for i in files_in_folder:
        if i[-4:] == 'spec':
            spectra_files.append(i)
    spec = []
    for i in range(len(spectra_files)):
        spec.append(np.loadtxt(os.path.join(
            bc03_folder, spectra_files[i]), skiprows=4))
    # define a range of potential line equivalent widths...
    EW_options = np.linspace(0, EW_max, EW_bins)
    # ... and a range of trial redshifts
    z_array = np.linspace(0, z_max, z_bins)
    # to contain lists of chi square values for various line strength
    # permutations for all redshift options
    chis = []
    # loop over trial redshifts
    for z in z_array:
        # to contain chi square values for various line strength permutations
        chi_active = []
        # transform data wavelength grid from observed to rest frame wavelength
        # i.e. make bluer
        wl_shifted = wl / (1 + z)
        for j in range(len(spec)):
            # resample model onto transformed data wavelength grid
            model_flux = np.interp(wl_shifted, spec[j].T[0], spec[j].T[1])
            # reset model flux to zero if data flux is zero
            model_flux = np.array([0 if flux[i] == 0 else model_flux[
                                  i] for i in range(len(flux))])
            # if the Ha line falls within the rest frame wavelength range...
            if wl_shifted[0] < 6463 and wl_shifted[-1] > 6663:
                # sample continuum in a 100 Å range bracketing the line
                # position
                continuum = np.median(
                    model_flux[np.abs(wl_shifted - 6563 < 100)])
                # define the line as a gaussian with sigma = 100 Å and unit
                # equivalent width
                Ha_line = continuum / \
                    (100 * np.sqrt(2 * np.pi)) * \
                    np.exp(-((wl_shifted - 6563) / (2 * 100))**2)
            else:  # array of zeros (FIXME: could use np.zeros_like())
                Ha_line = 0 * wl_shifted
            # Repeat process for 03 line
            if wl_shifted[0] < 4700 and wl_shifted[-1] > 5175:
                continuum = np.median(
                    model_flux[np.abs(wl_shifted - 4935 < 175)])
                O3_line = continuum / \
                    (175 * np.sqrt(2 * np.pi)) * \
                    np.exp(-((wl_shifted - 4935) / (2 * 175))**2)
            else:
                O3_line = wl_shifted * 0
            # Repeat process for 02 line
            if wl_shifted[0] < 3600 and wl_shifted[-1] > 3900:
                continuum = np.median(
                    model_flux[np.abs(wl_shifted - 3725 < 65)])
                O2_line = continuum / \
                    (65 * np.sqrt(2 * np.pi)) * \
                    np.exp(-((wl_shifted - 3725) / (2 * 65))**2)
            else:
                O2_line = wl_shifted * 0
            # Create a grid with all possible line equivalent widths assigned to each of the three potential lines
            # and loop over rows...
            for Ha_strength, O2_strength, O3_strength in itertools.product(EW_options, repeat=3):
                # Generate model as sum of line fluxes and continuum flux
                model_flux_lines = Ha_strength * Ha_line + O2_strength * \
                    O2_line + O3_strength * O3_line + model_flux
                # rescale model to have median flux equal to the data
                factor = np.median(model_flux_lines) / np.median(flux)
                model_flux_normed = model_flux_lines / factor
                # compute chi square for current model (i.e. line strength
                # permutation) w.r.t. data
                chisq = sum(((flux - model_flux_normed) / flux_error)**2)
                chi_active.append(chisq)
        chis.append(chi_active)
    min_chisq = min(np.array(chis).flatten())
    for j in range(len(chis)):
        # FIXME: Assumes minimum is unique. list.index() selects only the first
        # occurence of the minimum
        if min_chisq in chis[j]:
            jsave, lsave = j, list(chis[j]).index(min_chisq)
    try:
        # define best fit redshift
        z = z_array[jsave]
    except UnboundLocalError:
        # FIXME: Is this possible? The minimum must be in one of the elements of chis and the length of
        # chis should exactly equal that of z_array, since a new element is appended for every iteration
        # of the loop over z_array
        if return_spectrum:
            return -99, -99, -99, [-99], [-99]
        else:
            return -99, -99, -99
    else:
        # retrieve model parameters for best fitting model (minimum chi
        # squared)
        j = (int(lsave) / int(len(EW_options)**3))
        Ha_strength = int(lsave) % int(
            len(EW_options)**3) / int(len(EW_options)**2)
        O2_strength = int(lsave) % int(
            len(EW_options)**3) % int(len(EW_options)**2) / int(len(EW_options))
        O3_strength = int(lsave) % int(
            len(EW_options)**3) % int(len(EW_options)**2) % int(len(EW_options))
        # regenerate best fit model spectrum
        wl_shifted = wl / (1 + z)
        model_flux = np.interp(wl_shifted, spec[j].T[0], spec[j].T[1])
        model_flux = np.array([0 if flux[i] == 0 else model_flux[i]
                               for i in range(len(flux))])
        if wl_shifted[0] < 6463 and wl_shifted[-1] > 6663:
            continuum = np.median(model_flux[np.abs(wl_shifted - 6563 < 300)])
            Ha_line = continuum / (100 * np.sqrt(2 * np.pi)) * \
                np.exp(-((wl_shifted - 6563) / (2 * 100))**2)
        else:
            Ha_line = 0 * wl_shifted
        if wl_shifted[0] < 4700 and wl_shifted[-1] > 5175:
            continuum = np.median(model_flux[np.abs(wl_shifted - 4935 < 300)])
            O3_line = continuum / (175 * np.sqrt(2 * np.pi)) * \
                np.exp(-((wl_shifted - 4935) / (2 * 175))**2)
        else:
            O3_line = wl_shifted * 0
        if wl_shifted[0] < 3600 and wl_shifted[-1] > 3900:
            continuum = np.median(model_flux[np.abs(wl_shifted - 3725 < 300)])
            O2_line = continuum / (65 * np.sqrt(2 * np.pi)) * \
                np.exp(-((wl_shifted - 3725) / (2 * 65))**2)
        else:
            O2_line = wl_shifted * 0
        model_flux_lines = EW_options[Ha_strength] * Ha_line + EW_options[
            O2_strength] * O2_line + EW_options[O3_strength] * O3_line + model_flux
        factor = np.median(model_flux_lines) / np.median(flux)
        # HJD: Not sure how this works!
        mass = ML_ratio * (Distance(z=z).cm * (z + 1))**2 / 3.9e33 / factor
        model_flux_normed = model_flux_lines / factor
        chisq = sum(((flux - model_flux_normed) / flux_error)**2)
        if plot:
            plt.clf()
            plt.plot(wl_shifted, flux, label='Data')
            plt.plot(wl_shifted, model_flux_normed, label='Model')
            plt.legend(loc=1)
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.show()
        if return_spectrum:
            return z, EW_options[Ha_strength], mass, wl_shifted, model_flux_normed
        else:
            return z, EW_options[Ha_strength], mass
