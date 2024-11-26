from time import perf_counter as clock
from pathlib import Path
from urllib import request
import numpy as np
from astropy.io import fits
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def main(spectra):
    import numpy as np
    data = []

    def clean_and_normalize_spectrum(wave, flux, ivar):
        # Remove NaN and inf values
        good_idx = np.isfinite(flux) & np.isfinite(ivar)
        wave, flux, ivar = wave[good_idx], flux[good_idx], ivar[good_idx]

        # Simple normalization by median flux
        median_flux = np.median(flux)

        # normalize by 5300 instead
        area = wave[wave > 5200]
        area = area[area < 5400]
        if len(area) < 1:
            print("ERROR")

        normalizing_wave = np.median(area)

        flux = flux / normalizing_wave
        ivar = ivar * (normalizing_wave ** 2)

        return wave, flux, ivar


    def load_spectrum(filename):
        with fits.open(filename) as hdul:
            coadd = hdul[1].data  # Extension 1 contains the spectrum
            flux = coadd['flux']
            loglam = coadd['loglam']
            ivar = coadd['ivar']
            specobj = hdul[2].data
            z = specobj['Z'][0]

            wavelength = 10 ** loglam
            wavelength *= 1 / (1 + z)
            wavelength, flux, ivar = clean_and_normalize_spectrum(wavelength, flux, ivar)

            return wavelength, flux, ivar


    def parse_sdss_filename(filename):
        # Extract numbers using regex
        match = re.match(r'spec-(\d{4})-(\d{5})-(\d{4})\.fits', filename)
        if match:
            plate, mjd, fiber = map(int, match.groups())
            return plate, mjd, fiber
        return None, None, None


    catalogue = pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
    mgfe = []
    vdisps = []

    for filename in spectra:
        wave, flux, ivar = load_spectrum("fits_shortlist/" + filename)  # data[0]
        plate, mjd, fiber = parse_sdss_filename(filename)

        matching_row = catalogue[(catalogue['plate'] == plate) &
                                 (catalogue['mjd'] == mjd) &
                                 (catalogue['fiberid'] == fiber)]
        mgfe.append(float(matching_row['MgFe']))
        vd = matching_row["velDisp_ppxf_res"]
        vdisps.append(float(vd))

        data.append([wave, flux, ivar])

    mgfe_avg = round(np.mean(mgfe), 1)  # Rounds to nearest 0.1
    vd_avg = round(np.mean(vdisps), 1)  # Rounds to nearest 0.1
    #print(vd_avg)

    #print(data[0][0][0])  # shortest wavelength for first spectrum
    #print(data[1][0][0])  # "" "" for second spectrum
    # Clearly we will need to do some aligning ... ?

    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    c = 299792.458  # speed of light in km/s


    def calculate_sigma_diff(wave, sigma_gal, sigma_fin):
        # For log-binned spectra:
        ln_wave = np.log(wave)
        d_ln_wave = (ln_wave[-1] - ln_wave[0]) / (len(wave) - 1)  # log step size
        velscale = c * d_ln_wave  # Velocity scale in km/s per pixel

        wave_ref = np.mean(wave)  # Use mean wavelength as reference
        sigma_diff_kms = np.sqrt(sigma_fin ** 2 - sigma_gal ** 2)
        # sigma_ds.append(sigma_diff)

        sigma_gal_pxl = sigma_gal / velscale  # Convert km/s to pixels
        sigma_fin_pxl = sigma_fin / velscale

        sigma_diff = np.sqrt(sigma_fin_pxl ** 2 - sigma_gal_pxl ** 2)

        return sigma_diff, sigma_diff_kms


    def smooth_spectrum_to_sigma(wave, flux, sigma_gal, sigma_fin):
        if sigma_fin <= sigma_gal:
            return flux, sigma_gal

        sigma_diff, sigma_diff_kms = calculate_sigma_diff(wave, sigma_gal, sigma_fin)
        # Apply smoothing with single sigma_diff value
        smoothed_flux = gaussian_filter1d(flux, sigma_diff)
        # print(f"Original sigma: {sigma_gal:.2f} km/s, Smoothing kernel: {sigma_diff:.2f} pixels")
        return smoothed_flux, sigma_diff_kms


    def smooth(data, sigma_gals, sigma_fin):
        smoothed_data = []
        sigma_ds = []
        for (wave, flux, ivar), sigma_gal in zip(data, sigma_gals):
            smoothed_flux, sigma_diff_kms = smooth_spectrum_to_sigma(wave, flux, sigma_gal, sigma_fin)
            smoothed_data.append([wave, smoothed_flux, ivar])
            sigma_ds.append(sigma_diff_kms)

        return smoothed_data, sigma_ds



    sigma_fin = max(vdisps)
    smoothed, sigma_ds = smooth(data, vdisps, sigma_fin)

    sigma_ds_average = round(np.mean(sigma_ds), 1)  # Rounds to nearest 0.1
    #print(sigma_ds_average)

    from scipy import interpolate

    # resample using logbinning instead
    def resample_spectrum(wave, flux, ivar, new_wave):
        # Interpolate flux onto new wavelength grid
        f = interpolate.interp1d(wave, flux, bounds_error=False, fill_value=0)
        new_flux = f(new_wave)

        # Interpolate ivar (need to handle this carefully)
        f_ivar = interpolate.interp1d(wave, ivar, bounds_error=False, fill_value=0)
        new_ivar = f_ivar(new_wave)

        return new_flux, new_ivar


    # Create common wavelength grid
    wave_min = max([smoothed[i][0][0] for i in range(len(smoothed))])  # Maximum of all minimum wavelengths
    wave_max = min([smoothed[i][0][-1] for i in range(len(smoothed))])  # Minimum of all maximum wavelengths

    # Create logarithmically spaced wavelength grid
    # Method 1: using np.logspace
    wave_common = np.logspace(np.log10(wave_min), np.log10(wave_max), num=3828)

    # OR Method 2: using np.exp(np.linspace)
    # wave_common = np.exp(np.linspace(np.log(wave_min), np.log(wave_max), num=3828))


    # Resample all spectra
    resampled_data = []
    for spectrum in smoothed:
        new_flux, new_ivar = resample_spectrum(spectrum[0], spectrum[1], spectrum[2], wave_common)
        resampled_data.append([wave_common, new_flux, new_ivar])

    dlambda = np.diff(wave_common)
    dlambda_over_lambda = dlambda / wave_common[:-1]
    #print("Δλ/λ values:", dlambda_over_lambda[:5])


    def safe_combine_ivar(*ivars):
        ivar_stack = np.stack(ivars)

        # Create mask for any invalid values
        mask = np.any((ivar_stack == 0) | np.isinf(ivar_stack), axis=0)
        combined = np.zeros_like(ivars[0])

        # Combine valid values
        valid = ~mask
        if np.any(valid):
            # Sum of 1/ivar for valid points
            sum_inv_ivar = np.sum(1.0 / ivar_stack[:, valid], axis=0)
            combined[valid] = 1.0 / sum_inv_ivar

        return combined


    def combine_spectra(aligned_spectra):
        # Extract components
        wavelength = aligned_spectra[0][0]  # All wavelengths should be the same
        fluxes = [spec[1] for spec in aligned_spectra]
        ivars = [spec[2] for spec in aligned_spectra]

        # Calculate mean flux
        combined_flux = np.mean(fluxes, axis=0)

        # Combine inverse variances
        combined_ivar = safe_combine_ivar(*ivars)

        return wavelength, combined_flux, combined_ivar


    def safe_errors(stacked_ivar):
        errors = np.zeros_like(stacked_ivar)
        valid = (stacked_ivar > 0) & np.isfinite(stacked_ivar)
        errors[valid] = 1.0 / np.sqrt(stacked_ivar[valid])
        return errors


    wavelength, flux, combined_ivar = combine_spectra(resampled_data)
    # wavelength, flux, combined_ivar = combine_spectra(smoothed)

    errors = safe_errors(combined_ivar)  # use combined_ivar instead of ivar


    lam_gal = wavelength
    ln_lam_gal = np.log(lam_gal)  # Natural logarithm
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0]) / (ln_lam_gal.size - 1)  # Use full lam range for accuracy
    velscale = c * d_ln_lam_gal
    # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
    factor = 0.000015
    # factor = 0.001635

    noise = np.full_like(flux, factor)  # Assume constant noise per pixel here

    dlam_gal = np.gradient(lam_gal)  # Size of every pixel in Angstroms

    hdul = fits.open('fits_shortlist/spec-0273-51957-0005.fits')
    coadd = hdul[1].data
    wdisp = coadd['wdisp']  # assuming that the wdisp is constant between the two spectra for now.

    fwhm_gal = 2.355 * wdisp * dlam_gal

    # sps_name = 'fsps'
    # sps_name = 'galaxev'
    sps_name = 'emiles'
    # sps_name = 'xsl'

    basename = f"spectra_{sps_name}_9.0.npz"
    ppxf_dir = Path(lib.__file__).parent
    filename = ppxf_dir / 'sps_models' / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)

    fwhm_gal_dic = {"lam": lam_gal, "fwhm": fwhm_gal}
    sps = lib.sps_lib(filename, velscale, fwhm_gal_dic)

    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
    gas_templates, gas_names, gas_wave = \
        util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)
    templates = np.column_stack([stars_templates, gas_templates])
    n_temps = stars_templates.shape[1]

    component = [0] * n_temps  # Single stellar kinematic component=0 for all templates
    component += [1] * len(gas_names)  # First 6 gas templates
    component = np.array(component)
    moments = [2] * len(np.unique(component))  # Now this will create only 2 sets of moments (stars and gas)

    #print(component.shape)
    #print(templates.shape)
    vel0 = c * np.log(1)  # redshift meant to be in here too, using 1 as default?
    sol = [vel0, 200]
    start = [sol for j in range(len(moments))]  # adopt the same starting value for both gas and stars

    degree = -1
    mdegree = 10
    t = clock()

    pp = ppxf(templates, flux, noise, velscale, start, plot=False,
              moments=moments, degree=degree, mdegree=mdegree,
              lam=lam_gal, component=component,
              gas_component=component > 0, gas_names=gas_names,
              lam_temp=sps.lam_temp) #, clean=True)

    #print(f"Elapsed time in pPXF: {(clock() - t):.2f}")
    pp.plot()
    plt.title(f"pPXF fit with {sps_name} SPS templates")
    return pp




"""Choosing a list of .fits files to stack:

For testing purposes we select the high dor cluster from hierarchical clustering and form a list, data, in which each element is a list of 
[wave, flux, ivar] for one spectrum

"""

files = ["cluster_results/k-means_clusters.csv", "cluster_results/gmm_clusters.csv",
         "cluster_results/hierarchical_clusters.csv"]
filename = files[2]  # selecting hierarchical for now
clusters = pd.read_csv(filename)
two = clusters[clusters["Cluster"] == 2]["SDSS_ID"].tolist()  # list of file names selecting cluster 2
one = clusters[clusters["Cluster"] == 1]["SDSS_ID"].tolist()  # selecting 1
zero = clusters[clusters["Cluster"] == 0]["SDSS_ID"].tolist()  # selecting 0

# Now we also try using actual DoR ranges
all = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")

high = all[all['DoR'] > 0.6]
high_ids = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
            for plate, mjd, fiber in zip(high['plate'], high['mjd'], high['fiberid'])]

med1 = all[all['DoR'] < 0.6]
med = med1[med1['DoR'] > 0.3]
med_ids = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
           for plate, mjd, fiber in zip(med['plate'], med['mjd'], med['fiberid'])]

low = all[all['DoR'] < 0.3]
low_ids = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
           for plate, mjd, fiber in zip(low['plate'], low['mjd'], low['fiberid'])]

spectra = high_ids  # change to whichever you want
"""


# plt.figure(figsize=(15, 12))

colors = ['red', 'blue', 'green']
labels = ['High DoR', 'Medium DoR', 'Low DoR']

for idx, (spectra, color, label) in enumerate(zip([high_ids, med_ids, low_ids], colors, labels)):
    print("\n"*5)
    print("NOW FITTING:", label)
    pp = main(spectra)

    plt.plot(pp.lam, pp.galaxy, color=color, alpha=0.5, label=f'{label} Data')
    plt.plot(pp.lam, pp.bestfit, color='black', linestyle='--', label='Best Fit')
    plt.legend()
    plt.ylabel('Flux')
    # plt.ylim(0.0009, 0.007)
    # plt.xlim(0.35, 0.75)
    plt.show()"""

"""    plt.subplot(3, 1, idx + 1)
    plt.plot(pp.lam, pp.galaxy, color=color, alpha=0.5, label=f'{label} Data')
    plt.plot(pp.lam, pp.bestfit, color='black', linestyle='--', label='Best Fit')
    plt.legend()
    plt.ylabel('Flux')
    if idx == 2:
        plt.xlabel('Wavelength (Å)')

plt.tight_layout()
plt.show()"""


plt.figure(figsize=(15, 6))  # Single larger plot

colors = ['red', 'blue', 'green']
labels = ['High DoR', 'Medium DoR', 'Low DoR']
pp_results = []  # Store all pp objects

# First run all fits
for idx, (spectra, color, label) in enumerate(zip([high_ids, med_ids, low_ids], colors, labels)):
    print("\n"*5)
    print("NOW FITTING:", label)
    pp = main(spectra)
    pp_results.append(pp)

for pp, color, label in zip(pp_results, colors, labels):
    plt.plot(pp.lam, pp.galaxy, color=color, alpha=0.5, label=f'{label} Data')
    plt.plot(pp.lam, pp.bestfit, color='black', linestyle='--', alpha=0.3, label=f'{label} Fit')

plt.legend()
plt.ylabel('Flux')
plt.xlabel('Wavelength (Å)')
plt.title('Comparison of DoR Groups with Best Fits')

# Find global min/max for consistent axes
y_min = min([pp.galaxy.min() for pp in pp_results])
y_max = max([pp.galaxy.max() for pp in pp_results])
x_min = min([pp.lam.min() for pp in pp_results])
x_max = max([pp.lam.max() for pp in pp_results])

# Add some padding to the limits
plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
plt.xlim(x_min, x_max)

plt.tight_layout()
plt.show()
