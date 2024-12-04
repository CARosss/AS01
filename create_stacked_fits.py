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

def main(spectra, factor, cluster_name):
    import numpy as np
    data = []

    def clean_and_normalize_spectrum(wave, flux, ivar):
        # Remove NaN and inf values
        good_idx = np.isfinite(flux) & np.isfinite(ivar)
        wave, flux, ivar = wave[good_idx], flux[good_idx], ivar[good_idx]

        # normalise to flux at 5300
        target_wavelength = 5300
        tolerance = 2
        closest_idx = np.argmin(np.abs(wave - target_wavelength))
        normalizing_wave = wave[closest_idx]

        if abs(normalizing_wave - target_wavelength) > tolerance:
            raise ValueError(f"Closest wavelength {normalizing_wave:.2f} is more than "
                             f"{tolerance}Å from target {target_wavelength}Å")

        normalizing_flux = flux[closest_idx]

        if not np.isfinite(normalizing_flux) or normalizing_flux == 0:
            raise ValueError(f"Invalid normalizing flux value: {normalizing_flux}")

        flux = flux / normalizing_flux
        ivar = ivar * (normalizing_flux ** 2)

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
    ages = []
    metallicity = []
    DoRs = []

    for filename in spectra:
        wave, flux, ivar = load_spectrum("fits_shortlist/" + filename)  # data[0]
        plate, mjd, fiber = parse_sdss_filename(filename)

        matching_row = catalogue[(catalogue['plate'] == plate) &
                                 (catalogue['mjd'] == mjd) &
                                 (catalogue['fiberid'] == fiber)]

        mgfe.append(float(matching_row['MgFe'].iloc[0]))
        vd = matching_row["velDisp_ppxf_res"].iloc[0]
        vdisps.append(float(vd))
        ages.append(matching_row['age_mean_mass'])
        metallicity.append(matching_row['[M/H]_mean_mass'])
        DoRs.append(float(matching_row['DoR'].iloc[0]))


        data.append([wave, flux, ivar])

    print(f"STATS: ({len(spectra)} items)")
    print("--> Max vdisp:", max(vdisps))
    print("--> mgfe avg:", np.mean(mgfe))
    print("--> age avg:", np.mean(ages))
    print("--> metallicity avg:", np.mean(metallicity))
    print("--> DoR avg:", np.mean(DoRs))
    print("s.dev's:")
    print("--> mgfe std:", np.std(mgfe))
    print("--> age std:", np.std(ages))
    print("--> metallicity std:", np.std(metallicity))
    print("--> DoR std:", np.std(DoRs))

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

    """
    IF WE USE JOHN'S SCRIPT I DONT THINK WE NEED ANYTHING MORE THAN THE ABOVE DATA! JUST NEED TO WRITE IT OUT INTO A 
    FITS FILE. 
    """

    def run_fit():

        # wavelength, flux, combined_ivar = combine_spectra(smoothed)

        errors = safe_errors(combined_ivar)  # use combined_ivar instead of ivar


        lam_gal = wavelength
        ln_lam_gal = np.log(lam_gal)  # Natural logarithm
        d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0]) / (ln_lam_gal.size - 1)  # Use full lam range for accuracy
        velscale = c * d_ln_lam_gal
        # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

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

        vel0 = c * np.log(1)  # redshift meant to be in here too, using 1 as default?
        sol = [vel0, 200]
        start = [sol for j in range(len(moments))]  # adopt the same starting value for both gas and stars

        degree = -1
        mdegree = 10

        pp = ppxf(templates, flux, noise, velscale, start, plot=False,
                  moments=moments, degree=degree, mdegree=mdegree,
                  lam=lam_gal, component=component,
                  gas_component=component > 0, gas_names=gas_names,
                  lam_temp=sps.lam_temp) #, clean=True)

        pp.plot()
        plt.title(f"pPXF fit with {sps_name} SPS templates")
        return pp

    # run_fit()

    def create_fits(wavelength, flux, combined_ivar, cluster_name, mgfe, sigma_fin):

        coadd_data = np.zeros(len(wavelength), dtype=[
            ('flux', 'f8'),
            ('wave', 'f8'),
            ('ivar', 'f8'),
            ('wdisp', 'f8'),
            ('ALPHA', 'f8'),
            ('sigma', 'f8')
        ])

        coadd_data['flux'] = flux
        coadd_data['wave'] = wavelength
        coadd_data['ivar'] = combined_ivar
        # coadd_data['wdisp'] = np.full_like(wavelength, 2.76)  # SDSS instrumental resolution
        coadd_data['ALPHA'] = mgfe
        coadd_data['sigma'] = sigma_fin

        # Create the HDUs
        primary_hdu = fits.PrimaryHDU()
        coadd_hdu = fits.BinTableHDU(data=coadd_data, name='COADD')

        # Add minimal header info needed by analysis script
        primary_hdu.header['HIERARCH NAME'] = f'stacked_{cluster_name}'
        primary_hdu.header['HIERARCH z'] = 0  # Since already rest-framed

        # Create HDUList and write to file
        hdul = fits.HDUList([primary_hdu, coadd_hdu])
        output_file = f'stacked_fits/stacked_{cluster_name}.fits'
        hdul.writeto(output_file, overwrite=True)

        return output_file

    create_fits(wavelength, flux, combined_ivar, cluster_name, mgfe_avg, sigma_fin)
    return None


files = ["cluster_results/k-means_clusters.csv", "cluster_results/gmm_clusters.csv",
         "cluster_results/hierarchical_clusters.csv"]

hierarchical = pd.read_csv(files[2])
kmeans = pd.read_csv(files[0])
gmm = pd.read_csv(files[1])
catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")

# Create file lists for DoR ranges
dor_clusters = []
for threshold in [(0.6, float('inf')), (0.3, 0.6), (0, 0.3)]:
    dor_group = catalogue[(catalogue['DoR'] > threshold[0]) & (catalogue['DoR'] <= threshold[1])]
    file_list = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
                 for plate, mjd, fiber in zip(dor_group['plate'], dor_group['mjd'], dor_group['fiberid'])]
    dor_clusters.append(file_list)

cluster_groups = {
    'DoR': dor_clusters,
    'Hierarchical': [hierarchical[hierarchical["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)],
    'KMeans': [kmeans[kmeans["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)],
    'GMM': [gmm[gmm["Cluster"] == i]["SDSS_ID"].tolist() for i in range(max(gmm["Cluster"]) + 1)]
    #'GMM': [gmm[gmm["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)]
}

colors = ['red', 'blue', 'green']
n = 0.001635
factor = [n * 4, n * 2.7, n * 2.5]

for method, groups in cluster_groups.items():
    print(f"\n================================")
    # print(f"Processing {method} clustering")
    method_labels = [f"{method}_{i}" for i in range(len(groups))]
    method_colors = colors[:len(groups)]
    method_factors = factor[:len(groups)]

    for idx, (spectra, color, label, factor_val) in enumerate(
            zip(groups, method_colors, method_labels, method_factors)):
        print(f"\nStacking {label}")
        main(spectra, factor_val, label)
    print(f"================================")


"""labels = {method: [f"{method}_{i}" for i in range(3)] for method in cluster_groups.keys()}
#colors = ['red', 'blue', 'green']
n = 0.001635
#factor = [n*4, n*2.7, n*2.5]

for method, groups in cluster_groups.items():
    print(f"================================")
    print(f"\nProcessing {method} clustering")
    labels = [f"{method}_{i}" for i in range(len(groups))]
    colors = ['red', 'blue', 'green'][:len(groups)]
    factor = [n*4, n*2.7, n*2.5][:len(groups)]

    for idx, (spectra, color, label) in enumerate(zip(groups, colors, labels[method])):
        print(f"\nFitting {label}")
        main(spectra, factor[idx], label)
    print(f"================================")


"""


"""
Choosing a list of .fits files to stack:

For testing purposes we select the high dor cluster from hierarchical clustering and form a list, data, in which each element is a list of 
[wave, flux, ivar] for one spectrum

For actually fitting each one:


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

# spectra = high_ids  # change to whichever you want


plt.figure(figsize=(15, 6))  # Single larger plot

colors = ['red', 'blue', 'green']
labels_arr = [['High_DoR', 'Medium_DoR', 'Low_DoR'], ['cluster_0', 'cluster_1', 'cluster_2']*3]
labels = ['High_DoR', 'Medium_DoR', 'Low_DoR']
pp_results = []  # Store all pp objects
n = 0.001635
factor = [ n*4,  n*2.7,  n*2.5]

# First run all fits
for idx, (spectra, color, label) in enumerate(zip([high_ids, med_ids, low_ids], colors, labels)):
    print("\n"*5)
    print("NOW FITTING:", label)
    pp = main(spectra, factor[idx], label)
    pp_results.append(pp)

plotting = 0
if plotting ==1:
    for pp, color, label in zip(pp_results, colors, labels):
        plt.plot(pp.lam, pp.galaxy, color=color, alpha=0.5, label=f'{label} Data')
        plt.plot(pp.lam, pp.bestfit, color='black', linestyle='--', alpha=0.3, label=f'{label} Fit')

    plt.legend()
    plt.ylabel('Flux')
    plt.xlabel('Wavelength (Å)')
    plt.title('Comparison of DoR Groups with Best Fits')

    y_min = min([pp.galaxy.min() for pp in pp_results])
    y_max = max([pp.galaxy.max() for pp in pp_results])
    x_min = min([pp.lam.min() for pp in pp_results])
    x_max = max([pp.lam.max() for pp in pp_results])
    plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()"""
