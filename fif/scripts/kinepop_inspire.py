import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import ppxf_util as util

from ppxf import ppxf
from astropy.io import fits
from astropy.io import ascii
import os
import pandas as pd

def stacked_fits_to_csv(fits_file_path, output_csv_path):
    with fits.open(fits_file_path) as hdul:
        # Get the data from the COADD extension
        coadd_data = hdul['COADD'].data
        z = 0

        # Extract wavelength and flux
        wave = coadd_data['wave']
        spec = coadd_data['flux']

        # Create DataFrame with the required columns
        df = pd.DataFrame({
            'wave': wave,
            'spec': spec,
            'z': z
        })

        # Save to CSV
        df.to_csv(output_csv_path, index=False)

        print(f"Successfully converted {fits_file_path} to {output_csv_path}")
        print(f"Wavelength range: {wave[0]} - {wave[-1]}")
        print(f"Number of points: {len(wave)}")
        print(f"Redshift: {z}")
        return z

def fits_shortlist_to_csv(fits_file_path, output_csv_path):
    with fits.open(fits_file_path) as hdul:
        # Get spectrum data from extension 1
        coadd = hdul[1].data
        flux = coadd['flux']
        flux = flux * 1e-17
        loglam = coadd['loglam']

        # Get redshift from extension 2
        specobj = hdul[2].data
        z = specobj['Z'][0]

        # Convert log wavelength to wavelength and apply redshift correction
        wavelength = 10 ** loglam
        wavelength *= 1 / (1 + z)

        # Create DataFrame with the required columns
        df = pd.DataFrame({
            'wave': wavelength,
            'spec': flux,
            'z': z
        })

        # Save to CSV
        df.to_csv(output_csv_path, index=False)

        print(f"Successfully converted {fits_file_path} to {output_csv_path}")
        print(f"Wavelength range: {wavelength[0]} - {wavelength[-1]}")
        print(f"Number of points: {len(wavelength)}")
        print(f"Redshift: {z}")
        return z

# comment for whichever you want to use:

fits_file_path = '../../stacked_fits/stacked_Hierarchical_0.fits'
csv_file_path = '../data/spectrum.csv'
zgal = stacked_fits_to_csv(fits_file_path, csv_file_path)

# fits_file_path = '../../fits_shortlist/spec-0273-51957-0005.fits'
# csv_file_path = '../data/spectrum.csv'
# zgal = fits_shortlist_to_csv(fits_file_path, csv_file_path)

clight = 299792.458
noise_level = 0.035 #This is the "noise" (constant) level. It should be changed in order to get a chi2 as closer to 1 as possible

file_list = np.array([csv_file_path])
path = '../data/'

vel_disp_array = np.array([])
age_array = np.array([])
metal_array = np.array([])
imf_array = np.array([])
object_name_array = np.array([])





for i in range(len(file_list)):

    file = 'spectrum'  # Just use the base name
    file_csv = csv_file_path  # Use the full path you already defined
    file_pkl = path + '/spectrum.pkl'  # Add proper path separator
    print(file)
    #file_csv = path+ file + '.csv' #../data/non_relics/J0326-3303.csv
    #file_pkl = path+ file + '.pkl' #../data/non_relics/J0326-3303.pkl
    print(file_csv)
    dat = ascii.read(file_csv)
    lamb = dat['wave']
    flux = dat['spec']
    
    mask = (lamb > 3800) & (lamb < 7500)
    spec = flux[mask]
    wave = lamb[mask]


    # Plot raw spectrum
    plt.figure(figsize=(15, 8))
    plt.plot(wave, spec, 'k-', label='Data')
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.title('Input Spectrum')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()

    # Then continue with the log-rebinning
    galaxy, logLam1, velscale = util.log_rebin([min(wave), max(wave)], spec)
    
    # Log-rebin
    galaxy, logLam1, velscale = util.log_rebin([min(wave),max(wave)], spec)
    galaxy = galaxy / np.median(galaxy)
    
    
    # Prepare templates
    miles = glob.glob('../templates/PPXF_4/*')    
    
    # miles = glob.glob('../templates/PPXF_E/*T01*')
    # miles.extend(glob.glob('../templates/PPXF_E/*T02*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T03*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T04*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T06*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T08*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T10*')) 
    # miles.extend(glob.glob('../templates/PPXF_E/*T12*')) 
    
    # miles = glob.glob('../templates/Na_IMF/*T01*')
    # miles.extend(glob.glob('../templates/Na_IMF/*T02*')) 
    # miles.extend(glob.glob('../templates/Na_IMF/*T03*')) 
    # miles.extend(glob.glob('../templates/Na_IMF/*T04*')) 
    # miles.extend(glob.glob('../templates/Na_IMF/*T06*')) 
    # miles.extend(glob.glob('../templates/Na_IMF/*T08*')) 
    # miles.extend(glob.glob('../templates/Na_IMF/*T10*'))
    
    hdu_ssp = fits.open(miles[0])
    ssp = hdu_ssp[0].data
    h2 = hdu_ssp[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    mwave     = h2['CRVAL1'] + np.arange(h2['NAXIS1'])*h2['CDELT1']
    sspNew, logLam2, fvelscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    templates = np.empty((sspNew.size, len(miles)))
    
    # Convolve SSPs to MUSE resolution
    ages = np.zeros(len(miles))
    mets = np.zeros(len(miles))
    imfs = np.zeros(len(miles))
    for j, mfile in enumerate(miles):
        hdu = fits.open(mfile)
        print(mfile)
        ssp = hdu[0].data
        sspNew, logLam2, fvelscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
        
        # Save lum-weighted SSPs
        templates[:,j] = sspNew / np.median(sspNew)
        
        # ---> Also save the SSP values
        
        ages[j] = float(mfile[34:41])
        # print('ages:',ages[j])
        
        sign = mfile[28]
        if sign == 'm': mets[j] = - float(mfile[29:33])
        if sign == 'p': mets[j] =  float(mfile[29:33])
        # print('mets:',mets[j])
        
        imfs[j] = float(mfile[23:27])
        # print('imfs:',imfs[j])
        
        # ages[j] = hdu[0].header['Age']
        # mets[j] = hdu[0].header['MET']
        # imfs[j] = hdu[0].header['IMF']        
        
    # Now let's sort the SSPs
    
    uages = np.sort(list(set(ages)))
    umets = np.sort(list(set(mets)))
    uimfs = np.sort(list(set(imfs)))
    print('ages:',np.unique(ages))
    print('mets:',np.unique(mets))
    print('imfs:',np.unique(imfs))
    
    templates_pop = np.zeros((sspNew.size,len(uages),len(umets),len(uimfs)))
    tshape  = templates_pop.shape
    tlength = tshape[1]*tshape[2]*tshape[3]
    age_arr = np.zeros((len(uages),len(umets),len(uimfs)))
    met_arr = np.zeros((len(uages),len(umets),len(uimfs)))
    imf_arr = np.zeros((len(uages),len(umets),len(uimfs)))    
    for a, iage in enumerate(uages):
        for m, imet in enumerate(umets):
            for j, iimf in enumerate(uimfs):
                ssp_id = (ages == iage) & (mets == imet) & (imfs == iimf)
                templates_pop[:,a,m,j] = templates[:,ssp_id].flatten()
                age_arr[a,m,j] = iage
                met_arr[a,m,j] = imet
                imf_arr[a,m,j] = iimf
    
    # Gas templates
    nl_templates, nl_names, line_wave = util.emission_lines(logLam2, [3800.,7400.], 2.51)
    ftemplates = np.column_stack([templates,nl_templates])
    n_temps = templates.shape[1]
    n_nl = len(nl_names)
    
    component = [0]*n_temps + [1] * n_nl 
    gas_component = np.array(component) > 0
    moments = [2,2]
         
    # OK, it's pPXF time
    # ---> Check that every pixel is fine
    pbad = (galaxy >= 10) | (galaxy <= 0) | (np.isinf(galaxy)) | (np.isnan(galaxy))
    galaxy[pbad] = 0.
            
    # ---> Don't fit NaD
    goodpixels_bool = ( ((np.exp(logLam1) >= 4750.) & (np.exp(logLam1) <= 5860.)) | 
                        ((np.exp(logLam1) >= 5950.) & (np.exp(logLam1) <= 7400.)) 
                        )
    goodpixels = np.array([j for j, x in enumerate(goodpixels_bool) if x])
    
    
    # Stars + gas
    dv = (logLam2[0] - logLam1[0])*clight
    start = [zgal*clight, 100]
    start = [start, start]
    
    # Run pPXF!
    # print('   First pPXF run (kine+gas)')
    print(len(ftemplates))
    print(len(galaxy))
    
    galaxy_err = np.abs(galaxy*0.+noise_level)
    galaxy_err[~np.isfinite(galaxy_err)] = 100.
    pp = ppxf(ftemplates, galaxy, galaxy_err, velscale, start,quiet=False, clean=True,
            goodpixels=goodpixels, moments=moments, degree=-1,mdegree=12,vsyst=dv,lam=wave,
            plot=True,component=component,gas_component=gas_component)
    
    kine = pp.sol[0]
    
    avgerr = np.std(np.abs((galaxy[goodpixels]-pp.bestfit[goodpixels])))
    nerr   = np.zeros(len(galaxy)) + avgerr
    galaxy = galaxy - pp.gas_bestfit 
    
    rparam    = 0.025 #This is the regularisation level. The higher, the smoother the solution will be
    noise_level = 0.05 #This is the "noise" (constant) level. It should be changed in order to get a chi2 as closer to 1 as possible
    
    pp_reg = ppxf(templates_pop,galaxy,nerr,velscale,kine,quiet=False,clean=True,
            goodpixels=goodpixels,moments=-2,degree=-1,mdegree=12,lam=wave,
            vsyst=dv,regul=1./rparam,plot=True)
    
    
    mean_wts = pp_reg.weights.reshape(tshape[1:])/pp_reg.weights.sum()
    lage = np.sum(age_arr*mean_wts)
    lmet = np.sum(met_arr*mean_wts)
    limf = np.sum(imf_arr*mean_wts)
    
    # Redshift correction
    vred = kine[0]
    vsig = kine[1]
    zred = np.sqrt((1. + vred/clight) / (1. - vred/clight)) - 1.
    ln_lam  = logLam1 - np.log(zred + 1.)
    
    # Back to linear space
    lin_spec = np.interp(mwave, np.exp(ln_lam), galaxy)
    lin_fit  = np.interp(mwave, np.exp(ln_lam), pp_reg.bestfit)
    lin_noise =  np.abs(lin_fit-lin_spec)
    
    wgood = (mwave >= 3800.) & (mwave <= 9000.)
    lin_noise  = lin_noise / np.nanmedian(lin_spec[wgood])
    lin_spec = lin_spec / np.nanmedian(lin_spec[wgood])
    
    lin_spec[(mwave <= 3650.)] = 1.
    lin_noise[(mwave <= 3650.)] = 1.
    
    # Save results
    struct = {'sigma':vsig, 'spectra':lin_spec, 'error':lin_noise, 'Vsys':vred, 'wave':mwave,
            'lage':lage, 'lmet':lmet, 'limf':limf,'age_arr':age_arr, 'met_arr':met_arr, 
            'imf_arr':imf_arr, 'wts_arr':mean_wts,'tshape':tshape}
    
    
    print('Velocity dispersion', vsig)
    print('Age', lage)
    print('Metallicity', lmet)
    print('IMF slope:', limf)
    object_name_array = np.append(object_name_array,file)
    vel_disp_array = np.append(vel_disp_array,vsig)
    age_array = np.append(age_array,lage)
    metal_array = np.append(metal_array,lmet)
    imf_array = np.append(imf_array,limf)
    
    
    
    #Plot spectral fit
    plt.figure(figsize=(30,16))
    pp.plot()
    plt.show()

    #Plot reg_spectral fit
    plt.figure(figsize=(30,16))
    pp_reg.plot()
    plt.show()
    
    
    ofile = file_pkl
    output = open(ofile, 'wb')         
    pickle.dump(struct, output)
    output.close()
    
 

df = pd.DataFrame({
    'name': object_name_array,
    'ppxf_velocity_dispersion': vel_disp_array,
    'ppxf_age': age_array,
    'ppxf_metallicity': metal_array,
    'ppxf_imf':imf_array
})  

df.to_csv(path+'modelled_ppxf_data.csv', index=False)  

"""
#script finished running alert
from playsound import playsound
playsound("C:/Users/misia/Documents/sounds/X2Download.app - Heroes 3 Sounds - New Week (128 kbps).mp3")
import ctypes
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
Mbox('FINISHED!', 'FINISHED!', 1) """