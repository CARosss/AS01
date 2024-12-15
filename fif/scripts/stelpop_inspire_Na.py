def gaussian_prior(pval, gmean, gsig):
    return np.exp(-np.power(pval - gmean, 2.) / (2 * np.power(gsig, 2.)))

def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating  template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

def fif(wave, flux, model, index, dat):
    
    cont_b = ((wave >= dat[dat['NAME'] == index]['Blue_1'][0]) & 
            (wave <= dat[dat['NAME'] == index]['Blue_2'][0]) & (model > 0) & (flux > 0))
    
    cont_r = ((wave >= dat[dat['NAME'] == index]['Red_1'][0]) & 
            (wave <= dat[dat['NAME'] == index]['Red_2'][0]) & (model > 0) & (flux > 0))
    
    conti = cont_b | cont_r
    
    band_pass = ((wave >= dat[dat['NAME'] == index]['Line_1'][0]) & 
                 (wave <= dat[dat['NAME'] == index]['Line_2'][0]))
    
    resu = np.polyfit(wave[conti],(flux[conti]/model[conti]),1)
    
    return flux[band_pass]/np.polyval(resu,wave[band_pass])

def load_models(SIGgal):

    # Locate the models
    #!!!
    files = glob.glob('../templates/Na_IMF/Mbi*.fits')
    
    # Some relevant information
    tmp  = fits.open(files[0])[0].data
    plength =  len(tmp[pini:pend])
    nmodels = len(files)
    psize   = 0.9
    wave    = 3540.5 + np.arange(plength) * psize
    FWHM_gal = 2.*np.sqrt(2.*np.log(2.)) * SIGgal / cvel * wave
    FWHM_dif = np.sqrt((FWHM_gal**2).clip(0))
    sigma = FWHM_dif/2.355/psize # Sigma difference in pixels
    
    # Let's create the cube
    ages = np.zeros(len(files))
    mets = np.zeros(len(files))
    alps = np.zeros(len(files))
    imfs = np.zeros(len(files))
    tife = np.zeros(len(files))
    # !!!
    nafe = np.zeros(len(files))
    
    npix = len(anode)
    model_indices = np.zeros((len(files),npix))
    for i, mfile in enumerate(files):
        
        hdu = fits.open(mfile)        
        hdr = hdu[0].header
        ssp = gaussian_filter1d(hdu[0].data[pini:pend], sigma)
        model_indices[i,:] = ssp[anode] / np.median(ssp[anode])

        ages[i] = hdr['age']     # Age
        mets[i] = hdr['met']     # Met
        alps[i] = hdr['alp']     # Alp
        imfs[i] = hdr['imf']     # IMF
        tife[i] = hdr['tif']     # [Ti/Fe]
        # !!!
        nafe[i] = hdr['naf']     # [Na/Fe]
    print('maaax',np.max(mets)) 
    print('miin',np.min(mets))
    uages = np.unique(ages)
    umets = np.unique(mets)
    ualps = np.unique(alps)
    uimfs = np.unique(imfs)
    utifs = np.unique(tife)
    # !!!
    unafs = np.unique(nafe)
    # print(np.min(uages),np.max(uages),np.min(umets),np.max(umets),np.min(ualps),np.max(ualps),np.min(uimfs),np.max(uimfs),np.min(utifs),np.max(utifs),np.min(unafs),np.max(unafs))
    npix = len(anode)
    grid_indices = np.zeros((len(uages),len(umets),len(ualps),len(uimfs),len(utifs),len(unafs),npix))
    for y, ytmp in enumerate(uages):
        for m, mtmp in enumerate(umets):
            for a, atmp in enumerate(ualps):
                for i, itmp in enumerate(uimfs):
                    for t, ttmp in enumerate(utifs):
                        # !!!
                        for k, ktmp in enumerate(unafs):
                            idx = (ages == ytmp) & (mets == mtmp) & (alps == atmp) & (imfs == itmp) & (tife == ttmp) & (nafe == ktmp)
                            grid_indices[y,m,a,i,t,k,:] = np.squeeze(model_indices[idx,:])
    
    
    #!!!
    my_interpolating_function = RegularGridInterpolator((uages, umets, ualps, uimfs, utifs, unafs, np.arange(npix)), grid_indices[:,:,:,:,:,:,:])
    return my_interpolating_function, npix

def lnprior(par, pmet, pimf):
    """
    Iformative priors
    """    
    
    # prior_met = gaussian_prior(par[0], pmet, 0.02)
    prior_imf = gaussian_prior(par[2], pimf, 0.5)
    
    if (

       # Rejecting solutions outside some boundary limits
       # !!!
       (par[0] >= -0.66) and (par[0] <= 0.4) and
       (par[1] >=  0.0) and (par[1] <= 0.4) and
       (par[2] >=  0.3) and (par[2] <= 3.3) and
       (par[3] >= -0.3) and (par[3] <= 0.3) and
       (par[4] >= -0.3) and (par[4] <= 0.3)
       ): prior_abu = 0.0

    else:
        prior_abu = -np.inf
        
    prior_comb = np.log(prior_imf) + prior_abu
    # prior_comb = prior_abu

    return prior_comb

def lnprob(par,data,error,lage, pmet, pimf):
    """
    Calculate the posterior distribution
    """
    
    # Checking the priors
    lp = lnprior(par,pmet,pimf)
    if not np.isfinite(lp):
        return -np.inf

    # Interpolating the model grid indices at desired point
    try_par = np.append(lage,par)
    out_indices = model_pred(try_par,fint,npix,wave,data,anode,dat)
    
    # Computing the likelyhood for a given set of params
    bad = (error <= 0.0)
    if np.any(bad):
        error = error * 0.0 + 1E10

    data_arr  = np.array(data)
    error_arr = np.array(error)
    model_arr = np.array(out_indices)

    sigma2 = error_arr**2
    lnlike = -0.5*np.sum(((data_arr)[cnode]-(model_arr))**2 / sigma2[cnode] + np.log(sigma2[cnode]))
    
    # Safety check. If lnlike is not finite then return -np.inf
    if not np.isfinite(lnlike):
        return -np.inf
    
    return lp + lnlike

def ssppop_fitting(data,error,nwalkers,nchain,lage,pmet,pimf):
    """
    Run walkers, run!
    """

    # Defining an initial set of walkers
    # !!!
    zpt = [0.,0.1,1.3,0.11,0.11]
    p0  = [zpt+1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    # Setting up the sampler
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(data,error,lage,pmet,pimf))    
    sampler.run_mcmc(p0, nchain)
    
    return sampler

def model_pred(try_par,fint,npix,wave,data,anode,dat):

    pp =  np.array(list(map(lambda x:np.append(try_par,x), np.arange(npix)))) 
    tmp_indices = fint(pp)
    
    out_indices = []
    for index in dat['NAME']:
        out_indices = np.append(out_indices,fif(wave[anode], tmp_indices, data[anode], index,dat))

    return out_indices

def good_nodes(wave,ifile):
    
    iwave = np.arange(len(wave))
    gidx  = []
    aidx  = []

    for k in range(len(dat['Blue_1'])):

        cont_b = ((wave >= dat['Blue_1'][k]) & (wave <= dat['Blue_2'][k]))    
        cont_r = ((wave >= dat['Red_1'][k]) & (wave <= dat['Red_2'][k]))
        bpass  = (wave>=dat['Line_1'][k]) & (wave<=dat['Line_2'][k])
        pix = cont_b | cont_r | bpass

        gidx = np.append(gidx,iwave[bpass])
        aidx = np.append(aidx,iwave[pix])

    aidx = np.unique(aidx)
  
    return gidx, aidx



from matplotlib import cm
def plot_line_regions(wave,minimum):
    iwave = np.arange(len(wave))
    lower_end=0
    for k in range(len(dat['Blue_1'])):

        bpass  = (wave>=dat['Line_1'][k]) & (wave<=dat['Line_2'][k])
        spectral = cm.get_cmap('plasma', 500)
        col=spectral(k/len(dat['Blue_1']))
        
        plt.axvspan(lower_end, len(iwave[bpass])-1+lower_end, alpha=0.3,color=col)
        plt.annotate(dat['NAME'][k],(lower_end,0.99*minimum), color=col)
        lower_end=len(iwave[bpass])+lower_end
    return


#import corner    
import emcee    
import glob
import os
import sys 
from time import time
import pickle
import h5py

import numpy as np
import scipy.spatial.qhull as qhull

from astropy.io import ascii
from astropy.io import fits
from os import path as path

from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
# from ipdb import set_trace as stop




if __name__ == '__main__':
    
    # Fixed params
    FWHM_data =  2.51
    FWHM_models = 2.51
    cvel  = 299792.458
    #!!!
    pini = 450
    pend = 4200

    # EMCEE params
    nwalkers = 24
    nchain   = 1000
    nburn    = 400
    # !!!
    ndim     = 5
    progress = True
    w0       = 4861.135
    
    path = '../data/'
    all_file_list = os.listdir(path=path)
    file_list=np.array([])
    for x in all_file_list:
        if x.endswith(".pkl"): file_list=np.append(file_list,x)
    print(file_list)
    
    
    object_name_array = np.array([])
    metal_array = np.array([])
    metal_array_perror = np.array([])
    metal_array_merror = np.array([])
    a_Fe_array = np.array([])
    a_Fe_array_perror = np.array([])
    a_Fe_array_merror = np.array([])
    imf_array = np.array([])
    imf_array_perror = np.array([])
    imf_array_merror = np.array([])
    Ti_Fe_array = np.array([])
    Ti_Fe_array_perror = np.array([])
    Ti_Fe_array_merror = np.array([])
    Na_Fe_array = np.array([])
    Na_Fe_array_perror = np.array([])
    Na_Fe_array_merror = np.array([])
    
    for i in range(len(file_list)):
        
        file = file_list[i][:-4] # J0326-3303
        print(file)
        file_csv = path+ file + '.csv' #../data/non_relics/J0326-3303.csv
        file_pkl = path+ file + '.pkl' #../data/non_relics/J0326-3303.pkl
        print(file_csv)
        
        # Load measurements
        # pfile = '../data//J1142+0012.pkl'
        # pfile = '../data//J0909+0147.pkl'
        # pfile = '../data//J0211-3155.pkl'
        # pfile = '../data//J0842+0059.pkl'
        # pfile = '../data//J0847+0112.pkl'
        # pfile = '../data//J1040+0056.pkl'
        # pfile = '../data//J1438-0127.pkl'
        # pfile = '../data//J2204-3112.pkl'
        # pfile = '../data//J2305-3436.pkl'
        # pfile = '../data//J2359-3320.pkl'
        pfile = file_pkl
        pickle_in = open(pfile,'rb')
        prof = pickle.load(pickle_in,encoding='latin1') 
        pickle_in.close()
        
        sigma = np.array(prof['sigma'])
        wave  = np.array(prof['wave'])[pini:pend]
        print('wave: ',wave)
        # print(wave)
        # print(np.array(prof['spectra'])[pini:pend])
        
        # Index information
        ifile = 'index_AO_vTiO1_misia.def'
        dat   = ascii.read(ifile)
        cnode, anode = good_nodes(wave,ifile)
        cnode = cnode.astype('int')  
        anode = anode.astype('int')     
        
        # Prepare the models
        fint, npix = load_models(sigma) 
    
        lage   = np.array(prof['lage'])
        pmet   = np.array(prof['lmet'])
        print('max',np.max(pmet))
        print('1')
        pimf   = np.array(prof['limf'])
        
        data = np.array(prof['spectra'])[pini:pend]
        nois = np.array(prof['error'])[pini:pend]
        mwav = np.array(prof['wave'])[pini:pend]
               
        error = np.zeros(len(data))+np.median(nois)
    
        sampler = ssppop_fitting(data,error,nwalkers,nchain,lage,pmet,pimf)
    
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim)) 
        
        bf_par = np.percentile(samples,50,axis=0)
        try_par = np.append(lage,bf_par)
        bf_index = model_pred(try_par,fint,npix,wave,data,anode,dat)
        
        # !!!
        bf_met = np.percentile(samples[:,0],50)
        bf_alp = np.percentile(samples[:,1],50)
        bf_imf = np.percentile(samples[:,2],50)
        bf_tif = np.percentile(samples[:,3],50)
        bf_naf = np.percentile(samples[:,4],50)
        
        lw_met = np.percentile(samples[:,0],16)
        lw_alp = np.percentile(samples[:,1],16)
        lw_imf = np.percentile(samples[:,2],16)
        lw_tif = np.percentile(samples[:,3],16)
        lw_naf = np.percentile(samples[:,4],16)
    
        hg_met = np.percentile(samples[:,0],84)
        hg_alp = np.percentile(samples[:,1],84)
        hg_imf = np.percentile(samples[:,2],84)
        hg_tif = np.percentile(samples[:,3],84)
        hg_naf = np.percentile(samples[:,4],84)
    
        sampler.reset()
    
        # Plot the results
        plt.figure(figsize=(20/500*len(cnode),8),dpi=200)
        plt.plot(data[cnode],label='data',color='k')
        plt.plot(bf_index,label='Best-fit',color='r')
        
        title = file
        plt.title(label=title,loc='center',weight='bold')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        minimum=np.min(data[cnode])
        plot_line_regions(wave,minimum)
        
        plt.legend()
        
        plt.savefig(path + file + '_corner_plot.png')
        
        
        print('IMF slope', bf_imf, '+',hg_imf-bf_imf,'-',bf_imf-lw_imf)
        print('Metallicity', bf_met, '+',hg_met-bf_met,'-',bf_met-lw_met)
        print('Alpha/Fe', bf_alp, '+',hg_alp-bf_alp,'-',bf_alp-lw_alp)
        print('Ti/Fe', bf_tif, '+',hg_tif-bf_tif,'-',bf_tif-lw_tif)
        print('Na/Fe', bf_naf, '+',hg_naf-bf_naf,'-',bf_naf-lw_naf)
        
        object_name_array = np.append(object_name_array,file)
        metal_array = np.append(metal_array,bf_met)
        metal_array_perror = np.append(metal_array_perror,hg_met-bf_met)
        metal_array_merror = np.append(metal_array_merror,bf_met-lw_met)
        a_Fe_array = np.append(a_Fe_array,bf_alp)
        a_Fe_array_perror = np.append(a_Fe_array_perror,hg_alp-bf_alp)
        a_Fe_array_merror = np.append(a_Fe_array_merror,bf_alp-lw_alp)
        imf_array = np.append(imf_array,bf_imf)
        imf_array_perror = np.append(imf_array_perror,hg_imf-bf_imf)
        imf_array_merror = np.append(imf_array_merror,bf_imf-lw_imf)
        Ti_Fe_array = np.append(Ti_Fe_array,bf_tif)
        Ti_Fe_array_perror = np.append(Ti_Fe_array_perror,hg_tif-bf_tif)
        Ti_Fe_array_merror = np.append(Ti_Fe_array_merror,bf_tif-lw_tif)
        Na_Fe_array = np.append(Na_Fe_array,bf_naf)
        Na_Fe_array_perror = np.append(Na_Fe_array_perror,hg_naf-bf_naf)
        Na_Fe_array_merror = np.append(Na_Fe_array_merror,bf_naf-lw_naf)
        
       #plot corner plot
        import corner
        from matplotlib import cm
        cmap = cm.get_cmap('plasma_r')
        contour_kwargs = {'colors':None,'cmap':cmap}
    
        fig=plt.figure(figsize=(8,8),dpi=200)
        
        # !!!
        labels = [ 'Metallicity', r'$\alpha$\Fe', 'IMF','Ti/Fe','Na/Fe']
        corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True,
        title_kwargs={"fontsize": 11},title_fmt='.3f',fig=fig,smooth=True,contour_kwargs=contour_kwargs)
        fig.suptitle(title, fontsize=16, weight='bold')
        
        #annotate which lines were fitted to create corner plot
        fitted_lines = 'fitted lines:'
        #if len(dat['Blue_1']) >15:
            
        for k in range(len(dat['Blue_1'])):
            fitted_lines = fitted_lines + '\n'+ dat['NAME'][k]
        plt.figtext(0.82, 0.4, fitted_lines, fontsize=12)
        plt.savefig(path+file+'_spectrum_plot.png')
        
        
    df = pd.DataFrame({
        'name': object_name_array,
        'FIF_metallicity': metal_array,
        'FIF_metallicity_plus_error':metal_array_perror,
        'FIF_metallicity_minus_error':metal_array_merror,
        'FIF_alpha_Fe':a_Fe_array,
        'FIF_alpha_Fe_plus_error':a_Fe_array_perror,
        'FIF_alpha_Fe_minus_error':a_Fe_array_merror,
        'FIF_IMF':imf_array,
        'FIF_IMF_plus_error':imf_array_perror,
        'FIF_IMF_minus_error': imf_array_merror,
        'FIF_Ti_Fe': Ti_Fe_array,
        'FIF_Ti_Fe_plus_error': Ti_Fe_array_perror,
        'FIF_Ti_Fe_minus_error': Ti_Fe_array_merror,
        'FIF_Na_Fe':Na_Fe_array,
        'FIF_Na_Fe_plus_error': Na_Fe_array_perror,
        'FIF_Na_Fe_minus_error':Na_Fe_array_merror 
    })  

    df.to_csv(path+'modelled_FIF_data.csv', index=False)  
