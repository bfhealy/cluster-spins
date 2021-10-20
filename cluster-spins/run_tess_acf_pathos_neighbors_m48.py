import emcee
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,argrelmin
from scipy.stats import median_absolute_deviation
import eleanor
import pandas as pd
import urllib
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import TessTargetPixelFile
import eleanor
import glob
from astropy.io import ascii,fits
from astropy.table import Table,join
import astropy.units as u
import bokeh
import sys
from scipy.signal import savgol_filter
from requests.exceptions import HTTPError
import matplotlib.gridspec as gridspec
from matplotlib import patches
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
import astropy.units as u
from astropy.visualization import SqrtStretch, LogStretch, ZScaleInterval, MinMaxInterval, SquaredStretch, AsinhStretch
from astroquery.mast import Catalogs
from astroquery.exceptions import ResolverError
#from requests.exceptions import HTTPError


plt.rcParams['font.size']=12

#ngc2516mems = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_allCGmems.dat')
mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/singlestars_68conf_m48.csv')
#mems = mems[(mems['proba'] > 0.68) & (~np.isnan(mems['bp_rp']))]
gaia_ids = mems['source_id'].values
gmags = mems['phot_g_mean_mag'].values
ra = mems['ra'].values*u.deg
dec = mems['dec'].values*u.deg

SC = SkyCoord(ra,dec)

#close = np.zeros(len(SC),dtype=bool)
count = np.zeros(len(SC))
#inds = np.zeros(len(SC),dtype=int)
closeids = []
closemags = []
closeseps = []
#closeinds = []
'''
for i in range(len(SC)):
    sep = SC[i].separation(SC).arcsec
    #indx = sep != 0
    close = (sep <= 21*2) & (sep != 0)
    count[i] = np.sum(close)
    if count[i] != 0:
        #inds[i] = ngc2516mems['source_id'][close].data
        closeids += [ngc2516mems['source_id'][close].data]
        #closeinds += [i]
        #print(i,ngc2516mems['source_id'][close].data)
    else:
        closeids += [[0]]
'''

for i in range(len(SC)):
    print(i)
#for i in range(496,len(SC)):
#for i in range(93,94):

    #####
    df = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/close_stars/'+np.str(gaia_ids[i])+'.csv')
    closeids += [df['closeids'].values.tolist()]
    closemags += [df['closemags'].values.tolist()]
    closeseps += [df['closeseps'].values.tolist()]

    #####

'''
    for attempt in range(10):
        try:
            qry = Gaia.query_object_async(SC[i],radius=21*2*u.arcsec)
            break
        except:
            print('Continuing.')
            continue
    qry['dist'] = qry['dist'] / (21./3600)
    nearby_bright = qry[qry['phot_g_mean_mag'] <= 17][1:]
    #sep = SC[i].separation(SC).arcsec
    #indx = sep != 0
    #close = (sep <= 21*2) & (sep != 0)
    count[i] = len(nearby_bright)
    #print(count[i],'nearby')
    if count[i] != 0:
        #inds[i] = ngc2516mems['source_id'][close].data
        closeids += [nearby_bright['source_id'].data.tolist()] #[ngc2516mems['source_id'][close].data]
        cm1=[]
        for j in range(len(nearby_bright)):
            cm1 += [np.float(format(nearby_bright['phot_g_mean_mag'].data[j],'.1f'))]
        #closemags += [np.round(nearby_bright['phot_g_mean_mag'].data,0).tolist()]
        closemags += [cm1]
        closeseps += [np.round(nearby_bright['dist'].data,1).tolist()]

        #closeinds += [i]
        #print(i,ngc2516mems['source_id'][close].data)
    else:
        closeids += [[-1]]
        closemags += [[-1]]
        closeseps += [[-1]]
    #print(i)
    #print(closemags)
    #print(closemags[i])
#####
    df = pd.DataFrame(data={'closeids':closeids[i],'closemags':closemags[i],'closeseps':closeseps[i]})
    df.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/close_stars/'+np.str(gaia_ids[i])+'.csv',index=False)
#####
#print(closeids, closemags, closeseps)
'''



def get_toi_lc(n,mems,size=(17,17),sigma_thresh=5,tpfplot=False):

    #srch = lk.search_tesscut('TIC '+np.str(ticids[n]))

    #tpfcoll = srch.download_all(cutout_size=size)
    #apers = np.zeros((len(tpfcoll),size[0],size[1]))
    #for i in range(len(tpfcoll)):
    #    apers[i,:,:] = tpfcoll[i].create_threshold_mask(sigma_thresh)
    #    if tpfplot:
    #        tpfcoll[i].plot(aperture_mask=apers[i])
    gaia_ids = mems['source_id']
    #S = eleanor.multi_sectors('all',gaia = bsourceids[n], tc=True)
    #S = eleanor.multi_sectors('all',gaia = 5290720867621791872, tc=True)
    for attempt in range(10):
        try:
            S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)
            break
        except:
            print('Continuing.')
            continue
    #S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)

    t = np.array([])
    f = np.array([])
    e = np.array([])

    pca_t = np.array([])
    pca_f = np.array([])
    pca_e = np.array([])

    psf_t = np.array([])
    psf_f = np.array([])
    psf_e = np.array([])

    raw_t = np.array([])
    raw_f = np.array([])
    raw_e = np.array([])
    for i in range(len(S)):
        #if S[i].position_on_chip[0] > 0 and S[i].position_on_chip[1] > 0:
        try:
            tempdata = eleanor.TargetData(S[i], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True)
            TPF = TessTargetPixelFile(S[i].cutout)
            #apers = TPF.create_threshold_mask(5)
            apers = np.zeros((31,31))
            apers[9:22,9:22]=tempdata.aperture

            #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
            #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

            #tempdata.get_lightcurve(apers)

            q0 = tempdata.quality == 0
            #t = np.append(t,tempdata.time[q0])
            #f = np.append(f,tempdata.corr_flux[q0]/np.nanmedian(tempdata.corr_flux[q0]))
            #e = np.append(e,tempdata.flux_err[q0]/tempdata.corr_flux[q0])
            pca_t = np.append(pca_t,tempdata.time[q0])
            pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
            pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])
            #psf_t = np.append(psf_t,tempdata.time[q0])
            #psf_f = np.append(psf_f,tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]))
            #psf_e = np.append(psf_e,tempdata.flux_err[q0]/tempdata.psf_flux[q0])
            #raw_t = np.append(raw_t,tempdata.time[q0])
            #raw_f = np.append(raw_f,tempdata.raw_flux[q0]/np.nanmedian(tempdata.raw_flux[q0]))


        #else:
        except ValueError:
            print('Skipping sector - not observed here.')
        #except urllib.error.HTTPError:
        #    print('Skipping sector - timeout')

    #lc=lk.LightCurve(time=t,flux=f,flux_err=e)
    lc_pca=lk.LightCurve(time=pca_t,flux=pca_f,flux_err=pca_e)
    #lc_psf=lk.LightCurve(time=psf_t,flux=psf_f,flux_err=psf_e)

    #lc_raw=lk.LightCurve(time=raw_t,flux=raw_f)

    #tpf_p=TPF.plot(aperture_mask=apers)
    #tpf_p.figure.savefig('/Users/bhealy/Documents/PhD_Thesis/TOI_Figs/TPFs/'+np.str(n)+'_'+np.str(Xmatch[n]['TOI'])+'_tpf.pdf')

    #return lc, TPF, apers
    return lc_pca, TPF, apers

def get_acf_period(lc,smth=500):
    #t = data['time']
    #y = (data['flux'] - 1) * 1e3
    #yerr = (data['flux_err']) * 1e3

    t = lc.time
    y = (lc.flux-1) * 1e3
    #print(y)
    yerr = lc.flux_err * 1e3

    delta_t = np.median(np.diff(t))
    new_t = np.arange(t.min(), t.max(), delta_t)
    y_interp = np.interp(new_t, t, y)
    emp_acorr = emcee.autocorr.function_1d(y_interp) #* np.var(y_interp)

    #emp_acorr = emcee.autocorr.function_1d(y) * np.var(y)

    new_f = np.zeros(len(new_t))

    for i in range(len(lc.time)):
        t_diffs = np.abs(new_t - t[i])
        minindx = np.argmin(t_diffs)
        new_f[minindx] = y[i]

    #emp_acorr = emcee.autocorr.function_1d(y_interp)
    emp_acorr = emcee.autocorr.function_1d(new_f)

    #tau = np.arange(len(emp_acorr)) * delta_t

    #print(smth)

    emp_acorr_smooth = gaussian_filter1d(emp_acorr,smth*delta_t)
    #emp_acorr_smooth = gaussian_filter1d(emp_acorr,150*delta_t)

    #print(emp_acorr_smooth)
    peakinds = find_peaks(emp_acorr_smooth)[0]
    peakvals = emp_acorr_smooth[peakinds]
    #print(peakinds)
    valinds = argrelmin(emp_acorr_smooth)[0]
    #print(valinds)

    tau = np.arange(len(emp_acorr_smooth)) * delta_t

    #emp_acorr_smooth[peakinds[0]] - emp_acorr_smooth[peakinds[1]] > 0
    if len(peakinds) == 0:
        peakheights = []
    else:
        peakheights = np.zeros(len(peakinds)-1)
        for i in range(len(peakinds)-1):
            peakheights[i] = np.mean((emp_acorr_smooth[peakinds[i]]-emp_acorr_smooth[valinds[i]], emp_acorr_smooth[peakinds[i]]-emp_acorr_smooth[valinds[i+1]]))

    #print(peakheights)

    if len(peakheights) != 0:

        maxindx = np.argmax(peakheights)

        if maxindx == 1:
            maxindx = 1
            #print('!')

        elif maxindx != 1:
            maxindx = 0

        period=tau[peakinds[maxindx]]
        #print(period)
        maxheight=peakheights[maxindx]
        maxpeakval=peakvals[maxindx]


        integerinds = np.array([2,3,4,5],dtype=int)
        integermults = [period * n for n in range(2,6)]

        tau_prev = period
        periodmults = np.array([])
        linfit_peakinds = np.array([],dtype=int)
        peakdiffs = np.zeros(len(peakinds)-1)
        n = 1
        #for i in range(len(integermults)):
        #    for j in range(len(peakinds)):
        #        if (np.abs(integermults[i] - tau[peakinds[j]])/period <= 0.1) & ((tau[peakinds[j]]-tau_prev) > 0.5*period):
        #            #print(i,j)
        #            periodmults = np.concatenate((periodmults,[tau[peakinds[j]]]))
        #            tau_prev = tau[peakinds[j]]
        #            linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(j)]))
        #            n += 1

        for i in range(len(integermults)):
            for j in range(len(peakinds)-1):
                peakdiffs[j] = np.abs(integermults[i] - tau[peakinds[j]])/period
            if np.min(peakdiffs) <= 0.2: # change to 0.2?
                jj = np.argmin(peakdiffs)
                #if peakheights[jj]>= 0.5 * maxheight #0.1:
                if (peakheights[jj]>= 0.5 * maxheight) | (peakheights[jj] >= 0.1):
                    periodmults = np.concatenate((periodmults,[tau[peakinds[jj]]]))
                #linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(jj)]))
                    linfit_peakinds = np.concatenate((linfit_peakinds,[integerinds[i]]))
                    n += 1
                #if (np.abs(integermults[i] - tau[peakinds[j]])/period <= 0.1) & ((tau[peakinds[j]]-tau_prev) > 0.5*period):
                    #print(i,j)
                    #periodmults = np.concatenate((periodmults,[tau[peakinds[j]]]))
                    #tau_prev = tau[peakinds[j]]
                    #linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(j)]))



        tauslice = tau[valinds[maxindx]:valinds[maxindx+1]]
        acfslice = emp_acorr_smooth[valinds[maxindx]:valinds[maxindx+1]]
        #plt.plot(tauslice,acfslice)
        halfheight = maxheight/2
        halfheightval = maxpeakval - halfheight
        #if maxheight <0:
        #    halfmax = maxheight*2
        #if maxheight < 0:
        #    halfmax = maxheight*2
        #print(halfmax)
        #print(acfslice)
        #absdiffs = np.abs(maxheight/2 - acfslice)
        absdiffs = np.abs(halfheightval - acfslice)
        #print(absdiffs)
        #print(absdiffs)
        #print(maxheight)
        #print(acfslice)
        minval1 = np.argmin(absdiffs)
        absdiffs=np.delete(absdiffs,minval1)
        minval2 = np.argmin(absdiffs)
        #print(minval1,minval2)
        #print(tauslice)
        #print(tauslice[minval1])
        #
        #sigma_from_fwhm = (np.abs(period - tauslice[minval1]) + np.abs(period - tauslice[minval2]))/2.35482004503
        sigma_from_fwhm = (np.abs(period - tauslice[minval1]) + np.abs(period - tauslice[minval2]))/2

        if (minval1 == minval2) | (minval1 == minval2+1) | (minval1 == minval2-1):
                #print('!!!')
                sigma_from_fwhm = (np.abs(period - tauslice[minval1]) * 2)/2 #2.35482004503
                #print(period)
                ##print(tauslice[minval1])
                #print(sigma_from_fwhm)

        #print(sigma_from_fwhm)

        linfit_peakinds = np.concatenate(([maxindx],linfit_peakinds))
        #linfit_peakinds += 1
        linfit_peakinds = np.concatenate(([0],linfit_peakinds))
        #if linfit_peakinds[1] == 2:
        #    linfit_peakinds = linfit_peakinds / 2
        #print(linfit_peakinds)
        if linfit_peakinds[1] == 0:
            linfit_peakinds[1] = 1

        e_period_hwhm = sigma_from_fwhm
        #print(np.str(n))
        if (n == 1) | (np.sum(np.diff(np.diff(linfit_peakinds)))) != 0:
            #print('X')
            periodmults = np.array([0,period])
            period_unc = e_period_hwhm #sigma_from_fwhm
            finalperiod = period
            e_period_mad = -1
            e_period_std = -1

        elif n > 1:
            #print('!')
            periodmults = np.concatenate(([0,period],periodmults))
            period_diffs = np.diff(periodmults)
            #print(period_diffs)
            #print(period_diffs)
            #finalperiod = np.median(period_diffs)
            finalperiod = np.polyfit(linfit_peakinds,periodmults,1)[0]
            #print(linfit_peakinds)
            #print(periodmults)


            #print(median_absolute_deviation(period_diffs))
            #print(np.std(period_diffs))
            #print(n)
            e_period_mad = 1.483* median_absolute_deviation(period_diffs) / np.sqrt(n-1)
            e_period_std = np.std(period_diffs)

            #print(median_absolute_deviation(period_diffs))
            #period_unc = 1.483* median_absolute_deviation(period_diffs) / np.sqrt(n-1)

            ###period_unc = e_period_mad
            period_unc = e_period_std

            if period_unc == 0:
                #print('Period unc too small')
                period_unc = e_period_hwhm #sigma_from_fwhm #np.std(period_diffs) / np.sqrt(n-1)
    else:
        #print('!')
        finalperiod = -1
        period_unc = -1
        e_period_hwhm = -1
        e_period_mad = -1
        e_period_std = -1
        peakindx = -1
        valinds = -1
        maxheight = -1
        periodmults = [0,0]


    return finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults

def get_lc_and_period(n,mems,size=(31,31),sigma_thresh=5,tpfplot=False,sec=0):
    #size=(31,31)
    #srch = lk.search_tesscut('TIC '+np.str(ticids[n]))

    #tpfcoll = srch.download_all(cutout_size=size)
    #apers = np.zeros((len(tpfcoll),size[0],size[1]))
    #for i in range(len(tpfcoll)):
    #    apers[i,:,:] = tpfcoll[i].create_threshold_mask(sigma_thresh)
    #    if tpfplot:
    #        tpfcoll[i].plot(aperture_mask=apers[i])
    if len(mems) > 1:
        gaia_ids = mems['source_id']
    else:
        gaia_ids = mems

    #####obstbl = Observations.query_criteria(objectname='Gaia DR2 '+np.str(gaia_ids[i]),provenance_name='CDIPS',radius=1*u.arcsec)
    try:
        ticid, tmag = Catalogs.query_object(objectname='Gaia DR2 '+np.str(gaia_ids[n]),catalog='TIC',radius=1*u.arcsec)[0]['ID','Tmag']
        lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_PATHOS/mastDownload/HLSP/*'+ticid+'*')
    except ResolverError:
        lcdirs = []

    #lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/PATHOS/mastDownload/HLSP/*'+np.str(gaia_ids[n])+'*')


    pca_t = np.array([])
    pca_f = np.array([])
    pca_e = np.array([])
    secs = []

    if len(lcdirs) != 0: #PROCEED WITH PATHOS
        print('PATHOS')
        #print(lcdirs)
        for d in lcdirs:
            #secs += [np.int(d.split('-')[1][2:])]
            #secs += [np.int(d.split('-')[2].split('_')[0][-2:])]
            secs += [np.int(d.split('-')[2].split('_')[0][1:])]

        secs = np.array(secs)

        all_maxheights = np.zeros(len(secs))
        all_smths = np.copy(all_maxheights)
        all_periods = np.copy(all_maxheights)
        all_period_uncs = np.copy(all_maxheights)
        all_e_period_hwhm = np.copy(all_maxheights)
        all_e_period_mad = np.copy(all_maxheights)

        lctbl = Table(data=[lcdirs,secs],names=['lcdirs','secs'])

        lctbl = lctbl[(lctbl['secs'] != 4) & (lctbl['secs'] != 1)]

        lctbl.sort('secs')
        secs = lctbl['secs'].data
        print(secs)
        #CHANGE KEYWORDS LATER
        for i in range(len(lctbl)):
            #lcfile = glob.glob(lctbl['lcdirs'][i]+'/*.fits')
            lcfile = glob.glob(lctbl['lcdirs'][i]+'/*.txt')

            pathos_data = ascii.read(lcfile[0],format='commented_header',header_start=-1)

            #hdul=fits.open(lcfile[0])

            #cdips_time = hdul[1].data['TMID_BJD'] - 2457000
            pathos_time = pathos_data['TIME[d]']

            #print(tmag)
            if tmag <= 7:
                flux_kywrd = 'AP4_FLUX_COR[e-/s]'
            elif (tmag > 7) & (tmag <= 9):
                flux_kywrd = 'AP3_FLUX_COR[e-/s]'
            elif (tmag > 9) & (tmag <= 10.5):
                flux_kywrd = 'AP2_FLUX_COR[e-/s]'
            elif (tmag > 10.5) & (tmag <= 13.5):
                flux_kywrd = 'PSF_FLUX_COR[e-/s]'
            elif tmag > 13.5:
                flux_kywrd = 'AP1_FLUX_COR[e-/s]'


            #cdips_flux = 2.512**(-hdul[1].data['PCA1'])
            #cdips_flux /= np.nanmedian(cdips_flux)
            pathos_flux = pathos_data[flux_kywrd]
            pathos_err = np.sqrt(pathos_flux)/pathos_flux

            pathos_bkg = pathos_data['SKY_LOCAL[e-/s]']
            pathos_bkglc = lk.LightCurve(pathos_time,pathos_bkg)
            badsky = pathos_bkglc.remove_outliers(5,return_mask=True)[1]

            pathos_flux /= np.nanmedian(pathos_flux)

            #cdips_iflflux = hdul[1].data['IFL1']
            #cdips_flux = cdips_iflflux

            #cdips_err = hdul[1].data['IFE1']
            #cdips_fracerr = cdips_err/cdips_iflflux

            #cdips_err = cdips_fracerr * cdips_flux
            #cdips_qual = hdul[1].data['IRQ1']
            pathos_qual = pathos_data['DQUALITY']

            #goodqual = cdips_qual != 'X'
            goodqual = (pathos_qual == 0) & (~badsky)

            single_lc_pca=lk.LightCurve(time=pathos_time[goodqual],flux=pathos_flux[goodqual],flux_err = pathos_err[goodqual])
            #print(np.min(single_lc_pca.flux),np.max(single_lc_pca.flux))
            if (np.min(single_lc_pca.flux) != np.max(single_lc_pca.flux)) & (np.min(single_lc_pca.flux) != 1.0):
                single_lc_pca = single_lc_pca.remove_nans().remove_outliers(3.5).flatten(481)

            #pca_t = np.append(pca_t,pathos_time[goodqual])
            #pca_f = np.append(pca_f,pathos_flux[goodqual])
            #pca_e = np.append(pca_e,pathos_err[goodqual])

            pca_t = np.append(pca_t,single_lc_pca.time)
            pca_f = np.append(pca_f,single_lc_pca.flux)
            pca_e = np.append(pca_e,single_lc_pca.flux_err)

            if len(single_lc_pca) == 0:
                all_maxheights[i] = -1
                all_periods[i] = -1
                all_period_uncs[i] = -1
                all_e_period_hwhm[i] = -1
                all_e_period_mad[i] = -1
            else:
                maxpow_period = single_lc_pca.to_periodogram().period_at_max_power.value
                smth = 500
                if (maxpow_period < 1) & (maxpow_period >= 0.1):
                    smth = 150
                elif maxpow_period > 3:
                    smth = 800
                elif maxpow_period > 6:
                    smth = 1200

                all_smths[i] = smth
                finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth)
                all_maxheights[i] = maxheight
                all_periods[i] = finalperiod
                all_period_uncs[i] = period_unc
                all_e_period_hwhm[i] = e_period_hwhm
                all_e_period_mad[i] = e_period_mad
            #all_smths[i] = smth

        #print(all_maxheights)


        timemasks = {
            's1' : np.where((pca_t >= 1325.293) & (pca_t <= 1353.178))[0],
            's2' : np.where((pca_t >= 1354.101) & (pca_t <= 1381.515))[0],
            's3' : np.where((pca_t >= 1385.897) & (pca_t <= 1406.292))[0],
            's4' : np.where((pca_t >= 1410.900) & (pca_t <= 1436.849))[0],
            's5' : np.where((pca_t >= 1437.826) & (pca_t <= 1464.400))[0],
            's6' : np.where((pca_t >= 1468.270) & (pca_t <= 1490.044))[0],
            's7' : np.where((pca_t >= 1491.626) & (pca_t <= 1516.085))[0],
            's8' : np.where((pca_t >= 1517.342) & (pca_t <= 1542.000))[0],
            's9' : np.where((pca_t >= 1543.216) & (pca_t <= 1568.475))[0],
            's10' : np.where((pca_t >= 1569.432) & (pca_t <= 1595.680))[0],
            's11' : np.where((pca_t >= 1596.772) & (pca_t <= 1623.891))[0],
            's12' : np.where((pca_t >= 1624.950) & (pca_t <= 1652.891))[0],
            's13' : np.where((pca_t >= 1653.915) & (pca_t <= 1682.357))[0]}

        bestindx = np.argmax(all_maxheights)

    #print(all_maxheights, secs)

        bestsector = np.int(secs[bestindx])
        #print(bestsector)
        key = 's' + np.str(bestsector)
        #print(key)
        #print(key)
        #print(pca_t)
        #print(timemasks[key])

        #print(gaia_ids[n], bestsector)
        for attempt in range(10):
            try:
                S = eleanor.Source(gaia=gaia_ids[n], sector = bestsector, tc=False)
                break
            except:
                print('Continuing.')
                continue
        #S = eleanor.Source(gaia=gaia_ids[n], sector = bestsector, tc=False)
        pos = S.position_on_chip

        tempdata = eleanor.TargetData(S, height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=False, aperture_mode='small')
                    #TPF = TessTargetPixelFile(S[bestindx].cutout)
        TPF = tempdata.tpf[0,:,:]
        apers = tempdata.aperture

        lc_pca=lk.LightCurve(time=pca_t,flux=pca_f,flux_err=pca_e)
        single_lc_pca=lk.LightCurve(time=pca_t[timemasks[key]],flux=pca_f[timemasks[key]],flux_err=pca_e[timemasks[key]])
        #print(single_lc_pca.time)
        single_lc_pca = single_lc_pca.remove_nans()#.remove_outliers()

        t_start = single_lc_pca.time[0] - 1
        t_end = single_lc_pca.time[-1] + 1

        #if 50 not in all_smths:
        #    final_smth = 350
        #elif 350 not in all_smths:
        #    final_smth = 50
        #else:
        #    final_smth = 350
        final_smth = all_smths[0]


        label = 1
        #finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=final_smth)
        finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(lc_pca,smth=final_smth)


    #S = eleanor.multi_sectors('all',gaia = bsourceids[n], tc=True)
    #S = eleanor.multi_sectors('all',gaia = 5290720867621791872, tc=True)
    #ELSE.......eleanor
    else:
        print('eleanor')
    #except requests.exceptions.HTTPError:
        #if sec == 0:
        for attempt in range(10):
            try:
                S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)
                break
            except:
                print('Continuing.')
                continue
        #S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)

    #    else:
    #        S = eleanor.Source(gaia=gaia_ids[n],tc=False,sector=sec)

        #5290723204083832448
        #S = eleanor.multi_sectors([2,3,4,5,6,7,8,9,10,11,12,13],gaia=gaia_ids[n], tc=True)
        if S[0].sector == 1:
            S = S[1:]

        t = np.array([])
        f = np.array([])
        e = np.array([])

        pca_t = np.array([])
        pca_f = np.array([])
        pca_e = np.array([])

        psf_t = np.array([])
        psf_f = np.array([])
        psf_e = np.array([])

        raw_t = np.array([])
        raw_f = np.array([])
        raw_e = np.array([])

        all_maxheights = np.zeros(len(S))
        all_periods = np.copy(all_maxheights)
        all_period_uncs = np.copy(all_maxheights)
        all_e_period_hwhm = np.copy(all_maxheights)
        all_e_period_mad = np.copy(all_maxheights)
        all_smths = np.copy(all_maxheights)

        #sectors = np.zeros(len(S)-1,dtype=int)

        for i in range(len(S)):
            #print(i)
            #if S[i].position_on_chip[0] > 0 and S[i].position_on_chip[1] > 0:
            try:
                tempdata = eleanor.TargetData(S[i], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True, aperture_mode='small')
                #TPF = TessTargetPixelFile(S[i].cutout)
                #apers = TPF.create_threshold_mask(5)
                #apers = np.zeros((31,31))
                #apers[9:22,9:22]=tempdata.aperture

                #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
                #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

                #tempdata.get_lightcurve(apers)
                #sectors[i] = S[i].sector
                q0 = tempdata.quality == 0

                pca_t = np.append(pca_t,tempdata.time[q0])
                pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
                pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])

                single_lc_pca=lk.LightCurve(time=tempdata.time[q0],flux=tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]),flux_err=tempdata.flux_err[q0]/tempdata.pca_flux[q0])
                single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

                middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])
                maxpow_period = single_lc_pca.to_periodogram().period_at_max_power.value
                smth = 500
                if (maxpow_period < 1) & (maxpow_period >= 0.1):
                    smth = 150
                elif maxpow_period > 3:
                    smth = 800
                elif maxpow_period > 6:
                    smth = 1200
                all_smths[i] = smth

                #finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth)
                finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth)

                all_maxheights[i] = maxheight
                all_periods[i] = finalperiod
                all_period_uncs[i] = period_unc
                all_e_period_hwhm[i] = e_period_hwhm
                all_e_period_mad[i] = e_period_mad

                #print(all_maxheights)
                #t = np.append(t,tempdata.time[q0])
                #f = np.append(f,tempdata.corr_flux[q0]/np.nanmedian(tempdata.corr_flux[q0]))
                #e = np.append(e,tempdata.flux_err[q0]/tempdata.corr_flux[q0])
                #pca_t = np.append(pca_t,tempdata.time[q0])
                #pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
                #pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])
                #psf_t = np.append(psf_t,tempdata.time[q0])
                #psf_f = np.append(psf_f,tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]))
                #psf_e = np.append(psf_e,tempdata.flux_err[q0]/tempdata.psf_flux[q0])
                #raw_t = np.append(raw_t,tempdata.time[q0])
                #raw_f = np.append(raw_f,tempdata.raw_flux[q0]/np.nanmedian(tempdata.raw_flux[q0]))


            #else:
            except ValueError:
                print('ValueError - skipping sector.')
                all_maxheights[i] = 0
            except HTTPError:
                print('Skipping sector - HTTPError.')
                continue

        bestindx = np.argmax(all_maxheights)
        pos = S[bestindx].position_on_chip
        timemasks = {
            's1' : np.where((pca_t >= 1325.293) & (pca_t <= 1353.178))[0],
            's2' : np.where((pca_t >= 1354.101) & (pca_t <= 1381.515))[0],
            's3' : np.where((pca_t >= 1385.897) & (pca_t <= 1406.292))[0],
            's4' : np.where((pca_t >= 1410.900) & (pca_t <= 1436.849))[0],
            's5' : np.where((pca_t >= 1437.826) & (pca_t <= 1464.400))[0],
            's6' : np.where((pca_t >= 1468.270) & (pca_t <= 1490.044))[0],
            's7' : np.where((pca_t >= 1491.626) & (pca_t <= 1516.085))[0],
            's8' : np.where((pca_t >= 1517.342) & (pca_t <= 1542.000))[0],
            's9' : np.where((pca_t >= 1543.216) & (pca_t <= 1568.475))[0],
            's10' : np.where((pca_t >= 1569.432) & (pca_t <= 1595.680))[0],
            's11' : np.where((pca_t >= 1596.772) & (pca_t <= 1623.891))[0],
            's12' : np.where((pca_t >= 1624.950) & (pca_t <= 1652.891))[0],
            's13' : np.where((pca_t >= 1653.915) & (pca_t <= 1682.357))[0]}
        #print(all_maxheights)
        #print(bestsector)
        #print(bestindx)
        #print(S[bestindx].cutout)

        tempdata = eleanor.TargetData(S[bestindx], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True, aperture_mode='small')
        #TPF = TessTargetPixelFile(S[bestindx].cutout)
        TPF = tempdata.tpf[0,:,:]
        apers = tempdata.aperture
        if sec == 0:
            bestsector = S[bestindx].sector
        else:
            bestsector = sec

        key = 's' + np.str(bestsector)

        #print('bestsector')
                #apers = TPF.create_threshold_mask(5)
        #apers = np.zeros((31,31))
        #apers[9:22,9:22]=tempdata.aperture
        #print(S[bestindx].gaia)
                #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
                #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

                #tempdata.get_lightcurve(apers)
        #sectors[i] = S[i].sector
        q0 = tempdata.quality == 0

        single_lc_pca=lk.LightCurve(time=pca_t[timemasks[key]],flux=pca_f[timemasks[key]],flux_err=pca_e[timemasks[key]])
        single_lc_pca = single_lc_pca.remove_nans().remove_outliers()
####################################################################################################

        #single_lc_pca=lk.LightCurve(time=pca_t[q0],flux=pca_f[q0],flux_err=pca_e[q0])

        ###single_lc_pca=lk.LightCurve(time=tempdata.time[q0],flux=tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]),flux_err=tempdata.flux_err[q0]/tempdata.pca_flux[q0])
        #single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

        middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
        goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
        single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

        #if 50 not in all_smths:
        #    final_smth = 350
        #elif 350 not in all_smths:
        #    final_smth = 50
        #else:
        #    final_smth = 350
        final_smth = all_smths[0]


        t_start = single_lc_pca.time[0] - 1
        t_end = single_lc_pca.time[-1] + 1
        #finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=final_smth)
        finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=final_smth)


        #lc=lk.LightCurve(time=t,flux=f,flux_err=e)
        lc_pca=lk.LightCurve(time=pca_t,flux=pca_f,flux_err=pca_e)
        #lc_psf=lk.LightCurve(time=psf_t,flux=psf_f,flux_err=psf_e)
        label=0
        #lc_raw=lk.LightCurve(time=raw_t,flux=raw_f)

        #tpf_p=TPF.plot(aperture_mask=apers)
        #tpf_p.figure.savefig('/Users/bhealy/Documents/PhD_Thesis/TOI_Figs/TPFs/'+np.str(n)+'_'+np.str(Xmatch[n]['TOI'])+'_tpf.pdf')

    #return lc, TPF, apers
    #return lc_pca, TPF, apers, finalperiod, period_unc, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos
    return lc_pca, TPF, apers, finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label, bestsector

#periods = np.zeros(len(toi_unique))
periods = np.zeros(len(mems))
period_uncs = np.copy(periods)
e_period_hwhm = np.copy(periods)
e_period_mad = np.copy(periods)
e_period_std = np.copy(periods)
label = np.zeros(len(mems),dtype=int)

#perTbl = Table(data=[gaia_ids,periods,period_uncs,e_period_hwhm,e_period_mad,e_period_std,label],names=['source_id','period','period_unc','e_period_hwhm','e_period_mad','e_period_std','PATHOS'])
#perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat')

#UNCOMMENT WHEN CREATED
perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/M48_ptbl_pathos_neighbors.dat')


#import OpenSSL
#import astroquery.mast.core

except_count = 0
total_failures = 0
#for n in range(371,372):
#for n in range(496,len(ngc2516mems)):

for n in range(len(mems)):

#for n in range(93,94):


#for n in range(256,257):
#for n in range(58,len(mems)):

    #for attempt in range(10):
    #    try:

#requests.exceptions.SSLError: HTTPSConnectionPool(host='mast.stsci.edu', port=443): Max retries exceeded with url: /api/v0.1/Download/file?uri=mast:HLSP/eleanor/postcards/s0010/3-3/hlsp_eleanor_tess_ffi_postcard-s0010-3-3-cal-1000-0052_tess_v2_pc.fits (Caused by SSLError(SSLError("bad handshake: SysCallError(-1, 'Unexpected EOF')",),))
#ssl.SSLError
    #for n in range(38,len(ngc2516mems)):
    #for n in range(28,29):
    #for n in range(0,3):
    #for n in range(165,168):

    gaia_id = np.str(mems['source_id'][n])
    gmag = np.str(np.round(mems['phot_g_mean_mag'][n],2))
    bprp = np.str(np.round(mems['bp_rp'][n],2))
    print(n, gaia_id)
    #toi = toi_unique[n]

    #planets = toi_dict[toi_unique[n]]

    #lc, TPF, apers = get_toi_lc(n,ticids)
    #lc, TPF, apers = get_toi_lc(n,ngc2516mems)

    #lc_pca, TPF, apers, periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos = get_lc_and_period(n,ngc2516mems)
    lc_pca, TPF, apers, periods[n], period_uncs[n], e_period_hwhm[n], e_period_mad[n], e_period_std[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label[n], bestsec = get_lc_and_period(n,mems)


    lc_pca = lc_pca.remove_nans().remove_outliers()
    single_indx = (lc_pca.time > t_start) & (lc_pca.time < t_end)
    single_lc_pca = lk.LightCurve(time=lc_pca.time[single_indx], flux = lc_pca.flux[single_indx], flux_err = lc_pca.flux_err[single_indx])

    ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
    ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
    ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

    #repl_lc = cleanlc


    #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
    #periods[n], period_uncs[n] = get_acf_period(repl_lc)

    #for i in range(len(planets)):
    #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
    #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
        #tpf_p=TPF.plot(aperture_mask=apers)
    #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)
    #print(lc_pca.time)

    pdg = lc_pca.to_periodogram()

    lctbl = lc_pca.to_table()
    lctbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/Figs/Light_Curves/'+np.str(n)+'_'+gaia_id+'_lc.txt',format='ascii',overwrite=True)

    pdgtbl = pdg.to_table()
    pdgtbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/Figs/Periodograms/'+np.str(n)+'_'+gaia_id+'_pdg.txt',format='ascii',overwrite=True)

    #fig2 = plt.figure(figsize=(15,20))
    #fig2 = plt.figure(figsize=(9,12))
    fig2 = plt.figure(figsize=(9,12))

    gs=gridspec.GridSpec(ncols=3, nrows=4,hspace=.3,wspace=.3)
    gss=gridspec.GridSpec(ncols=3, nrows=16,hspace=.5,wspace=.3)

    #gs=gridspec.GridSpec(ncols=7, nrows=4,hspace=.3,wspace=.3)

    ax1=plt.subplot(gs[0:1,0:])

    if label[n] == 1:
        lbl = 'PT'
    else:
        lbl = 'el'
    #ax1.set_title('P = '+np.str(np.round(periods[n],2))+'$\pm$'+np.str(np.round(period_uncs[n],2))+  '        $G$ = '+gmag,loc='left')
    ax1.set_title('P = '+np.str(np.round(periods[n],2))+'$\pm$'+np.str(np.round(period_uncs[n],2))+  '    $G$ = '+gmag +'    '+lbl,loc='left')

    ax1.set_title('Gaia DR2 '+ gaia_id,loc='right')
    #ax[0].text(.2,.9,gaia_id,horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    single_lc_pca.scatter(ax=ax1,c='black',s=3)

    ax2=plt.subplot(gs[1:2,0:])
    ax2.plot(tau, emp_acorr_smooth,color='blue',zorder=0)
    ax2.scatter(tau[peakinds],emp_acorr_smooth[peakinds],color='blue',s=50)
    ax2.scatter(tau[valinds],emp_acorr_smooth[valinds],color='orange',s=50)
    ax2.axvline(periods[n], color="green", alpha=0.9)
    for xx in range(1,6):
        ax2.axvline(periodmults[1]*xx,color='k',alpha=.75)#,ls='dashed')
    for yy in range(len(periodmults)):
        ax2.axvline(periodmults[yy],color='k',alpha=.5,ls='dashed')
    ax2.axvline(periodmults[1]*.8, color="k", alpha=0.5,ls="dashed")
    ax2.axvline(periodmults[1]*1.2, color="k", alpha=0.5,ls="dashed")
    ax2.set_xlim(0,25)
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Acorr')

    ax3=plt.subplot(gs[2:3,0:])
    lc_pca.scatter(ax=ax3,s=.3,c='black')
    ax3.axvline(t_start,color='k',ls='dashed',alpha=.5)
    ax3.axvline(t_end,color='k',ls='dashed',alpha=.5)

    ax4=plt.subplot(gs[3:4,0:1])
    #tpf_p=TPF.plot(aperture_mask=apers,ax=ax4,show_colorbar=False)
    interval = MinMaxInterval() #ZScaleInterval()
    stretch = SqrtStretch() #LogStretch()
    #ax4.imshow(stretch(interval(TPF)),origin='lower')
    ax4.imshow(stretch(interval(TPF)),origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)])
    #ax4.imshow(apers,origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)],alpha=.5, cmap=binary_r)

    #ax4.imshow(apers,alpha=0.5,cmap='spring_r')
    for i in range(apers.shape[0]):
        for j in range(apers.shape[1]):
            if apers[i, j]:
                #print('!')
                #print(j+np.int(pos[1]-TPF.shape[1]/2)-.5)
                #print(i+np.int(pos[0]-TPF.shape[0]/2)-.5)
                #plt.scatter(i,j,color='pink',alpha=.8,marker='s',s=50,facecolors='None')
                ax4.scatter(i+np.int(pos[0]-TPF.shape[0]/2)+.5,j+np.int(pos[1]- TPF.shape[1]/2)+.5,color='pink',alpha=.8,marker='s',s=20,facecolors='None')

                #ax4.add_patch(patches.Rectangle((j-.5, i-.5),
                #                                1, 1, color='pink', fill=False,
                #                                alpha=.75))
                #ax4.add_patch(patches.Rectangle((j+np.int(pos[1]- TPF.shape[1]/2)-.5, i+np.int(pos[0]-TPF.shape[0]/2)-.5),
                #                                1, 1, color='pink', fill=False,
                #                                alpha=.75))

    ax4.set_title('')

    ax5=plt.subplot(gs[3:4,1:])
    pdg.plot(view='period',ax=ax5,color='k',ylabel='')
    ax5.set_xlim(0,25)
    #ax5.text(.85,.9,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
    if np.float(bprp) < 1:
        bprp_clr = 'blue'
    else:
        bprp_clr = 'red'
    ax5.text(.5,1.05,'BP-RP = '+bprp,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color=bprp_clr)
    ax5.text(.85,1.05,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
    ax5.axvline(pdg.period_at_max_power.value, color="brown", alpha=0.75,ls="dashed")
    #if closeids[n] != 0:

    if -1 not in closeids[n]:

       for m in range(len(closeids[n])):
           if closeids[n][m] in mems['source_id'].values:
               clr = 'green'
           #elif (np.abs(gmags[n]-closemags[n][m]) < 1) & (closeseps[n][m] < 1):
           elif (closemags[n][m] - gmags[n] < 1) & (closeseps[n][m] < 1):
               clr = 'red'
           else:
               clr = 'darkorange'
           ax5.text(.6,.9 - (m*.075), np.str(closeids[n][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
           ax5.text(.85,.9 - (m*.075), np.str(closemags[n][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
           ax5.text(.95,.9 - (m*.075), np.str(closeseps[n][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)




            ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
            ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
            ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

            #repl_lc = cleanlc


            #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
            #periods[n], period_uncs[n] = get_acf_period(repl_lc)

            #for i in range(len(planets)):
            #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
            #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
            #tpf_p=TPF.plot(aperture_mask=apers)
            #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)

    #if 0 not in closeids[0]:
    #    for m in range(len(closeids[0])):
    #        if closeids[0][m] in ngc2516mems['source_id']:
    #            clr = 'green'
    #        else:
    #            clr = 'darkorange'
    #        ax5.text(.6,.9 - (m*.075), np.str(closeids[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
    #        ax5.text(.85,.9 - (m*.075), np.str(closemags[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
    #        ax5.text(.95,.9 - (m*.075), np.str(closeseps[0][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)

    #fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/Figs/DVRs/'+np.str(n)+'_'+gaia_id+'_dvr.pdf',overwrite=True,bbox_inches='tight')
    fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/PATHOS_LC_DVRs_neighbors/'+np.str(n)+'_'+gaia_id+'_dvr.pdf',overwrite=True,bbox_inches='tight')
    plt.close('all')

    fig3 = plt.figure(figsize=(8,24))

    if -1 not in closeids[n]:

       for m in range(len(closeids[n])):
           #lc_pca, TPF, apers, periods[n], period_uncs[n], e_period_hwhm[n], e_period_mad[n], e_period_std[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label[n], bstsc = get_lc_and_period(0,[closeids[n][m]],sec=bestsec)
           #print(n,m)
           try:
               #ticid, tmag, gaiabp, gaiarp = Catalogs.query_object(objectname='Gaia DR2 '+np.str(closeids[n][m]),catalog='TIC',radius=1*u.arcsec)[0]['ID','Tmag','gaiabp','gaiarp']
               ticid, tmag = Catalogs.query_object(objectname='Gaia DR2 '+np.str(closeids[n][m]),catalog='TIC',radius=1*u.arcsec)[0]['ID','Tmag']
               lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_PATHOS/mastDownload/HLSP/*'+ticid+'*')
               #gaia_bp_rp = gaiabp - gaiarp
           except ResolverError:
               lcdirs = []
               #gaia_bp_rp = 99

           #ticid, tmag = Catalogs.query_object(objectname='Gaia DR2 '+np.str(closeids[n][m]),catalog='TIC',radius=1*u.arcsec)[0]['ID','Tmag']
           #lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/PATHOS/mastDownload/HLSP/*'+ticid+'*')

           #lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/PATHOS/mastDownload/HLSP/*'+np.str(closeids[n][m])+'*')

           #if closeids[n][m] in gaia_ids:
           if len(lcdirs) != 0:
               print('PATHOS')
               #print(closeids[n][m])

               pca_t = np.array([])
               pca_f = np.array([])
               pca_e = np.array([])
               secs = []

               for d in lcdirs:
                  #secs += [np.int(d.split('-')[1][2:])]
                   secs += [np.int(d.split('-')[2].split('_')[0][1:])]

               secs = np.array(secs)

               all_maxheights = np.zeros(len(secs))
               all_smths = np.copy(all_maxheights)
               all_periods = np.copy(all_maxheights)
               all_period_uncs = np.copy(all_maxheights)
               all_e_period_hwhm = np.copy(all_maxheights)
               all_e_period_mad = np.copy(all_maxheights)

               lctbl = Table(data=[lcdirs,secs],names=['lcdirs','secs'])

               lctbl = lctbl[(lctbl['secs'] != 4) & (lctbl['secs'] != 1)]

               lctbl.sort('secs')
               secs = lctbl['secs'].data
               if np.int(bestsec) in secs:
                   #print('!')
                   sindx = secs == bestsec
               elif len(secs) != 0:
                   sindx = secs == secs[0]
               #else:
            #       S = eleanor.Source(gaia=closeids[n][m], sector = np.int(bestsec), tc=False)
            #       tempdata = eleanor.TargetData(S, height=13, width=13, bkg_size=31, do_psf=False, do_pca=True, crowded_field=True)
            #       q0 = tempdata.quality == 0
            #       single_lc_pca = lk.LightCurve(time=tempdata.time[q0], flux = tempdata.pca_flux[q0], flux_err = tempdata.flux_err[q0])


                  #lc_pca = lc_pca.remove_nans().remove_outliers()
                  #single_indx = (lc_pca.time > t_start) & (lc_pca.time < t_end)

               #print(lctbl['lcdirs'][sindx].data[0])
               #print(lctbl['lcdirs'][sindx].data[0]+'/*.fits')

               #lcfile = glob.glob(lctbl['lcdirs'][sindx][0]+'/*.fits')
               lcfile = glob.glob(lctbl['lcdirs'][sindx][0]+'/*.txt')

               #hdul=fits.open(lcfile[0])
               pathos_data = ascii.read(lcfile[0],format='commented_header',header_start=-1)
               pathos_time = pathos_data['TIME[d]']

               if tmag <= 7:
                   flux_kywrd = 'AP4_FLUX_COR[e-/s]'
               elif (tmag > 7) & (tmag <= 9):
                   flux_kywrd = 'AP3_FLUX_COR[e-/s]'
               elif (tmag > 9) & (tmag <= 10.5):
                   flux_kywrd = 'AP2_FLUX_COR[e-/s]'
               elif (tmag > 10.5) & (tmag <= 13.5):
                   flux_kywrd = 'PSF_FLUX_COR[e-/s]'
               elif tmag > 13.5:
                   flux_kywrd = 'AP1_FLUX_COR[e-/s]'

               #cdips_time = hdul[1].data['TMID_BJD'] - 2457000

               #cdips_flux = 2.512**(-hdul[1].data['PCA1'])
               #cdips_flux /= np.nanmedian(cdips_flux)

               pathos_flux = pathos_data[flux_kywrd]
               pathos_err = np.sqrt(pathos_flux)/pathos_flux

               pathos_bkg = pathos_data['SKY_LOCAL[e-/s]']
               pathos_bkglc = lk.LightCurve(pathos_time,pathos_bkg)
               badsky = pathos_bkglc.remove_outliers(5,return_mask=True)[1]

               pathos_flux /= np.nanmedian(pathos_flux)


               #cdips_iflflux = hdul[1].data['IFL1']
               #cdips_flux = cdips_iflflux

               #cdips_err = hdul[1].data['IFE1']
               #cdips_fracerr = cdips_err/cdips_iflflux

               #cdips_err = cdips_fracerr * cdips_flux
               #cdips_qual = hdul[1].data['IRQ1']
              # goodqual = cdips_qual != 'X'

               pathos_qual = pathos_data['DQUALITY']
               goodqual = (pathos_qual == 0) & (~badsky)

              # pca_t = np.append(pca_t,pathos_time[goodqual])
              # pca_f = np.append(pca_f,pathos_flux[goodqual])
              # pca_e = np.append(pca_e,pathos_err[goodqual])

               #single_lc_pca=lk.LightCurve(time=cdips_time[goodqual],flux=cdips_flux[goodqual],flux_err=cdips_err[goodqual])
               single_lc_pca=lk.LightCurve(time=pathos_time[goodqual],flux=pathos_flux[goodqual],flux_err = pathos_err[goodqual])

               if (np.min(single_lc_pca.flux) != np.max(single_lc_pca.flux)) & (np.min(single_lc_pca.flux) != 1.0):
                   single_lc_pca = single_lc_pca.remove_nans().remove_outliers(3.5).flatten(481)

               pca_t = np.append(pca_t,single_lc_pca.time)
               pca_f = np.append(pca_f,single_lc_pca.flux)
               pca_e = np.append(pca_e,single_lc_pca.flux_err)
           else:
               for attempt in range(10):
                   try:
                       S = eleanor.Source(gaia=closeids[n][m], sector = np.int(bestsec), tc=False)
                       break
                   except:
                       print('Continuing.')
                       continue
               tempdata = eleanor.TargetData(S, height=13, width=13, bkg_size=31, do_psf=False, do_pca=True, aperture_mode='small')
               q0 = tempdata.quality == 0

               #lc_pca = lc_pca.remove_nans().remove_outliers()
               #single_indx = (lc_pca.time > t_start) & (lc_pca.time < t_end)
               single_lc_pca = lk.LightCurve(time=tempdata.time[q0], flux = tempdata.pca_flux[q0], flux_err = tempdata.flux_err[q0])

           single_lc_pca = single_lc_pca.normalize().remove_outliers()
           pdg = single_lc_pca.to_periodogram()

           lctbl = single_lc_pca.to_table()
           lctbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/Figs/Neighbor_Light_Curves/'+np.str(n)+'_'+gaia_id+'_lc.txt',format='ascii',overwrite=True)

           pdgtbl = pdg.to_table()
           pdgtbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/Figs/Neighbor_Periodograms/'+np.str(n)+'_'+gaia_id+'_pdg.txt',format='ascii',overwrite=True)

           ax = plt.subplot(gss[0+m*2:1+m*2,0:])
           axx = plt.subplot(gss[0+m*2+1:1+m*2+1,0:])

          # ax.set_title(np.str(closeids[n][m]))
           smth = 500
           if (periods[n] < 1) & (periods[n] >= 0.1):
               smth = 150
           elif periods[n] > 3:
               smth = 800
           elif periods[n] > 6:
               smth = 1200
           #finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(single_lc_pca,smth=smth)
           finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(lc_pca,smth=smth)


           #single_lc_pca.scatter(ax=plt.subplot(gs[0+m:1+m,0:]),c='black',s=3)
           single_lc_pca.scatter(ax=ax,c='black',s=1)
           ax.set_xlabel('')
           ax.set_ylabel('')
           axx.plot(ta, emp_acorr_smoot,color='blue',zorder=0)
           axx.scatter(ta[peakind],emp_acorr_smoot[peakind],color='blue',s=10)
           axx.scatter(ta[valind],emp_acorr_smoot[valind],color='orange',s=10)
           axx.axvline(finalperio, color="green", alpha=0.9)
           for xx in range(1,6):
               axx.axvline(periodmult[1]*xx,color='k',alpha=.75)#,ls='dashed')
           for yy in range(len(periodmult)):
               axx.axvline(periodmult[yy],color='k',alpha=.5,ls='dashed')
           axx.axvline(periodmult[1]*.8, color="k", alpha=0.5,ls="dashed")
           axx.axvline(periodmult[1]*1.2, color="k", alpha=0.5,ls="dashed")
           axx.set_xlim(0,10)
           #axx.text(0.2,0.8, np.str(closeids[n][m]),transform=axx.transAxes)
           axx.text(0.3,0.8, np.str(closeids[n][m]),transform=axx.transAxes)
           #axx.text(0.65,0.8, np.str(np.round(gaia_bp_rp,2)),transform=axx.transAxes)
           axx.text(0.75,0.8, np.str(np.round(finalperio,2)),transform=axx.transAxes,color='blue')
           axx.text(0.85,0.8, np.str(np.round(single_lc_pca.to_periodogram().period_at_max_power.value,2)),color='brown',transform=axx.transAxes)


    fig3.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/PATHOS_LC_DVRs_neighbors/'+np.str(n)+'_'+gaia_id+'_neighbors.pdf',overwrite=True,bbox_inches='tight')

    #plt.subplot(gs[1:2])
    plt.close('all')
    perTbl['period'][n] = periods[n]
    perTbl['period_unc'][n] = period_uncs[n]
    perTbl['e_period_hwhm'][n] = e_period_hwhm[n]
    perTbl['e_period_mad'][n] = e_period_mad[n]
    perTbl['e_period_std'][n] = e_period_std[n]
    perTbl['PATHOS'][n] = label[n]

    #perTbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat',format='ascii',overwrite=True)
    perTbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/M48_ptbl_pathos_neighbors.dat',format='ascii',overwrite=True)


        #    break

        #except:
        #    except_count += 1
        #    print('Continuing.')
        #    continue
#####################################
        #except IndexError:
    #        except_count += 1
            #continue
    #        break
        #    print('NRError - continuing.')

        #except HTTPError:
        #    except_count += 1
        #    print('HTTPError - continuing.')
        #    continue


    #    except OpenSSL.SSL.SysCallError(-1, 'Unexpected EOF'):
    #        except_count += 1
    #        print('SSL Error - continuing.')





        #except:
        #    except_count += 1
        #    print('Error - continuing.')
        #    continue
        #except ValueError:
        #    print('ValueError: continuing.')
        #    continue
#        except HTTPError:
#            except_count += 1
#            print('Timeout - continuing.')
#            continue
#        except IndexError:
#            except_count += 1
#            print('IndexError - continuing.')
#            continue

    #else:
    #    total_failures += 1
    #    print('Tried 10 times. Moving on.')

#print(except_count)
#print(total_failures)
#perTbl = Table(data=[periods,period_uncs],names=['Period','Period_unc'])
#perTbl.write('/Users/bhealy/NGC_2516/NGC_2516_ptbl.dat',format='ascii',overwrite=True)
