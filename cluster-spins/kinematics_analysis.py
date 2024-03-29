import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import binned_statistic
import pandas as pd
#from uncertainties.umath import *
from uncertainties import unumpy
from scipy.stats import gaussian_kde
import corner
from astroquery.simbad import Simbad
from astroquery.exceptions import TableParseError
from IPython.display import display, Math

#Our tangential velocity convention:
#Positive -> clockwise
#Negative -> counterclockwise

plt.rcParams['font.size']= 18

#def kinematics(stars,cluster):
def bv_to_bprp(bv):
    #return 0.0187 + 1.6814*bv - 0.3357*bv**2 +0.1117*bv**3
    return 0.0545 + 1.4939*bv - 0.6274*bv**2 +0.2957*bv**3

def bprp_to_bv(bprp):
    bvs = np.array([])
    poly = np.poly1d([0.2957, -0.6274, 1.4939, 0.0545])
    #poly = np.polynomial.Polynomial([0.2957, -0.6274, 1.4939, 0.0545])
    for x in bprp:
        subtracted_poly = poly - x
        roots = (poly - x).roots
        real = np.imag(roots) == 0
        #print(roots,real)
        bv = np.real(roots[real])[0]
        bvs = np.append(bvs,bv)
    return bvs

def edr3ToICRF (pmra ,pmdec ,ra ,dec ,G):
    """
    Input: source position , coordinates ,
    and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    if G >=13:
        return pmra , pmdec
    import numpy as np
    def sind(x):
        return np.sin(np.radians(x))
    def cosd(x):
        return np.cos(np.radians(x))

    table1=""" 0.0 9.0 18.4 33.8 -11.3
    9.0 9.5 14.0 30.7 -19.4
    9.5 10.0 12.8 31.4 -11.8
    10.0 10.5 13.6 35.7 -10.5
    10.5 11.0 16.2 50.0 2.1
    11.0 11.5 19.4 59.9 0.2
    11.5 11.75 21.8 64.2 1.0
    11.75 12.0 17.7 65.6 -1.9
    12.0 12.25 21.3 74.8 2.1
    12.25 12.5 25.7 73.6 1.0
    12.5 12.75 27.3 76.6 0.5
    12.75 13.0 34.9 68.9 -2.9 """

    table1 = np.fromstring(table1, sep=' ').reshape((12 ,5)).T
    Gmin = table1[0]
    Gmax = table1[1]
    #pick the appropriate omegaXYZ for the source’s magnitude:
    omegaX = table1[2][( Gmin <=G)&(Gmax >G)][0]
    omegaY = table1[3][( Gmin <=G)&(Gmax >G)][0]
    omegaZ = table1[4][( Gmin <=G)&(Gmax >G)][0]
    pmraCorr = -1* sind(dec)*cosd(ra)*omegaX -sind(dec)*sind(ra)*omegaY + cosd(dec)*omegaZ
    pmdecCorr = sind(ra)*omegaX -cosd(ra)*omegaY

    return pmra -pmraCorr /1000. , pmdec - pmdecCorr /1000.

def drop_y(df):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith('_y')]
    print(to_drop)
    df.drop(to_drop, axis=1, inplace=True)

def drop(df,suffix):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith(suffix)]
    print(to_drop)
    df.drop(to_drop, axis=1, inplace=True)


#t.logpdf(1,loc=vperp[0],scale=e_vperp[0],df=6)
def log_likelihood(theta, vperp, e_vperp, posangs):
    posang_c, vrot, v0, siglos = theta
    vc = v0 + vrot * np.sin(posangs - posang_c)
    #vc = v0 + vrot * np.sin((posangs - posang_c)*np.pi/180)
    A = np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2))
    B = (vperp - vc)**2 / (2*(siglos**2 + e_vperp**2))
    #return #-np.sum(np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2)) - ((vperp - vc)**2 / (2*siglos**2 + e_vperp**2)))
    return -1 * np.sum(A + B)

#def log_likelihood(theta, vperp, e_vperp, posangs):
#    posang_c, vrot, v0, siglos = theta
#    loglikes = np.zeros(len(vperp))
#    trial_varr = np.arange(-10,10,0.01)
#    for i in range(len(vperp)):
#        vc = v0 + vrot * np.sin(posangs[i] - posang_c)
#        gauss1 = norm.pdf(trial_varr,loc=0,scale=siglos)
        #gauss2 = norm.pdf(trial_varr,loc=0,scale=e_vperp[i])
#        t1 = t.pdf(trial_varr,loc=0,scale=e_vperp[i],df=6)
        #conv = np.convolve(gauss1,gauss2,mode='same')
#        conv = np.convolve(gauss1,t1,mode='same')
#        conv /= np.trapz(conv,trial_varr)
#        eval_conv = np.interp(vperp[i]-vc,trial_varr,conv)
#        loglikes[i] = np.log(eval_conv)
#    return np.sum(loglikes)

    #A = np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2))
    #B = (vperp - vc)**2 / (2*(siglos**2 + e_vperp**2))
    #return #-np.sum(np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2)) - ((vperp - vc)**2 / (2*siglos**2 + e_vperp**2)))
    #return -1 * np.sum(A + B)

#def log_likelihood(theta, vperp, e_vperp, posangs):
#    posang_c, vrot, v0, siglos = theta
#    vc = v0 + vrot * np.sin(posangs - posang_c)

#    gauss_lklhd = norm.pdf(vc,loc=v0,scale=siglos)
#    t_lklhd = t.pdf(vc,loc=vperp,scale=e_vperp,df=6)
#    conv_lklhd = np.convolve(gauss_lklhd,t_lklhd,mode='same')
    #vc = v0 + vrot * np.sin((posangs - posang_c)*np.pi/180)
#    A = np.log(np.sqrt(2*np.pi*siglos**2))
#    B = (vperp - vc)**2 / (siglos**2)
#    C = t.logpdf(vc,loc=vperp,scale=e_vperp,df=6)
    #C = t.logpdf(vc,loc=vperp,scale=np.sqrt(siglos**2+e_vperp**2),df=6)
    #return np.sum(C)
 #   return np.sum(np.log(conv_lklhd))

    #return #-np.sum(np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2)) - ((vperp - vc)**2 / (2*siglos**2 + e_vperp**2)))
#    return -0.5 * np.sum(A + B) + np.sum(C)
'''
def log_prior(theta):
    posang_c, vrot, v0, siglos = theta
    #if -np.pi <= posang_c <= np.pi and 0 <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
    #if 0 <= posang_c < 2*np.pi and 0 <= vrot <= 3. and 15 <= v0 <= 35 and 0 <= siglos <= 5:

    ###if 5.207 - np.pi <= posang_c < 5.207 + np.pi and 0 <= vrot <= 3. and 0 <= v0 <= 50 and 0 <= siglos <= 5:
    #if  0 <= posang_c < 2*np.pi and 0 <= vrot <= 3. and 0 <= v0 <= 50 and 0 <= siglos <= 5:
    if  0 <= posang_c < 2*np.pi and 0 <= vrot <= 50. and -50 <= v0 <= 50 and 0 <= siglos <= 50:

    #if 0 <= vrot <= 3. and 15 <= v0 <= 35 and 0 <= siglos <= 5:
        lp = 0
    else:
        lp = -np.inf

        #return 0.0
        #lp = 0

    #GAUSS PRIOR ON THETA
    #mn= 6.3
    #sgma = 1.3
    #lp -= 0.5*((posang_c - mn)/sgma)**2

    #if 0 <= posang_c < np.pi and -3 <= vrot <= 3. and 15 <= v0 <= 35 and 0 < siglos <= 5:
    #if -180 <= posang_c <= 180 and 0. <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
    #return -np.inf
    return lp
'''

def log_prior(theta, pc0):
    posang_c, vrot, v0, siglos = theta
    #if -np.pi <= posang_c <= np.pi and 0 <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
    #if 0 <= posang_c < 2*np.pi and 0 <= vrot <= 3. and 15 <= v0 <= 35 and 0 <= siglos <= 5:

    ###if 5.207 - np.pi <= posang_c < 5.207 + np.pi and 0 <= vrot <= 3. and 0 <= v0 <= 50 and 0 <= siglos <= 5:
    #if  0 <= posang_c < 2*np.pi and 0 <= vrot <= 3. and 0 <= v0 <= 50 and 0 <= siglos <= 5:
    #if  0 <= posang_c < 2*np.pi and 0 <= vrot <= 50. and -50 <= v0 <= 50 and 0 <= siglos <= 50:
    if  pc0 - np.pi <= posang_c < pc0 + np.pi and 0 <= vrot <= 50. and -50 <= v0 <= 50 and 0 <= siglos <= 50:

    #if 0 <= vrot <= 3. and 15 <= v0 <= 35 and 0 <= siglos <= 5:
        lp = 0
    else:
        lp = -np.inf

        #return 0.0
        #lp = 0

    #GAUSS PRIOR ON THETA
    #mn= 6.3
    #sgma = 1.3
    #lp -= 0.5*((posang_c - mn)/sgma)**2

    #if 0 <= posang_c < np.pi and -3 <= vrot <= 3. and 15 <= v0 <= 35 and 0 < siglos <= 5:
    #if -180 <= posang_c <= 180 and 0. <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
    #return -np.inf
    return lp

def log_probability(theta, vperp, e_vperp, posangs, pc0):
    lp = log_prior(theta, pc0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, vperp, e_vperp, posangs)
###################################################################
def log_likelihood_r(theta, vrad, e_vrad):
    v0_r, sigr = theta
    vc = v0_r #+ vrot * np.sin(posangs - posang_c)
    #vc = v0 + vrot * np.sin((posangs - posang_c)*np.pi/180)
    A = np.log(2*np.pi*np.sqrt(sigr**2 + e_vrad**2))
    B = (vrad - vc)**2 / (2*(sigr**2 + e_vrad**2))
    #return #-np.sum(np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2)) - ((vperp - vc)**2 / (2*siglos**2 + e_vperp**2)))
    return -1 * np.sum(A + B)

def log_prior_r(theta):
    v0_r, sigr = theta
    if -5 <= v0_r <= 5 and 0 <= sigr <= 5:
    #if -180 <= posang_c <= 180 and 0. <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
        return 0.0
    return -np.inf

def log_probability_r(theta, vrad, e_vrad):
    lp = log_prior_r(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_r(theta, vrad, e_vrad)

def log_likelihood_t(theta, vtan, e_vtan):
    v0_t, sigt = theta
    vc = v0_t #+ vrot * np.sin(posangs - posang_c)
    #vc = v0 + vrot * np.sin((posangs - posang_c)*np.pi/180)
    A = np.log(2*np.pi*np.sqrt(sigt**2 + e_vtan**2))
    B = (vtan - vc)**2 / (2*(sigt**2 + e_vtan**2))
    #return #-np.sum(np.log(2*np.pi*np.sqrt(siglos**2 + e_vperp**2)) - ((vperp - vc)**2 / (2*siglos**2 + e_vperp**2)))
    return -1 * np.sum(A + B)

def log_prior_t(theta):
    v0_t, sigt = theta
    if -5 <= v0_t <= 5 and 0 <= sigt <= 5:
    #if -180 <= posang_c <= 180 and 0. <= vrot <= 1. and 20 <= v0 <= 30 and 0 <= siglos <= 5:
        return 0.0
    return -np.inf

def log_probability_t(theta, vtan, e_vtan):
    lp = log_prior_t(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_r(theta, vtan, e_vtan)

'''def query_simbad(targets):
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otype')

    sbindx = []
    singindx = []

    for i in range(len(targets)):
        qry = customSimbad.query_object('Gaia DR2 '+np.str(targets['source_id'][i]))
        try:
            print(i,qry['OTYPE'].data[0])
            if (qry['OTYPE'].data[0] == b'LPV*') or (qry['OTYPE'].data[0] == b'EB*') or (qry['OTYPE'].data[0] == b'SB*') or (qry['OTYPE'].data[0] == b'PulsV*delSct') or (qry['OTYPE'].data[0] == b'PulsV*') or (qry['OTYPE'].data[0] == 'LPV*') or (qry['OTYPE'].data[0] == 'EB*') or (qry['OTYPE'].data[0] == 'SB*') or (qry['OTYPE'].data[0] == 'PulsV*delSct') or (qry['OTYPE'].data[0] == 'PulsV*') or (qry['OTYPE'].data[0] == b'Eruptive*') or (qry['OTYPE'].data[0] == 'Eruptive') or (qry['OTYPE'].data[0] == 'CataclyV*') or (qry['OTYPE'].data[0] == b'CataclyV*'):
                #print('!')
                sbindx += [i]
            else:
                singindx += [i]
        except TypeError:
            sbindx += [i]

    targets = targets.loc[singindx].reset_index(drop=True)
    print(len(targets))
    return targets'''

def query_simbad(targets):
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otype')

    sbindx = []
    singindx = []

    for i in range(len(targets)):
        #qry = customSimbad.query_object('Gaia DR2 '+np.str(targets['source_id'][i]))
        try:
            qry = customSimbad.query_object('Gaia DR2 '+np.str(targets['source_id'][i]))
            print(i,qry['OTYPE'].data[0])
            if (qry['OTYPE'].data[0] == b'LPV*') or (qry['OTYPE'].data[0] == b'EB*') or (qry['OTYPE'].data[0] == b'SB*') or (qry['OTYPE'].data[0] == b'PulsV*delSct') or (qry['OTYPE'].data[0] == b'PulsV*') or (qry['OTYPE'].data[0] == 'LPV*') or (qry['OTYPE'].data[0] == 'EB*') or (qry['OTYPE'].data[0] == 'SB*') or (qry['OTYPE'].data[0] == 'PulsV*delSct') or (qry['OTYPE'].data[0] == 'PulsV*') or (qry['OTYPE'].data[0] == b'Eruptive*') or (qry['OTYPE'].data[0] == 'Eruptive') or (qry['OTYPE'].data[0] == 'CataclyV*') or (qry['OTYPE'].data[0] == b'CataclyV*'):
                #print('!')
                sbindx += [i]
            else:
                singindx += [i]
        except TypeError:
            sbindx += [i]
        except TableParseError:
            singindx += [i]

    targets = targets.loc[singindx].reset_index(drop=True)
    print(len(targets))
    return targets

def calc_outlying_ruwe(targets,bw=0.2,percentile=0.99):
    gkde = gaussian_kde(targets['ruwe'],bw_method=bw)

    arry = np.linspace(0,3,10000)
    good_ruwe_arry = arry < 1.1

    #plt.plot(arry,gkde.pdf(arry))
    #plt.plot(arry[good_ruwe_arry],gkde.pdf(arry[good_ruwe_arry]))
    #print(np.argmax(gkde.pdf(arry)), arry[np.argmax(gkde.pdf(arry))])

    amax_ruwe = np.argmax(gkde.pdf(arry))
    reflection = gkde.pdf(arry[:amax_ruwe])[::-1]

    full_ruwe = np.concatenate([arry[:amax_ruwe+1], arry[amax_ruwe+1:2*amax_ruwe+1], arry[2*amax_ruwe+1:]])

    full_prob = np.concatenate([gkde.pdf(arry[:amax_ruwe+1]), reflection, np.zeros(len(arry[2*amax_ruwe+1:]))])

    full_prob /= np.trapz(full_prob,full_ruwe)

    for i in range(len(full_prob-1)):
        integ = np.trapz(full_prob[0:i],full_ruwe[0:i])
        if integ > percentile:
            #print(i)
            ruwe_99 = i
            break

            #print(full_ruwe[ruwe_99])
    return full_ruwe[ruwe_99]

def make_kinematics_tables(merge_gaia, spectro_data, CRV=None, gdr2=False):

    #cra = np.mean(merge_gaia['ra'])
    #cdec = np.mean(merge_gaia['dec'])

    cra = np.mean(merge_gaia['ra'])
    cdec = np.mean(merge_gaia['dec'])

    for i in range(len(merge_gaia)):
        new_pm_ra, new_pm_dec = edr3ToICRF(merge_gaia.loc[i,'pmra'],merge_gaia.loc[i,'pmdec'],merge_gaia.loc[i,'ra'],merge_gaia.loc[i,'dec'],merge_gaia.loc[i,'phot_g_mean_mag'])
        merge_gaia.loc[i,['pmra','pmdec']] = [new_pm_ra, new_pm_dec]

    cpmra = np.mean(merge_gaia['pmra'])
    cpmdec = np.mean(merge_gaia['pmdec'])

    cdist = 1e3/np.mean(merge_gaia['parallax'])

    u_cpmra = unumpy.uarray(cpmra,0)
    u_cpmdec = unumpy.uarray(cpmdec,0)
    u_cdist = unumpy.uarray(cdist,0)

    good_simbad = query_simbad(merge_gaia)

    if not gdr2:
        ruwe_cutoff = calc_outlying_ruwe(good_simbad)
        print(ruwe_cutoff)
        good_ruwe = good_simbad[good_simbad['ruwe'] < ruwe_cutoff].reset_index(drop=True)
    else:
        good_ruwe=good_simbad

    if CRV != None:
        crv = unumpy.uarray(CRV, 0)
    elif (CRV == None) & (len(spectro_data) == 0):
        print('Please specify cluster RV.')
        return

    vlos_Tbl=[]
    if len(spectro_data) > 0:
        merged_spec = pd.merge(good_ruwe, spectro_data, on='source_id',suffixes=['','_y'])
        drop_y(merged_spec)
        rvmems = merged_spec[~np.isnan(merged_spec['selected_RV'])].reset_index(drop=True)

        if (CRV == None):
            crv = unumpy.uarray(np.mean(rvmems['selected_RV']), 0)

        bprp_rv = rvmems['bp_rp'].values
        sid_rv = rvmems['source_id'].values

        x_rv = np.cos(rvmems['dec']*np.pi/180) * np.sin((rvmems['ra'] - cra)*np.pi/180)
        y_rv = np.sin(rvmems['dec']*np.pi/180) * np.cos(cdec*np.pi/180) - np.cos(rvmems['dec']*np.pi/180) * np.sin(cdec*np.pi/180) * np.cos((rvmems['ra'] - cra)*np.pi/180)

        x_rv *= 206265/60
        y_rv *= 206265/60

        perppos_rv = 1e3/rvmems['parallax'] - cdist

        z_rv = perppos_rv

        vperp = (rvmems['selected_RV']-unumpy.nominal_values(crv))#*3.24078e-14*np.pi*1e7

        u_rv = unumpy.uarray(rvmems['selected_RV'],rvmems['selected_RV_error'])
        u_vperp = u_rv - crv

        pred_vperp = 1.3790e-3 * (x_rv*cpmra + y_rv*cpmdec) * (cdist/1e3)

        rr_rv = np.sqrt(x_rv**2+y_rv**2)
        tt_rv = np.arctan2(x_rv,y_rv)

        for i in range(len(tt_rv)):
            if tt_rv[i] < 0:
                tt_rv[i] += 2*np.pi

        e_vperp = unumpy.std_devs(u_vperp)

        vlos_Tbl = Table(data=[sid_rv,bprp_rv,x_rv, y_rv, rr_rv, tt_rv, vperp, e_vperp, pred_vperp],names=['source_id','bp_rp','x_rv','y_rv','r_rv','t_rv','vperp','e_vperp','pred_vperp'])

    ssmems = good_ruwe

    ra = ssmems['ra']
    dec = ssmems['dec']

    bprp = good_ruwe['bp_rp'].values

    sid = good_ruwe['source_id'].values


    rapos = (ssmems['ra']-cra)*60# arcmin *(np.pi/180)*(1e3/mems['parallax'])
    decpos = (ssmems['dec']-cdec)*60# arcmin *(np.pi/180)*(1e3/mems['parallax'])

    perppos = 1e3/ssmems['parallax'] - cdist

    z = perppos

    x = np.cos(ssmems['dec']*np.pi/180) * np.sin((ssmems['ra'] - cra)*np.pi/180)
    y = np.sin(ssmems['dec']*np.pi/180) * np.cos(cdec*np.pi/180) - np.cos(ssmems['dec']*np.pi/180) * np.sin(cdec*np.pi/180) * np.cos((ssmems['ra'] - cra)*np.pi/180)

    x *= 206265/60
    y *= 206265/60

    u_pmra = unumpy.uarray(ssmems['pmra'],ssmems['pmra_error'])
    u_pmdec = unumpy.uarray(ssmems['pmdec'],ssmems['pmdec_error'])

    mux = (ssmems['pmra']-cpmra) * np.cos((ssmems['ra'] - cra)*np.pi/180) - (ssmems['pmdec']-cpmdec) * np.sin(ssmems['dec']*np.pi/180) * np.sin((ssmems['ra']-cra)*np.pi/180)
    muy = (ssmems['pmra']-cpmra) * np.sin(cdec*np.pi/180) * np.sin((ssmems['ra']-cra)*np.pi/180) + (ssmems['pmdec']-cpmdec) * (np.cos(ssmems['dec']*np.pi/180)*np.cos(cdec*np.pi/180) + np.sin(ssmems['dec']*np.pi/180) * np.sin(cdec*np.pi/180) * np.cos((ssmems['ra']-cra)*np.pi/180))

    vx = 4.74*cdist/1e3 * mux
    vy = 4.74*cdist/1e3 * muy

    ###

    pred_mux = -6.1363e-5 * x * unumpy.nominal_values(crv) / (cdist/1e3) #mas/yr
    pred_muy = -6.1363e-5 * y * unumpy.nominal_values(crv) / (cdist/1e3)

    u_pred_mux = -6.1363e-5 * x * crv / (u_cdist/1e3) #mas/yr
    u_pred_muy = -6.1363e-5 * y * crv / (u_cdist/1e3)

    #pred_vperp = 1.3790e-3 * (x_rv*cpmra + y_rv*cpmdec) * (cdist/1e3)

    pred_vx = 4.74*cdist/1e3 * pred_mux
    pred_vy = 4.74*cdist/1e3 * pred_muy

    u_pred_vx = 4.74*u_cdist/1e3 * u_pred_mux
    u_pred_vy = 4.74*u_cdist/1e3 * u_pred_muy

    pmra_sub = u_pmra - u_cpmra
    pmdec_sub = u_pmdec - u_cpmdec

    u_mux = (pmra_sub) * np.cos((ssmems['ra'] - cra)*np.pi/180) - (pmdec_sub) * np.sin(ssmems['dec']*np.pi/180) * np.sin((ssmems['ra']-cra)*np.pi/180)
    u_muy = (pmra_sub) * np.sin(cdec*np.pi/180) * np.sin((ssmems['ra']-cra)*np.pi/180) + (pmdec_sub) * (np.cos(ssmems['dec']*np.pi/180)*np.cos(cdec*np.pi/180) + np.sin(ssmems['dec']*np.pi/180) * np.sin(cdec*np.pi/180) * np.cos((ssmems['ra']-cra)*np.pi/180))

    u_vx = 4.74*u_cdist/1e3 * u_mux
    u_vy = 4.74*u_cdist/1e3 * u_muy

    mux = unumpy.nominal_values(u_mux)
    muy = unumpy.nominal_values(u_muy)

    ###

    rr = np.sqrt(x**2+y**2)
    tt = np.arctan2(x,y)

    for i in range(len(tt)):
        if tt[i] < 0:
            tt[i] += 2*np.pi


    ###

    u_vr_mu = (x*u_mux + y*u_muy)/rr
    u_vr = (x*u_vx + y*u_vy)/rr

    u_vt_mu = (-y*u_mux + x*u_muy)/rr
    u_vt = (-y*u_vx + x*u_vy)/rr

    u_pred_mur = (x*u_pred_mux + y*u_pred_muy)/rr
    u_pred_vr = (x*u_pred_vx + y*u_pred_vy)/rr

    pred_mur = unumpy.nominal_values(u_pred_mur)

    u_pred_mut = (-y*u_pred_mux + x*u_pred_muy)/rr
    u_pred_vt = (-y*u_pred_vx + x*u_pred_vy)/rr

    pred_mut = unumpy.nominal_values(u_pred_mut)

    pred_rtTbl = Table(data=[x,y,rr,tt,pred_mur,pred_mut,pred_mux, pred_muy],names=['x','y','r','t','pred_mur','pred_mut','pred_mux','pred_muy'])
    pred_rtTbl.sort('r')

    rtTbl = Table(data=[sid,bprp,rr,tt,unumpy.nominal_values(u_vr_mu),unumpy.nominal_values(u_vt_mu),x,y,unumpy.nominal_values(u_mux), unumpy.nominal_values(u_muy)],names=['source_id','bp_rp','r','t','mur','mut','x','y','mux','muy'])
    rtTbl.sort('r')

    xyTbl = Table(data=[sid,bprp,x,y,unumpy.nominal_values(u_mux), unumpy.nominal_values(u_muy)], names=['source_id','bp_rp','x','y','mux','muy'])

    rtTbl_mu = Table(data=[sid,bprp,rr,tt,unumpy.nominal_values(u_vr_mu),unumpy.std_devs(u_vr_mu),unumpy.nominal_values(u_vt_mu),unumpy.std_devs(u_vt_mu),x,y,unumpy.nominal_values(u_mux), unumpy.nominal_values(u_muy)],names=['source_id','bp_rp','r','t','mur','e_mur','mut','e_mut','x','y','mux','muy'])
    rtTbl_mu.sort('r')

    ###

    return rtTbl_mu, pred_rtTbl, vlos_Tbl, xyTbl

def display_dynamical_times(rtTbl, pred_rtTbl, cdist, N_mem=None):
    if N_mem == None:
        n_mem = len(rtTbl)
    else:
        n_mem = N_mem
    avx = np.mean(np.abs(rtTbl['mux']-pred_rtTbl['pred_mux'])) / (np.pi*1e7) * 1e-3 / 206265 * cdist * 3.086e+13   #, np.mean(rtTbl['mux'])
    print('Avg. vx', avx)
    avg_v = np.sqrt(3)*avx
    print('Avg. v', avg_v)
    tcross = (np.max(rtTbl['r']) /60 * np.pi/180 * cdist * 3.086e+13) / avg_v / (np.pi*1e7) / 1e6
    print('tcross', tcross)
    trelax = tcross * n_mem / (6*np.log(n_mem/2))
    print('trelax', trelax)

    return tcross, trelax

def bin_kinematics_values(rtTbl, min_delta_r_arcmin=10, N_stars=75):
    rr = rtTbl['r']
    tt = rtTbl['t']
    mur = rtTbl['mur']
    mut = rtTbl['mut']

    edgs = []
    edgs_sim = []

    edgs += [np.min(rtTbl['r'])-0.01]


    n_stars = 0
    n_stars_bins = []
    n_stars_bins_sim = []

    dummy_r = []

    for i in range(len(rtTbl)-1):
        cur_r = rtTbl[i]['r']
        next_r = rtTbl[i+1]['r']
        dummy_r += [cur_r]
        n_stars += 1
        delta_r = cur_r - edgs[-1]
        if (delta_r) >= min_delta_r_arcmin and (n_stars >= N_stars):
            edgs += [(cur_r+next_r)/2]
            n_stars_bins += [n_stars]
            n_stars=0
    edgs += [rtTbl[-1]['r']+0.01]
    n_stars_bins += [n_stars]

    edgs = np.array(edgs)

    print(n_stars_bins,edgs)

    #return n_stars_bins, edgs

    medsr_mu,edges,number = binned_statistic(rr,mur,bins=edgs,statistic='mean')
    stdvsr_mu,edges,number = binned_statistic(rr,mur,bins=edgs,statistic='std')

    width = (edges[1] - edges[0])
    centers = edges[:-1]+np.diff(edges)/2 #edges[1:] - width/2

    medst_mu,edges2,number2 = binned_statistic(rr,mut,bins=edgs,statistic='mean')
    stdvst_mu,edges2,number2 = binned_statistic(rr,mut,bins=edgs,statistic='std')

    width2 = (edges2[1] - edges2[0])
    centers2 = edges[:-1]+np.diff(edges)/2#edges2[1:] - width2/2

    return width, centers, edges, medsr_mu, stdvsr_mu, medst_mu, stdvst_mu

def plot_kinematics_vectors(rtTbl_mu, pred_rtTbl, rdist=0, limit=150, cluster='Cluster'):

    r = rtTbl_mu[rtTbl_mu['r']>rdist]['r'].data
    x = rtTbl_mu[rtTbl_mu['r']>rdist]['x'].data
    y = rtTbl_mu[rtTbl_mu['r']>rdist]['y'].data
    mux = rtTbl_mu[rtTbl_mu['r']>rdist]['mux'].data - pred_rtTbl[pred_rtTbl['r'] > rdist]['pred_mux']
    muy = rtTbl_mu[rtTbl_mu['r']>rdist]['muy'].data - pred_rtTbl[pred_rtTbl['r'] > rdist]['pred_muy']
    mut = rtTbl_mu[rtTbl_mu['r']>rdist]['mut'].data - pred_rtTbl[pred_rtTbl['r'] > rdist]['pred_mut']
    mur = rtTbl_mu[rtTbl_mu['r']>rdist]['mur'].data - pred_rtTbl[pred_rtTbl['r'] > rdist]['pred_mur']

    f=plt.figure(figsize=(7,7))
    plt.quiver(x,y,mux,muy,np.sign(mut),angles='xy')

    #plt.quiver(x[5],y[5],mux[5],muy[5],np.sign(mut),angles='xy')
    #plt.colorbar()
    plt.xlim(limit,-1*limit)
    plt.ylim(-1*limit,limit)
    if rdist > 0:
        plt.title(cluster+', r > '+np.str(rdist) + ' arcmin')
    else:
        plt.title(cluster)
    plt.xlabel('East <---     x [arcmin]')
    plt.ylabel('y [arcmin]     North --->')

    #f.savefig('/Users/bhealy/Downloads/Praesepe_outer_proper_motions.pdf',bbox_inches='tight')

    return f

def mcmc_rt_bins(rr, mu_rt, e_mu_rt, edges, v_true, sig_true, crv, cdist):

    vrt_plus = mu_rt *(4.74*cdist/1e3)

    e_vrt = e_mu_rt *4.74*cdist/1e3
    #startvals = np.array([0,0,crv,1])
    startvals_r = np.array([v_true,sig_true])


    labels_rt = [r"$v_{0rt}$",r"$\sigma_{rt}$"]
    #labels_t = [r"$v_{0t}$",r"$\sigma_{t}$"]
    #labels = [r"$\theta_{c}$", "$v_{rot}$", "$v_{0}$","$\sigma_{LOS}$"]

    pos_r = startvals_r + 1e-4 * np.random.randn(100, 2)
    nwalkers, ndim = pos_r.shape

    bnz = len(edges) - 1

    medsr_mcmc = np.zeros(bnz)
    rerr_lo_mcmc = np.zeros(bnz)
    rerr_hi_mcmc = np.zeros(bnz)

    sigr_mcmc = np.zeros(bnz)
    srerr_lo_mcmc = np.zeros(bnz)
    srerr_hi_mcmc = np.zeros(bnz)

    for i in range(bnz):
        rindx = (rr > edges[i]) & (rr < edges[i+1])

        sampler_r = emcee.EnsembleSampler(nwalkers, ndim, log_probability_r, args=(vrt_plus[rindx],e_vrt[rindx]))
        sampler_r.run_mcmc(pos_r, 5000, progress=True);

        maxprob_indx = np.argmax(sampler_r.get_log_prob(discard=500, thin=15, flat=True))
        flat_samples_r = sampler_r.get_chain(discard=500, thin=15, flat=True)

       #medsr_mcmc[i] = flat_samples_r[maxprob_indx,0]
        medsr_mcmc[i] = np.percentile(flat_samples_r[:,0],50)

        rerr_lo_mcmc[i] = medsr_mcmc[i] - np.percentile(flat_samples_r[:,0], 16)
        rerr_hi_mcmc[i] = np.percentile(flat_samples_r[:,0], 84) - medsr_mcmc[i]


        sigr_mcmc[i] = flat_samples_r[maxprob_indx,1]
        srerr_lo_mcmc[i] = sigr_mcmc[i] - np.percentile(flat_samples_r[:,1], 16)
        srerr_hi_mcmc[i] = np.percentile(flat_samples_r[:,1], 84) - sigr_mcmc[i]

    sampler_r = emcee.EnsembleSampler(nwalkers, ndim, log_probability_r, args=(vrt_plus,e_vrt))
    sampler_r.run_mcmc(pos_r, 5000, progress=True);
    flat_samples_r = sampler_r.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(
        flat_samples_r, labels=labels_rt)#, truths=[posang_c_true, vrot_true, v0_true, siglos_true]
    #);
    plt.show()

    for i in range(2):
        mcmc = np.percentile(flat_samples_r[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels_rt[i])
        #print(txt)
        display(Math(txt))

    centers = edges[:-1]+np.diff(edges)/2
    plt.scatter(centers,medsr_mcmc)
    plt.errorbar(centers,medsr_mcmc,yerr=[rerr_lo_mcmc,rerr_hi_mcmc])
    plt.show()

    plt.scatter(centers,sigr_mcmc)
    plt.errorbar(centers,sigr_mcmc,yerr=[srerr_lo_mcmc,srerr_hi_mcmc])
    plt.show()

    return medsr_mcmc, rerr_lo_mcmc, rerr_hi_mcmc, sigr_mcmc, srerr_lo_mcmc, srerr_hi_mcmc, fig

def mcmc_vlos(vperp_plus, e_vperp, posangs, posang_c_true, vrot_true, v0_true, siglos_true):

    startvals = np.array([posang_c_true,vrot_true,v0_true,siglos_true])
    pos = startvals + 1e-4 * np.random.randn(100, 4)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(vperp_plus, e_vperp, posangs, posang_c_true))
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(vperp_plus, e_vperp, posangs))

    sampler.run_mcmc(pos, 5000, progress=True)

    #plt.rcParams['font.size']=17
    labels = [r"$\theta_{c}$", r"$v_{rot}$", r"$v_{0}$",r"$\sigma_{LOS}$"]

    #flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    logprob = sampler.get_log_prob(discard=500, thin=15, flat=True)

    fig = corner.corner(
        flat_samples, labels=labels,max_n_ticks=4)#, truths=[posang_c_true, vrot_true, v0_true, siglos_true]
    #);
    for i in range(4):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        #print(txt)
        display(Math(txt))

    mcmc_posang_c = np.percentile(flat_samples[:, 0], [16, 50, 84])
    q_posang_c = np.diff(mcmc_posang_c)
    posang_c = mcmc_posang_c[1]
    posang_c_lo_err = q_posang_c[0]
    posang_c_hi_err = q_posang_c[1]

    mcmc_vrot = np.percentile(flat_samples[:, 1], [16, 50, 84])
    q_vrot = np.diff(mcmc_vrot)
    vrot = mcmc_vrot[1]
    vrot_lo_err = q_vrot[0]
    vrot_hi_err = q_vrot[1]

    mcmc_v0 = np.percentile(flat_samples[:, 2], [16, 50, 84])
    q_v0 = np.diff(mcmc_v0)
    v0 = mcmc_v0[1]
    v0_lo_err = q_v0[0]
    v0_hi_err = q_v0[1]

    mcmc_siglos = np.percentile(flat_samples[:, 3], [16, 50, 84])
    q_siglos = np.diff(mcmc_siglos)
    siglos = mcmc_siglos[1]
    siglos_lo_err = q_siglos[0]
    siglos_hi_err = q_siglos[1]

    return posang_c, posang_c_lo_err, posang_c_hi_err, vrot, vrot_lo_err, vrot_hi_err, v0, v0_lo_err, v0_hi_err, siglos, siglos_lo_err, siglos_hi_err, fig

def store_mcmc_values(cluster, pred_rtTbl, centers,medsr_mcmc,sigr_mcmc,rerr_lo_mcmc,rerr_hi_mcmc,srerr_lo_mcmc,srerr_hi_mcmc,medst_mcmc,sigt_mcmc,terr_lo_mcmc,
terr_hi_mcmc,sterr_lo_mcmc,sterr_hi_mcmc, posang_c, posang_c_lo_err, posang_c_hi_err, vrot, vrot_lo_err, vrot_hi_err, v0, v0_lo_err, v0_hi_err, siglos, siglos_lo_err, siglos_hi_err):

    pred_rtTbl.to_pandas().set_index('r').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_pred_rtTbl.csv')

    Table(data=[centers,medsr_mcmc,sigr_mcmc,rerr_lo_mcmc,rerr_hi_mcmc,srerr_lo_mcmc,srerr_hi_mcmc,medst_mcmc,sigt_mcmc,terr_lo_mcmc,
    terr_hi_mcmc,sterr_lo_mcmc,sterr_hi_mcmc],names=['centers','medsr_mcmc','sigr_mcmc','rerr_lo_mcmc','rerr_hi_mcmc','srerr_lo_mcmc',
    'srerr_hi_mcmc','medst_mcmc','sigt_mcmc','terr_lo_mcmc','terr_hi_mcmc','sterr_lo_mcmc','sterr_hi_mcmc']).to_pandas().set_index('centers').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_kinematics_values.csv')

    pd.DataFrame({'posang_c':[posang_c],'posang_c_lo_err':[posang_c_lo_err],'posang_c_hi_err':[posang_c_hi_err],'vrot':[vrot],'vrot_lo_err':[vrot_lo_err],
    'vrot_hi_err':[vrot_hi_err],'v0':[v0],'v0_lo_err':[v0_lo_err],'v0_hi_err':[v0_hi_err],'siglos':[siglos],'siglos_lo_err':[siglos_lo_err],
    'siglos_hi_err':[siglos_hi_err]}).set_index('posang_c').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_LOS_kinematics.csv')

    return

def store_mcmc_values_rt(cluster, pred_rtTbl, centers,medsr_mcmc,sigr_mcmc,rerr_lo_mcmc,rerr_hi_mcmc,srerr_lo_mcmc,srerr_hi_mcmc,medst_mcmc,sigt_mcmc,terr_lo_mcmc,
terr_hi_mcmc,sterr_lo_mcmc,sterr_hi_mcmc):

    pred_rtTbl.to_pandas().set_index('r').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_pred_rtTbl.csv')

    Table(data=[centers,medsr_mcmc,sigr_mcmc,rerr_lo_mcmc,rerr_hi_mcmc,srerr_lo_mcmc,srerr_hi_mcmc,medst_mcmc,sigt_mcmc,terr_lo_mcmc,
    terr_hi_mcmc,sterr_lo_mcmc,sterr_hi_mcmc],names=['centers','medsr_mcmc','sigr_mcmc','rerr_lo_mcmc','rerr_hi_mcmc','srerr_lo_mcmc',
    'srerr_hi_mcmc','medst_mcmc','sigt_mcmc','terr_lo_mcmc','terr_hi_mcmc','sterr_lo_mcmc','sterr_hi_mcmc']).to_pandas().set_index('centers').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_kinematics_values.csv')

    return

def store_mcmc_values_los(cluster, posang_c, posang_c_lo_err, posang_c_hi_err, vrot, vrot_lo_err, vrot_hi_err, v0, v0_lo_err, v0_hi_err, siglos, siglos_lo_err, siglos_hi_err):
    pd.DataFrame({'posang_c':[posang_c],'posang_c_lo_err':[posang_c_lo_err],'posang_c_hi_err':[posang_c_hi_err],'vrot':[vrot],'vrot_lo_err':[vrot_lo_err],
    'vrot_hi_err':[vrot_hi_err],'v0':[v0],'v0_lo_err':[v0_lo_err],'v0_hi_err':[v0_hi_err],'siglos':[siglos],'siglos_lo_err':[siglos_lo_err],
    'siglos_hi_err':[siglos_hi_err]}).set_index('posang_c').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_LOS_kinematics.csv')

    return

def read_mcmc_values(cluster):

    kinematics = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_kinematics_values.csv')
    pred_rtTbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_pred_rtTbl.csv')
    los_kinematics = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_LOS_kinematics.csv')

    centers=kinematics['centers']
    medsr_mcmc=kinematics['medsr_mcmc']
    sigr_mcmc=kinematics['sigr_mcmc']
    rerr_lo_mcmc=kinematics['rerr_lo_mcmc']
    rerr_hi_mcmc=kinematics['rerr_hi_mcmc']
    srerr_lo_mcmc=kinematics['srerr_lo_mcmc']
    srerr_hi_mcmc=kinematics['srerr_hi_mcmc']
    medst_mcmc=kinematics['medst_mcmc']
    sigt_mcmc=kinematics['sigt_mcmc']
    terr_lo_mcmc=kinematics['terr_lo_mcmc']
    terr_hi_mcmc=kinematics['terr_hi_mcmc']
    sterr_lo_mcmc=kinematics['sterr_lo_mcmc']
    sterr_hi_mcmc=kinematics['sterr_hi_mcmc']

    posang_c = los_kinematics['posang_c'].values[0]
    posang_c_lo_err = los_kinematics['posang_c_lo_err'].values[0]
    posang_c_hi_err = los_kinematics['posang_c_hi_err'].values[0]

    vrot = los_kinematics['vrot'].values[0]
    vrot_lo_err = los_kinematics['vrot_lo_err'].values[0]
    vrot_hi_err = los_kinematics['vrot_hi_err'].values[0]

    v0 = los_kinematics['v0'].values[0]
    v0_lo_err = los_kinematics['v0_lo_err'].values[0]
    v0_hi_err = los_kinematics['v0_hi_err'].values[0]

    siglos = los_kinematics['siglos'].values[0]
    siglos_lo_err = los_kinematics['siglos_lo_err'].values[0]
    siglos_hi_err = los_kinematics['siglos_hi_err'].values[0]



    return kinematics, pred_rtTbl, los_kinematics, centers,medsr_mcmc,sigr_mcmc,rerr_lo_mcmc,rerr_hi_mcmc,srerr_lo_mcmc,srerr_hi_mcmc,medst_mcmc,sigt_mcmc,terr_lo_mcmc,terr_hi_mcmc,sterr_lo_mcmc,sterr_hi_mcmc, posang_c, posang_c_lo_err, posang_c_hi_err, vrot, vrot_lo_err, vrot_hi_err, v0, v0_lo_err, v0_hi_err, siglos, siglos_lo_err, siglos_hi_err
