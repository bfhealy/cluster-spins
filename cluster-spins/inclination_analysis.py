import numpy as np
from scipy.stats import norm,uniform,t
import pandas as pd
from zero_point import zpt
from scipy.stats import gaussian_kde
import glob
from uncertainties import unumpy
from astropy.table import Table
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astroquery.exceptions import TableParseError
import sys
sys.path.append('/Users/bhealy/Documents/PhD_Thesis/cluster-spins/cluster-spins/')
import jackson_cone_model
import emcee
import corner
from IPython.display import display, Math

def bv_to_bprp(bv):
    #return 0.0187 + 1.6814*bv - 0.3357*bv**2 +0.1117*bv**3
    return 0.0545 + 1.4939*bv - 0.6274*bv**2 +0.2957*bv**3

def bv_to_grp(bv):
    return 0.0348 + 0.9429*bv -0.1979*bv**2 + 0.0181*bv**3

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

def weighted_average(distribution, weights):
    return round(sum([distribution[i]*weights[i] for i in range(len(distribution))])/sum(weights),2)

def merge_gaia_edr3_data(mems,cluster,phase=3):
    #gaia_qry = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_EDR3-result.csv')
    gaia_qry = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_'+np.str(phase)+'/'+cluster+'/'+cluster+'_EDR3-result.csv')

    gaia_qry.rename({'dr2_source_id':'source_id'},axis=1,inplace=True)
    gaia_qry.rename({'bp_rp':'edr3_bp_rp'},axis=1,inplace=True)
    gaia_qry.rename({'phot_g_mean_mag':'edr3_phot_g_mean_mag'},axis=1,inplace=True)

    targets = pd.merge(mems,gaia_qry,on='source_id',suffixes=['_dr2',''])
    drop(targets,'_dr2')
    #print(len(targets))

    targets = targets.sort_values('angular_distance').drop_duplicates('source_id',keep='first').reset_index(drop=True).drop_duplicates('dr3_source_id',keep='first').reset_index(drop=True).sort_values('bp_rp').reset_index(drop=True)

    zpt.load_tables()
    zeropoints = targets.apply(zpt.zpt_wrapper, axis=1)
    targets['parallax_corrected'] = targets['parallax'] - zeropoints

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

def calc_posterior(meanv,sigv,meanu,sigu):

    varr = np.linspace(-20000,20000,400000)
    cosiarr = np.linspace(0,1.000,1000)
    #cosiarr = np.linspace(0,1.999,2000)


    lv = norm.pdf(varr,meanv,sigv)
    pv = uniform.pdf(varr,np.min(varr),2*np.max(varr))
    pcosi = uniform.pdf(cosiarr,0.000,1.000)
    post = []
    for i in range(len(cosiarr)):
        lu = norm.pdf(varr*np.sqrt(1-cosiarr[i]**2),meanu,sigu)

        #area = np.trapz(lu,varr)
        #lu = lu/area
        post += [pcosi[i]*np.trapz(lv*lu*pv,varr)]
    post = np.array(post)

    return cosiarr,post

def calc_posterior_limit(meanv,sigv,ulimit):

    varr = np.linspace(-20000,20000,400000)
    cosiarr = np.linspace(0,1.000,1000)
    #cosiarr = np.linspace(0,1.999,2000)


    lv = norm.pdf(varr,meanv,sigv)
    pv = uniform.pdf(varr,np.min(varr),2*np.max(varr))
    pcosi = uniform.pdf(cosiarr,0.000,1.000)
    post = []
    for i in range(len(cosiarr)):
        lu = uniform.pdf(varr*np.sqrt(1-cosiarr[i]**2),0,ulimit)
        #area = np.trapz(lu,varr)
        #lu = lu/area
        post += [pcosi[i]*np.trapz(lv*lu*pv,varr)]
    post = np.array(post)

    return cosiarr,post

def calc_posterior_sini(meanv,sigv,meanu,sigu):

    #varr = np.linspace(0,300,30000)
    #varr = np.linspace(0,10000,1e5)
    #varr = np.linspace(0,20000,200000)

    #varr = np.linspace(0,50000,500000)
    varr = np.linspace(-20000,20000,400000)





    #cosiarr = np.linspace(0,0.999,1000)

    siniarr = np.linspace(0.000,1.000,1000)
    #siniarr = np.linspace(0.000,3.000,3000)
    #siniarr = np.linspace(0.000,4.000,4000)


    lv = norm.pdf(varr,meanv,sigv)
    pv = uniform.pdf(varr,np.min(varr),2*np.max(varr))
    #pcosi = uniform.pdf(cosiarr,0,1)

    psini = uniform.pdf(siniarr,0.000,1.000)
    #psini = uniform.pdf(siniarr,0.000,3.000)
    #psini = uniform.pdf(siniarr,0.000,4.000)


    post = []
    for i in range(len(siniarr)):
        lu = norm.pdf(varr*siniarr[i],meanu,sigu)
        #lu = t.pdf(varr*siniarr[i],loc=meanu,scale=sigu,df=2)

        area = np.trapz(lu,varr)
        lu = lu/area

        post += [psini[i]*np.trapz(lv*lu*pv,varr)]
    post = np.array(post)
    #post = post / np.trapz(post,siniarr)

    return siniarr,post

def calc_posterior_sini_limit(meanv,sigv,ulimit,normalize=False):

    #varr = np.linspace(0,300,30000)
    #varr = np.linspace(0,10000,1e5)
    #varr = np.linspace(0,20000,200000)

    #varr = np.linspace(0,50000,500000)
    varr = np.linspace(-20000,20000,400000)

    #cosiarr = np.linspace(0,0.999,1000)

    siniarr = np.linspace(0.000,1.000,1000)
    #siniarr = np.linspace(0.000,3.000,3000)
    #siniarr = np.linspace(0.000,4.000,4000)


    lv = norm.pdf(varr,meanv,sigv)
    pv = uniform.pdf(varr,0,np.max(varr))
    #pcosi = uniform.pdf(cosiarr,0,1)

    psini = uniform.pdf(siniarr,0.000,1.000)
    #psini = uniform.pdf(siniarr,0.000,3.000)
    #psini = uniform.pdf(siniarr,0.000,4.000)

    post = []
    for i in range(len(siniarr)):
        #lu = norm.pdf(varr*siniarr[i],meanu,sigu)
        lu = uniform.pdf(varr*siniarr[i],0,ulimit)
        #lu = t.pdf(varr*siniarr[i],loc=meanu,scale=sigu,df=2)
        if normalize:
            area = np.trapz(lu,varr)
            lu = lu/area

        post += [psini[i]*np.trapz(lv*lu*pv,varr)]
    post = np.array(post)
    #post = post / np.trapz(post,siniarr)

    return siniarr,post

def calc_posterior_sini_new(meanv,sigv,meanu,sigu,distrib='norm',df=None):

    #varr = np.linspace(0,300,30000)
    #varr = np.linspace(0,10000,1e5)
    varr = np.linspace(-20000,20000,400000)

    #varr = np.linspace(0,50000,500000)

    #cosiarr = np.linspace(0,0.999,1000)

    #siniarr = np.linspace(0.000,1.000,1000)
    siniarr = np.linspace(0.000,3.000,3000)
    #siniarr = np.linspace(0.000,4.000,4000)


    lv = norm.pdf(varr,meanv,sigv)
    pv = uniform.pdf(varr,np.min(varr),2*np.max(varr))
    #pcosi = uniform.pdf(cosiarr,0,1)

    #psini = uniform.pdf(siniarr,0.000,1.000)
    psini = uniform.pdf(siniarr,0.000,3.000)
    #psini = uniform.pdf(siniarr,0.000,4.000)


    post = []
    for i in range(len(siniarr)):
        if distrib == 'norm':
            lu = norm.pdf(varr*siniarr[i],meanu,sigu)
        elif (distrib == 't') & (df != None):
            lu = t.pdf(varr*siniarr[i],loc=meanu,scale=sigu,df=df)

        #area = np.trapz(lu,varr)
        #lu = lu/area

        post += [psini[i]*np.trapz(lv*lu*pv,varr)]
    post = np.array(post)
    post = post / np.trapz(post,siniarr)

    return siniarr,post

def calc_posterior_sini_eq11(meanv,sigv,meanu,sigu):

    #varr = np.linspace(0,300,30000)
    #varr = np.linspace(0,10000,1e5)
    vsiniarr = np.linspace(-20000,20000,400000)

    #varr = np.linspace(0,50000,500000)

    #cosiarr = np.linspace(0,0.999,1000)

    siniarr = np.linspace(0.000,3.000,3000)
    #siniarr = np.linspace(0.000,4.000,4000)


    #lv = norm.pdf(varr,meanv,sigv)
    lu = norm.pdf(vsiniarr,meanu,sigu)

    #pv = uniform.pdf(varr,0,np.max(varr))

    #pcosi = uniform.pdf(cosiarr,0,1)

    psini = uniform.pdf(siniarr,0.000,3.000)
    #psini = uniform.pdf(siniarr,0.000,4.000)


    post = []
    for i in range(len(siniarr)):
        pv = uniform.pdf(vsiniarr/siniarr[i],np.min(vsiniarr/siniarr[i]),2*np.max(vsiniarr/siniarr[i]))
        area_pv = np.trapz(pv,vsiniarr)
        pv = pv/area_pv
        #lu = norm.pdf(varr*siniarr[i],meanu,sigu)
        lv = norm.pdf(vsiniarr/siniarr[i],meanv,sigv)
        #lu = t.pdf(varr*siniarr[i],loc=meanu,scale=sigu,df=2)

        area = np.trapz(lv,vsiniarr)
        lv = lv/area

        post += [(psini[i] / siniarr[i])*np.trapz(lv*lu*pv,vsiniarr)]
    post = np.array(post)
    post = post / np.trapz(post,siniarr)

    return siniarr,post
def determine_eprop_sini(targets,cluster):

    vsini = targets['vsini']
    vsini_err = targets['e_vsini'] #* np.sqrt(targets['NW'])

    #targets.drop('e_vsini',axis=1,inplace=True)
    #targets['e_vsini'] = vsini_err

    radius = targets['radius_ng'] * 7e5
    #old_radius_err = np.array([np.max((finalresults['r_hi_err'][i],finalresults['r_lo_err'][i])) for i in range(len(radius))])
    radius_err = targets['e_radius_ng'] * 7e5
    #radius_err *= 7e5

    period = targets['period'] * 24 * 3600
    period_err = targets['period_unc'] * 24 * 3600
    #period_err = finalresults['period'].data * 24 * 3600 * 0.05

    V = 2*np.pi*radius/period
    V_err = V * np.sqrt((period_err/period)**2 + (radius_err/radius)**2)

    targets['V'] = V
    targets['V_err'] = V_err

    Per = unumpy.uarray(period,period_err)
    Rad = unumpy.uarray(radius,radius_err) #* 1.07
    Vsini = unumpy.uarray(vsini, vsini_err)

    incnames = []
    # Calculate sini and its uncertainty
    #Sini = (Per*Vsini)/(2*np.pi*Rad)
    Sini = (Per*Vsini)/(2*np.pi*Rad)
    sini = np.zeros(len(Sini))
    sini_err = np.copy(sini)
    for k in range(len(Sini)):
        incnames += [targets['source_id'][k]]
        Sinik = Sini[k]
        nomval = Sinik.nominal_value
        print(nomval)
        std_dev = Sinik.std_dev
        print(std_dev)
        sini[k] = nomval
        sini_err[k] = std_dev/nomval

    # Calculate inclination from all sini values <= 1
    Inclination = unumpy.arcsin(Sini[Sini<=1])

    # Create separate arrays of inclination and uncertainty
    err = np.zeros(len(Inclination))
    inc = np.copy(sini_err)
    for j in range(len(Inclination)):
        Ij = Inclination[j]
        nomval = Ij.nominal_value
        std_dev = Ij.std_dev

        inc[j] = nomval
        err[j] = std_dev#/nomval
    incnames = np.array(incnames)

    siniTbl = Table(data=[incnames,sini,sini_err],names=['source_id','eprop_sini','eprop_sini_err']).to_pandas()
    merge_eprop_sini = pd.merge(siniTbl,targets,on='source_id')

    #merge_eprop_sini.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/paperdata/'+cluster+'/'+cluster+'_merge_eprop_sini.csv')

    return merge_eprop_sini

def determine_bayes_sini(merge_eprop_sini,cluster,plot_posteriors=True,distrib='norm',df=None):

    sini_all = []
    posteriors_all = []
    best_sini = []
    for i in range(len(merge_eprop_sini)):
        print(i)
        #sini_sing, posterior_sing = calc_posterior_sini(V[i],V_err[i],vsini[i],vsini_err[i])
        sini_sing, posterior_sing = calc_posterior_sini_new(merge_eprop_sini.loc[i]['V'],merge_eprop_sini.loc[i]['V_err'],merge_eprop_sini.loc[i]['vsini'],merge_eprop_sini.loc[i]['e_vsini'],distrib=distrib,df=df)

        sini_all += [sini_sing]
        posteriors_all += [posterior_sing]
        best_sini += [sini_sing[np.nanargmax(posterior_sing)]]

    best_sini = np.array(best_sini)
    posteriors_all = np.array(posteriors_all)
    sini_all = np.array(sini_all)

    if plot_posteriors:
        for i in range(len(merge_eprop_sini)):
            fig = plt.figure(figsize=(10,10))
            plt.plot(sini_all[i],posteriors_all[i])
            fig.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/posteriors_sini/'+np.str(merge_eprop_sini['source_id'][i])+'_'+np.str(i)+'.pdf',overwrite=True)
            plt.close()

    bayes_sini=[]

    bayes_sini_lo_err=best_sini  #np.zeros(len(merge_eprop_sini))
    bayes_sini_hi_err=best_sini #np.zeros(len(merge_eprop_sini))

    for i in range(len(merge_eprop_sini)):

        posterior_new = posteriors_all[i][~np.isnan(posteriors_all[i])]
        sini_new = sini_all[i][~np.isnan(posteriors_all[i])]

        bayes_index = np.argmax(posterior_new)

        bayes_sini += [sini_new[bayes_index]]

        for j in range(0,bayes_index):

            idx = bayes_index - j
            integral_ratio = np.trapz(posterior_new[idx:bayes_index],sini_new[idx:bayes_index]) / np.trapz(posterior_new,sini_new)
            if integral_ratio < 0.34:
                continue

            else:
                bayes_sini_lo_err[i] = bayes_sini[i] - sini_new[idx]

                break
        for j in range(0,len(sini_new-bayes_index)):
            idx = bayes_index + j
            integral_ratio = np.trapz(posterior_new[bayes_index:idx],sini_new[bayes_index:idx]) / np.trapz(posterior_new,sini_new)
            if integral_ratio < 0.34:
                continue
            else:
                bayes_sini_hi_err[i] = sini_new[idx] - bayes_sini[i]
                break

    bayes_sini = np.array(bayes_sini)
    bayes_sini_lo_err = np.array(bayes_sini_lo_err)
    bayes_sini_hi_err = np.array(bayes_sini_hi_err)

    bayes_finalresults = Table(data=[merge_eprop_sini['source_id'],bayes_sini,bayes_sini_lo_err,bayes_sini_hi_err],names=['source_id','bayes_sini','bayes_sini_lo_err','bayes_sini_hi_err']).to_pandas()

    joined_finalresultstbl = pd.merge(merge_eprop_sini,bayes_finalresults,on='source_id')
    joined_finalresultstbl.sort_values('bayes_sini',inplace=True)

    return joined_finalresultstbl

def check_sed_mags(joined_finalresultstbl,cluster,suffix='nogaia',n_mags_min=3,n_optical_mags_min=1):
    mags=[]
    n_mags = np.zeros(len(joined_finalresultstbl))
    SDSS_u_vals = np.zeros(len(joined_finalresultstbl))
    B_vals = np.zeros(len(joined_finalresultstbl))
    V_vals = np.zeros(len(joined_finalresultstbl))
    Tycho_B_vals = np.zeros(len(joined_finalresultstbl))
    Tycho_V_vals = np.zeros(len(joined_finalresultstbl))
    J_vals = np.zeros(len(joined_finalresultstbl))
    H_vals = np.zeros(len(joined_finalresultstbl))
    K_vals = np.zeros(len(joined_finalresultstbl))
    PS_g_vals = np.zeros(len(joined_finalresultstbl))
    GALEX_NUV_vals = np.zeros(len(joined_finalresultstbl))
    GALEX_FUV_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_u_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_v_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_g_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_r_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_i_vals = np.zeros(len(joined_finalresultstbl))
    SkyMapper_z_vals = np.zeros(len(joined_finalresultstbl))
    W1_vals = np.zeros(len(joined_finalresultstbl))
    W2_vals = np.zeros(len(joined_finalresultstbl))
    W3_vals = np.zeros(len(joined_finalresultstbl))
    W4_vals = np.zeros(len(joined_finalresultstbl))
    atleastone_optical_mag = np.zeros(len(joined_finalresultstbl))
    atleast_all_mags = np.zeros(len(joined_finalresultstbl))
    atleast_index = np.zeros(len(joined_finalresultstbl))

    for i in range(len(joined_finalresultstbl)):
        SED = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/SEDfiles_'+cluster+'_'+suffix+'/'+np.str(joined_finalresultstbl.loc[i,'source_id'])+'.csv',index_col=0)
        keys_SED = [x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')]
        #print(keys_SED)
        mags+=[keys_SED]
        n_mags[i] = len(keys_SED)

        SDSS_u_vals[i] = 'SDSS_u' in keys_SED
        B_vals[i] = 'B' in keys_SED
        V_vals[i] = 'V' in keys_SED
        Tycho_B_vals[i] = 'Tycho_B' in keys_SED
        Tycho_V_vals[i] = 'Tycho_V' in keys_SED
        J_vals[i] = 'J' in keys_SED
        H_vals[i] = 'H' in keys_SED
        K_vals[i] = 'K' in keys_SED
        PS_g_vals[i] = 'PS_g' in keys_SED
        GALEX_NUV_vals[i] = 'GALEX_NUV' in keys_SED
        GALEX_FUV_vals[i] = 'GALEX_FUV' in keys_SED
        SkyMapper_u_vals[i] = 'SkyMapper_u' in keys_SED
        SkyMapper_v_vals[i] = 'SkyMapper_v' in keys_SED
        SkyMapper_g_vals[i] = 'SkyMapper_g' in keys_SED
        SkyMapper_r_vals[i] = 'SkyMapper_r' in keys_SED
        SkyMapper_i_vals[i] = 'SkyMapper_i' in keys_SED
        SkyMapper_z_vals[i] = 'SkyMapper_z' in keys_SED
        W1_vals[i] = 'W1' in keys_SED
        W2_vals[i] = 'W2' in keys_SED
        W3_vals[i] = 'W3' in keys_SED
        W4_vals[i] = 'W4' in keys_SED

        atleastone_optical_mag[i] = np.sum([B_vals[i],V_vals[i],Tycho_B_vals[i],Tycho_V_vals[i],PS_g_vals[i],SkyMapper_g_vals[i],SkyMapper_v_vals[i]]) >= n_optical_mags_min
        atleast_all_mags[i] = n_mags[i] >= n_mags_min

        atleast_index[i] = atleastone_optical_mag[i] and atleast_all_mags[i]

    return atleast_index.astype(bool)

def log_likelihood(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    Alpha, Lamda = theta

    redpoints = jackson_cone_model.mc_sini_distrib_new(sim_vrot,Alpha,Lamda,deltav_ln_frac,sigv_ln_frac,deltaP,deltaR,vsini_threshold=vsini_threshold,interp=True,xdatapoints=incempx1,ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    A = np.log(np.sqrt(2*np.pi*err**2))

    B = (incempx1 - model_incempx1)**2 / (2*(len(err)-2)*(err**2))
    return -1 * np.sum(A + B)

def log_likelihood_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    Alpha, Lamda = theta

    redpoints = jackson_cone_model.mc_sini_distrib_new(sim_vrot,Alpha,Lamda,deltav_ln_frac,sigv_ln_frac,deltaP,deltaR,vsini_threshold=vsini_threshold,interp=True,xdatapoints=incempx1,ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    lo = incempx1 > model_incempx1
    hi = ~lo

    #A_lo = np.log(np.sqrt(2*np.pi*err_lo**2))
    #A_hi = np.log(np.sqrt(2*np.pi*err_hi**2))
    #A = np.concatenate([A_lo[lo], A_hi[hi]])

    err_avg = (err_lo + err_hi)/2
    A = np.log(np.sqrt(2*np.pi*err_avg**2))

    B_lo = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_lo**2))
    B_hi = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_hi**2))

    B = np.concatenate([B_lo[lo], B_hi[hi]])

    return -1 * np.sum(A + B)

def reduced_chi2(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0,tdistrib=False):
    Alpha, Lamda = theta
    redpoints = jackson_cone_model.mc_sini_distrib_new(sim_vrot,Alpha,Lamda,deltav_ln_frac,sigv_ln_frac,deltaP,deltaR,vsini_threshold=vsini_threshold,interp=True,xdatapoints=incempx1,ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    A = np.log(np.sqrt(2*np.pi*err**2))

    B = (incempx1 - model_incempx1)**2 / (2*(len(err)-2)*(err**2))
    return B

def reduced_chi2_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0,tdistrib=False):
    Alpha, Lamda = theta
    redpoints = jackson_cone_model.mc_sini_distrib_new(sim_vrot,Alpha,Lamda,deltav_ln_frac,sigv_ln_frac,deltaP,deltaR,vsini_threshold=vsini_threshold,interp=True,xdatapoints=incempx1,ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    lo = incempx1 > model_incempx1
    hi = ~lo
    #A = np.log(np.sqrt(2*np.pi*err**2))

    #B = (incempx1 - model_incempx1)**2 / (2*(len(err)-2)*(err**2))

    #A_lo = np.log(np.sqrt(2*np.pi*err_lo**2))
    #A_hi = np.log(np.sqrt(2*np.pi*err_hi**2))

    #A = np.concatenate([A_lo[lo], A_hi[hi]])

    err_avg = (err_lo + err_hi)/2
    A = np.log(np.sqrt(2*np.pi*err_avg**2))

    B_lo = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_lo**2))
    B_hi = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_hi**2))

    B = np.concatenate([B_lo[lo], B_hi[hi]])

    return B

def log_likelihood_3param(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    Alpha, Lamda, Frac = theta
    redpoints = jackson_cone_model.mc_sini_distrib_uncerts_fraction(sim_vrot,Alpha,Lamda,Frac,deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=vsini_threshold, interp=True, xdatapoints=incempx1, ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    A = np.log(np.sqrt(2*np.pi*err**2))

    B = (incempx1 - model_incempx1)**2 / (2*(len(err)-2)*(err**2))
    return -1 * np.sum(A + B)

def log_likelihood_3param_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    Alpha, Lamda, Frac = theta
    redpoints = jackson_cone_model.mc_sini_distrib_uncerts_fraction(sim_vrot,Alpha,Lamda,Frac,deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=vsini_threshold, interp=True, xdatapoints=incempx1, ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    lo = incempx1 > model_incempx1
    hi = ~lo

    #A_lo = np.log(np.sqrt(2*np.pi*err_lo**2))
    #A_hi = np.log(np.sqrt(2*np.pi*err_hi**2))

    #A = np.concatenate([A_lo[lo], A_hi[hi]])

    err_avg = (err_lo + err_hi)/2
    A = np.log(np.sqrt(2*np.pi*err_avg**2))

    B_lo = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_lo**2))
    B_hi = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_hi**2))

    B = np.concatenate([B_lo[lo], B_hi[hi]])

    return -1 * np.sum(A + B)

def reduced_chi2_3param(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0,tdistrib=False):
    Alpha, Lamda, Frac = theta
    redpoints = jackson_cone_model.mc_sini_distrib_uncerts_fraction(sim_vrot,Alpha,Lamda,Frac,deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=vsini_threshold, interp=True, xdatapoints=incempx1, ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    A = np.log(np.sqrt(2*np.pi*err**2))

    B = (incempx1 - model_incempx1)**2 / (2*(len(err)-2)*(err**2))
    return B

def reduced_chi2_3param_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0,tdistrib=False):
    Alpha, Lamda, Frac = theta
    redpoints = jackson_cone_model.mc_sini_distrib_uncerts_fraction(sim_vrot,Alpha,Lamda,Frac,deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=vsini_threshold, interp=True, xdatapoints=incempx1, ydatapoints=incempy1,tdistrib=tdistrib)

    model_incempx1 = redpoints

    lo = incempx1 > model_incempx1
    hi = ~lo

    #A_lo = np.log(np.sqrt(2*np.pi*err_lo**2))
    #A_hi = np.log(np.sqrt(2*np.pi*err_hi**2))

    #A = np.concatenate([A_lo[lo], A_hi[hi]])

    err_avg = (err_lo + err_hi)/2
    A = np.log(np.sqrt(2*np.pi*err_avg**2))

    B_lo = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_lo**2))
    B_hi = (incempx1 - model_incempx1)**2 / (2*(len(incempx1)-2)*(err_hi**2))

    B = np.concatenate([B_lo[lo], B_hi[hi]])

    return B

def log_prior(theta):
    Alpha, Lamda = theta
    if 0 <= Alpha <= 90 and 0 <= Lamda <= 90:
        lp = 0
    else:
        lp = -np.inf
    return lp

def log_prior_3param(theta):
    Alpha, Lamda, Frac = theta

    if 1 <= Alpha <= 90 and 1 <= Lamda <= 90 and 0 <= Frac <= 1:
        lp = 0
    else:
        lp = -np.inf
    return lp

def log_probability(theta,incempx1,incempy1,err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib)

def log_probability_asym(theta,incempx1,incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib)

def log_probability_3param(theta,incempx1,incempy1,err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    lp = log_prior_3param(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_3param(theta, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib)

def log_probability_3param_asym(theta,incempx1,incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, tdistrib=False):
    lp = log_prior_3param(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_3param_asym(theta, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib)

def fit_inclinations_mcmc(log_probability, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, startvals=[45,90], nsteps=10000, nwalkers=50,tdistrib=False):
    startvals = np.array(startvals)
    pos = startvals + 1e-4 * np.random.randn(nwalkers, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib))

    sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler

def fit_inclinations_mcmc_asym(log_probability, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, startvals=[45,90], nsteps=10000, nwalkers=50,tdistrib=False):
    startvals = np.array(startvals)
    pos = startvals + 1e-4 * np.random.randn(nwalkers, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_asym, args=(incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib))

    sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler

def fit_inclinations_mcmc_3param(log_probability_3param, incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, startvals=[45,90,0], nsteps=10000, nwalkers=50, tdistrib=False):
    startvals=np.array(startvals)
    pos = startvals + 1e-4 * np.random.randn(nwalkers, 3)
    nwalkers, ndim = pos.shape

    sampler_3param = emcee.EnsembleSampler(nwalkers, ndim, log_probability_3param, args=(incempx1, incempy1, err, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib))

    sampler_3param.run_mcmc(pos, nsteps, progress=True)

    return sampler_3param

def fit_inclinations_mcmc_3param_asym(log_probability_3param, incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold=0, startvals=[45,90,0], nsteps=10000, nwalkers=50, tdistrib=False):
    startvals=np.array(startvals)
    pos = startvals + 1e-4 * np.random.randn(nwalkers, 3)
    nwalkers, ndim = pos.shape

    sampler_3param = emcee.EnsembleSampler(nwalkers, ndim, log_probability_3param_asym, args=(incempx1, incempy1, err_lo, err_hi, sim_vrot, deltav_ln_frac, sigv_ln_frac, deltaP, deltaR, vsini_threshold, tdistrib))

    sampler_3param.run_mcmc(pos, nsteps, progress=True)

    return sampler_3param

def get_flat_samples(sampler,discard=1000):
    labels = ["Alpha","Lambda"]
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    flat_lnprob = sampler.get_log_prob(discard=discard,flat=True)

    fig = corner.corner(
        flat_samples, labels=labels,quantiles=[.5,.5])
    for i in range(2):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
    plt.show()

    h_alf=plt.hist(flat_samples[:,0],bins=90)
    f_val = np.mean(h_alf[1][np.argmax(h_alf[0]):np.argmax(h_alf[0])+2])
    print(f_val)
    plt.axvline(f_val,color='red')
    plt.show()

    h_lam=plt.hist(flat_samples[:,1],bins=90)
    f_val = np.mean(h_lam[1][np.argmax(h_lam[0]):np.argmax(h_lam[0])+2])
    print(f_val)
    plt.axvline(f_val,color='red')
    plt.show()

    print('Alpha', np.percentile(flat_samples[:,0], [5,16,50,84,95]))
    print('Lambda', np.percentile(flat_samples[:,1], [5,16,32,50,84,95]))

    return flat_samples, flat_lnprob, h_alf, h_lam

def get_flat_samples_3param(sampler_3param,discard=1000):
    labels = ["Alpha","Lambda","Fraction"]
    flat_samples_3param = sampler_3param.get_chain(discard=discard, flat=True)
    flat_lnprob_3param = sampler_3param.get_log_prob(discard=discard,flat=True)

    fig = corner.corner(
        flat_samples_3param, labels=labels,quantiles=[.5,.5,.5])

    for i in range(3):
        mcmc = np.percentile(flat_samples_3param[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])

        display(Math(txt))
    plt.show()

    h_alf=plt.hist(flat_samples_3param[:,0],bins=90)
    f_val = np.mean(h_alf[1][np.argmax(h_alf[0]):np.argmax(h_alf[0])+2])
    print(f_val)
    plt.axvline(f_val,color='red')
    plt.show()

    h_lam=plt.hist(flat_samples_3param[:,1],bins=90)
    f_val = np.mean(h_lam[1][np.argmax(h_lam[0]):np.argmax(h_lam[0])+2])
    print(f_val)
    plt.axvline(f_val,color='red')
    plt.show()

    h_f=plt.hist(flat_samples_3param[:,2],bins=90)
    f_val = np.mean(h_f[1][np.argmax(h_f[0]):np.argmax(h_f[0])+2])
    print(f_val)
    plt.axvline(f_val,color='red')
    plt.show()

    print('Alpha', np.percentile(flat_samples_3param[:,0], [5,16,50,84,95]))
    print('Lambda', np.percentile(flat_samples_3param[:,1], [5,16,32,50,84,95]))
    print('f', np.percentile(flat_samples_3param[:,2], [5,16,50,84,95]))
    print()
    print(sampler_3param.flatchain[np.argmax(sampler_3param.flatlnprobability,axis=0)])

    return flat_samples_3param, flat_lnprob_3param, h_alf, h_lam, h_f
