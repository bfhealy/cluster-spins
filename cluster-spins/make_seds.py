#from isochrones import get_ichrone, SingleStarModel
#from isochrones.priors import AgePrior, AVPrior, DistancePrior, EEP_prior, FehPrior, FlatPrior, GaussianPrior, ChabrierPrior

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18

from astroquery.mast import Catalogs
from astroquery.irsa import Irsa
import astropy.units as u
from astroquery.vizier import Vizier, VizierClass
from astroquery.gaia import Gaia

#Gaia.login(user='bhealy',password='exAtm17$')

from astroquery.esasky import ESASky
from astropy.coordinates import SkyCoord
import pandas as pd
from astropy.io import fits,ascii
from astropy.table import Table

def plot_isochrone_fit(SED,mod1,fgsz=(8,8)):
    plt.rcParams['font.size']=20
    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-1])
    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-2])
    mist = get_ichrone('mist', bands=[x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')])



    #fig = plt.figure(figsize=fgsz)
    fig = mist.isochrone(mod1.derived_samples.describe()['age'][1], mod1.derived_samples.describe()['feh'][1],AV=mod1.derived_samples.describe()['AV'][1]).plot('logTeff','logL',zorder=0,color='black',figsize=fgsz,legend=None)
    plt.scatter(mod1.derived_samples.describe()['logTeff'][1], mod1.derived_samples.describe()['logL'][1],color='red',zorder=1,s=120,edgecolor='black')
    plt.ylabel('Log L')
    plt.xlim(5.65,3.35)
    fig.set_xlabel(r'Log T$_{\rm eff}$')

    plt.rcParams['font.size']=12
    return fig

def plot_obs_SED(SED,fgsz=(8,8)):

    d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference_updated_angstrom.csv')
    zpts = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_flux_zeropoints_updated.csv')

    mag_mapper = {'BP':'BP_mag', 'RP':'RP_mag', 'J':'J_mag', 'H':'H_mag', 'K':'K_mag', 'W3':'W3_mag',
                'W4':'W4_mag', 'W1':'W1_mag', 'W2':'W2_mag', 'Tycho_B':'Tycho_B_mag', 'Tycho_V':'Tycho_V_mag',
                'B':'B_mag', 'V':'V_mag', 'R':'R_mag', 'I':'I_mag', 'GALEX_FUV':'GALEX_FUV_mag', 'GALEX_NUV':'GALEX_NUV_mag', 'PS_y':'PS_y_mag',
                'PS_g':'PS_g_mag', 'PS_r':'PS_r_mag', 'PS_i':'PS_i_mag', 'PS_z':'PS_z_mag',
                'SDSS_g':'SDSS_g_mag', 'SDSS_r':'SDSS_r_mag', 'SDSS_i':'SDSS_i_mag', 'SDSS_z':'SDSS_z_mag','SDSS_u':'SDSS_u_mag',
                'SkyMapper_u':'SkyMapper_u_mag', 'SkyMapper_v':'SkyMapper_v_mag', 'SkyMapper_g':'SkyMapper_g_mag',
                'SkyMapper_r':'SkyMapper_r_mag', 'SkyMapper_i':'SkyMapper_i_mag', 'SkyMapper_z':'SkyMapper_z_mag'
                }


    #keys_SED = [x for x in SED.keys()][:-1]
    keys_SED = [x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')]

    #values_SED = [x[0] for x in SED[keys_SED].values()]
    values_SED = [x[0] for x in [SED[x] for x in keys_SED]]
    #values_SED = [x[0] for x in SED.values()]

    errors_SED = [x[1] for x in [SED[x] for x in keys_SED]]
    #errors_SED = [x[1] for x in SED[keys_SED].values()]
    #errors_SED = [x[1] for x in SED.values()]

    #obs_mags = np.array(values_SED[:-1])
    #obs_errs = np.array(errors_SED[:-1])
    obs_mags = np.array(values_SED)
    obs_errs = np.array(errors_SED)


    weff_SED = np.array([d[x][0] for x in keys_SED])
    width_SED = np.array([d[x][1] for x in keys_SED])
    zeropoints = np.array([zpts[x] for x in keys_SED]).transpose()[0]

    #derived_mags = mod1.derived_samples.describe()[['BP_mag', 'RP_mag','J_mag', 'H_mag', 'K_mag', 'W3_mag', 'W1_mag', 'W2_mag',
    #       'Tycho_B_mag','Tycho_V_mag']].loc['mean'].values

    #print(weff_SED,derived_mags,zeropoints)

    obs_fluxes = zeropoints * 10**(-1*obs_mags/2.512) * weff_SED
    obs_flux_errs = obs_fluxes - zeropoints * 10**(-1*(obs_mags + obs_errs)/2.512) * weff_SED

    #print(df_derived_mags)

    fig=plt.figure(figsize=(8,8))
    #plt.gca().invert_yaxis()

    #plt.scatter(weff_SED,obs_mags,color='black',marker='D',s=50)
    #####plt.scatter(weff_SED * 1e-4,obs_fluxes,marker='s',s=110,facecolors='none',edgecolors='gray')
    ##plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='slategrey',capsize=7,linestyle='none',zorder=2,linewidth=2)
    plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='black',capsize=7,linestyle='none',zorder=0,linewidth=2)
    return fig

def plot_SED_fit(SED,mod1,fgsz=(8,8)):
    plt.rcParams['font.size']=20

    #d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference.csv')
    d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference_updated_angstrom.csv')
    zpts = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_flux_zeropoints_updated.csv')

    mag_mapper = {'BP':'BP_mag', 'RP':'RP_mag', 'J':'J_mag', 'H':'H_mag', 'K':'K_mag', 'W3':'W3_mag',
                'W4':'W4_mag', 'W1':'W1_mag', 'W2':'W2_mag', 'Tycho_B':'Tycho_B_mag', 'Tycho_V':'Tycho_V_mag',
                'B':'B_mag', 'V':'V_mag', 'R':'R_mag', 'I':'I_mag', 'GALEX_FUV':'GALEX_FUV_mag', 'GALEX_NUV':'GALEX_NUV_mag', 'PS_y':'PS_y_mag',
                'PS_g':'PS_g_mag', 'PS_r':'PS_r_mag', 'PS_i':'PS_i_mag', 'PS_z':'PS_z_mag',
                'SDSS_g':'SDSS_g_mag', 'SDSS_r':'SDSS_r_mag', 'SDSS_i':'SDSS_i_mag', 'SDSS_z':'SDSS_z_mag','SDSS_u':'SDSS_u_mag',
                'SkyMapper_u':'SkyMapper_u_mag', 'SkyMapper_v':'SkyMapper_v_mag', 'SkyMapper_g':'SkyMapper_g_mag',
                'SkyMapper_r':'SkyMapper_r_mag', 'SkyMapper_i':'SkyMapper_i_mag', 'SkyMapper_z':'SkyMapper_z_mag'
                }


    #keys_SED = [x for x in SED.keys()][:-1]
    keys_SED = [x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')]

    #values_SED = [x[0] for x in SED[keys_SED].values()]
    values_SED = [x[0] for x in [SED[x] for x in keys_SED]]
    #values_SED = [x[0] for x in SED.values()]

    errors_SED = [x[1] for x in [SED[x] for x in keys_SED]]
    #errors_SED = [x[1] for x in SED[keys_SED].values()]
    #errors_SED = [x[1] for x in SED.values()]

    #obs_mags = np.array(values_SED[:-1])
    #obs_errs = np.array(errors_SED[:-1])
    obs_mags = np.array(values_SED)
    obs_errs = np.array(errors_SED)

    derived_mags = mod1.derived_samples.describe()[[mag_mapper[keys_SED[x]] for x in range(len(keys_SED))]].loc['mean'].values

    weff_SED = np.array([d[x][0] for x in keys_SED])
    width_SED = np.array([d[x][1] for x in keys_SED])
    zeropoints = np.array([zpts[x] for x in keys_SED]).transpose()[0]

    #derived_mags = mod1.derived_samples.describe()[['BP_mag', 'RP_mag','J_mag', 'H_mag', 'K_mag', 'W3_mag', 'W1_mag', 'W2_mag',
    #       'Tycho_B_mag','Tycho_V_mag']].loc['mean'].values

    #print(weff_SED,derived_mags,zeropoints)
    df_derived_mags = pd.DataFrame(data={'wvl':weff_SED, 'mag':derived_mags, 'zeropoint':zeropoints})

    obs_fluxes = zeropoints * 10**(-1*obs_mags/2.512) * weff_SED
    obs_flux_errs = obs_fluxes - zeropoints * 10**(-1*(obs_mags + obs_errs)/2.512) * weff_SED

    df_derived_mags['flux'] = df_derived_mags['zeropoint'] * 10**(-1*df_derived_mags['mag']/2.512) * df_derived_mags['wvl']


    df_derived_mags.set_index('wvl',inplace=True)
    df_derived_mags.sort_index(inplace=True)
    #print(df_derived_mags)

    fig=plt.figure(figsize=fgsz)
    #plt.gca().invert_yaxis()

    #plt.scatter(weff_SED,obs_mags,color='black',marker='D',s=50)
    #####plt.scatter(weff_SED * 1e-4,obs_fluxes,marker='s',s=110,facecolors='none',edgecolors='gray')
    ##plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='slategrey',capsize=7,linestyle='none',zorder=2,linewidth=2)
    plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='black',capsize=7,linestyle='none',zorder=0,linewidth=2)


    #plt.plot(df_derived_mags['mag'].index,df_derived_mags['mag'].values,'ro-')
    #plt.plot(df_derived_mags.index * 1e-4,df_derived_mags['flux'],'ro-')

    ###plt.scatter(df_derived_mags.index * 1e-4,df_derived_mags['flux'],color='red',s=120,edgecolor='black')
    plt.scatter(df_derived_mags.index * 1e-4,df_derived_mags['flux'],s=120,edgecolor='red',facecolor='none',linewidths=2,zorder=1)


    plt.xlabel(r'Wavelength [$\mu$m]')
    #plt.ylabel('Magnitude')
    plt.ylabel(r'$\lambda F_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$]')
    plt.yscale('log')
    plt.xscale('log')
    #plt.ylim(ymax,ymin)
    plt.rcParams['font.size']=12
    return fig

def plot_SED_fit_mags(SED,mod1,fgsz=(8,8)):
    plt.rcParams['font.size']=18

    #d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference.csv')
    d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference_updated_angstrom.csv')

    mag_mapper = {'BP':'BP_mag', 'RP':'RP_mag', 'J':'J_mag', 'H':'H_mag', 'K':'K_mag', 'W3':'W3_mag',
                'W4':'W4_mag', 'W1':'W1_mag', 'W2':'W2_mag', 'Tycho_B':'Tycho_B_mag', 'Tycho_V':'Tycho_V_mag',
                'B':'B_mag', 'V':'V_mag', 'R':'R_mag', 'I':'I_mag', 'GALEX_FUV':'GALEX_FUV_mag', 'GALEX_NUV':'GALEX_NUV_mag', 'PS_y':'PS_y_mag',
                'PS_g':'PS_g_mag', 'PS_r':'PS_r_mag', 'PS_i':'PS_i_mag', 'PS_z':'PS_z_mag',
                'SDSS_g':'SDSS_g_mag', 'SDSS_r':'SDSS_r_mag', 'SDSS_i':'SDSS_i_mag', 'SDSS_z':'SDSS_z_mag',
                'SkyMapper_u':'SkyMapper_u_mag', 'SkyMapper_v':'SkyMapper_v_mag', 'SkyMapper_g':'SkyMapper_g_mag',
                'SkyMapper_r':'SkyMapper_r_mag', 'SkyMapper_i':'SkyMapper_i_mag', 'SkyMapper_z':'SkyMapper_z_mag'
                }


    keys_SED = [x for x in SED.keys()][:-1]
    values_SED = [x[0] for x in SED.values()]
    errors_SED = [x[1] for x in SED.values()]

    obs_mags = values_SED[:-1]
    obs_errs = errors_SED[:-1]

    derived_mags = mod1.derived_samples.describe()[[mag_mapper[keys_SED[x]] for x in range(len(keys_SED))]].loc['mean'].values

    weff_SED = np.array([d[x][0] for x in keys_SED])
    width_SED = np.array([d[x][1] for x in keys_SED])

    #derived_mags = mod1.derived_samples.describe()[['BP_mag', 'RP_mag','J_mag', 'H_mag', 'K_mag', 'W3_mag', 'W1_mag', 'W2_mag',
    #       'Tycho_B_mag','Tycho_V_mag']].loc['mean'].values

    df_derived_mags = pd.DataFrame(data={'wvl':weff_SED, 'mag':derived_mags})
    df_derived_mags.set_index('wvl',inplace=True)
    df_derived_mags.sort_index(inplace=True)

    print(df_derived_mags)

    fig=plt.figure(figsize=fgsz)
    plt.gca().invert_yaxis()

    plt.scatter(weff_SED*1e-4,obs_mags,color='black',marker='D',s=50)
    plt.errorbar(weff_SED*1e-4,obs_mags,xerr=width_SED*1e-4,yerr=obs_errs,color='black',capsize=5,linestyle='none')
    plt.plot(df_derived_mags['mag'].index * 1e-4,df_derived_mags['mag'].values,'ro-')

    plt.xlabel('Wavelength [microns]')
    plt.ylabel('Magnitude')

    #plt.ylim(ymax,ymin)
    plt.rcParams['font.size']=12
    return fig

def query_catalogs(targets,apassdata,cluster,skymapperdata=[],sdssdata=[],panstarrsdata=[],phase='Phase_3',suffix='nogaia',apass_coords=[],young_cluster=False):
    if (len(apass_coords) == 0) & (len(apassdata) > 0):
        apass_coords = SkyCoord(apassdata['radeg'],apassdata['decdeg'],unit=(u.deg,u.deg))

    status = np.zeros(len(targets))
    for i in range(len(targets)):
        print(np.str(i+1) + '/' + np.str(len(targets)))
        target_0 = targets[i:i+1]
        sid = target_0['source_id'].values[0]
        t0_coords = SkyCoord(target_0['ra'].values[0],target_0['dec'].values[0],unit=(u.deg,u.deg))

        if (len(apassdata) > 0):
            min_sep_indx = np.argmin(apass_coords.separation(t0_coords))
            min_sep = apass_coords[min_sep_indx].separation(t0_coords)
            #print(min_sep)
            apass_qry = apassdata.iloc[min_sep_indx]
        else:
            apass_qry = []
            min_sep = 99999*u.arcsec

        if len(skymapperdata) != 0:
            try:
                skymapper_qry = skymapperdata.set_index('source_id').loc[sid]
            except KeyError:
                skymapper_qry=[]
        else:
            skymapper_qry=[]

        if len(sdssdata) != 0:
            try:
                sdss_qry = sdssdata.set_index('source_id').loc[sid]
            except KeyError:
                sdss_qry=[]
        else:
            sdss_qry=[]

        if len(panstarrsdata) != 0:
            try:
                panstarrs_qry = panstarrsdata.set_index('source_id').loc[sid]
            except KeyError:
                panstarrs_qry=[]
        else:
            panstarrs_qry=[]

        for attempt in range(10):
            try:
                #tic_qry = Catalogs.query_object('Gaia DR2 '+np.str(target_0['source_id'].values[0]),catalog='TIC',radius=1*u.arcsec)
                tic_qry = Catalogs.query_region(t0_coords,catalog='TIC',radius=1*u.arcsec)
                print('TIC')
                #irsa_qry = Irsa.query_region('Gaia DR2 '+np.str(target_0['source_id'].values[0]),catalog='catwise_2020',radius=1*u.arcsec)
                irsa_qry = Irsa.query_region(t0_coords,catalog='catwise_2020',radius=1*u.arcsec)
                print('IRSA')
                irsa_qry_2 = Irsa.query_region(t0_coords,catalog='allwise_p3as_psd',radius=1*u.arcsec)
                print('IRSA2')
                #galex_qry = Catalogs.query_object('Gaia DR2 '+np.str(target_0['source_id'].values[0]),catalog='Galex',radius=1*u.arcsec)
                galex_qry = Catalogs.query_region(t0_coords,catalog='Galex',radius=1*u.arcsec)

                print('GALEX')

        #####tycho_qry = Vizier.query_region(t0_coords,catalog='I/259',radius=1*u.arcsec)

                VC = VizierClass(columns=['all'])
                tycho_qry = VC.query_region(t0_coords,catalog='I/259/tyc2',radius=1*u.arcsec)
                print('Tycho')
                print('Finished.')
                #status[i] = 1
                break
            except:
                print('Continuing.')
                continue

        SED = dict()

        if len(tic_qry) != 0:
            tic_qry = tic_qry[0]
            twom_flgs = tic_qry['TWOMflag']

            #print(tic_qry['Bmag'], tic_qry['Vmag'])
            if ~np.isnan(tic_qry['Bmag']) & ~np.isnan(tic_qry['e_Bmag']):
                if (tic_qry['e_Bmag'] > 0):
                    Bmag, e_Bmag = tic_qry['Bmag','e_Bmag']

                    #SED['B'] = (Bmag,e_Bmag)

            if ~np.isnan(tic_qry['Vmag']) & ~np.isnan(tic_qry['e_Vmag']):
                if (tic_qry['e_Vmag'] > 0):
                    Vmag, e_Vmag = tic_qry['Vmag','e_Vmag']

                    #SED['V'] = (Vmag,e_Vmag)

            if ~np.isnan(tic_qry['gaiabp']) & ~np.isnan(tic_qry['e_gaiabp']):
                if (tic_qry['e_gaiabp'] > 0):
                    bpmag, e_bpmag = tic_qry['gaiabp','e_gaiabp']

                    #SED['BP'] = (bpmag,e_bpmag)
                #SED['e_BP'] = e_bpmag

            if ~np.isnan(tic_qry['gaiarp']) & ~np.isnan(tic_qry['e_gaiarp']):
                if (tic_qry['e_gaiarp'] > 0):
                    rpmag, e_rpmag = tic_qry['gaiarp','e_gaiarp']

                    #SED['RP'] = (rpmag,e_rpmag)
                #SED['e_RP'] = e_rpmag

            if not np.ma.is_masked(twom_flgs):
                ph_qual, rd_flag, bl_flag, cc_flag, gal_contam, mp_flg  = twom_flgs.split('-')
                if ~np.isnan(tic_qry['Jmag']) & ~np.isnan(tic_qry['e_Jmag']):
                    if ((ph_qual[0]=='A') | ((rd_flag[0] == '1') | (rd_flag[0] == '3'))) & (bl_flag[0] == '1') & (cc_flag[0] == '0') & (tic_qry['e_Jmag'] > 0):
                        Jmag, e_Jmag = tic_qry['Jmag','e_Jmag']

                        SED['J'] = (Jmag,e_Jmag)
                    #SED['e_J'] = e_Jmag

                if ~np.isnan(tic_qry['Hmag']) & ~np.isnan(tic_qry['e_Hmag']):
                    if ((ph_qual[1]=='A') | ((rd_flag[1] == '1') | (rd_flag[1] == '3'))) & (bl_flag[1] == '1') & (cc_flag[1] == '0') & (tic_qry['e_Hmag'] > 0):
                        Hmag, e_Hmag = tic_qry['Hmag','e_Hmag']

                        SED['H'] = (Hmag,e_Hmag)
                    #SED['e_H'] = e_Hmag

                if ~np.isnan(tic_qry['Kmag']) & ~np.isnan(tic_qry['e_Kmag']):
                    if ((ph_qual[2]=='A') | ((rd_flag[2] == '1') | (rd_flag[2] == '3'))) & (bl_flag[2] == '1') & (cc_flag[2] == '0') & (tic_qry['e_Kmag'] > 0):
                        Kmag, e_Kmag = tic_qry['Kmag','e_Kmag']

                        SED['K'] = (Kmag,e_Kmag)

            #if ~np.isnan(tic_qry['w3mag']) & ~np.isnan(tic_qry['e_w3mag']) & ~(young_cluster):
        #        if (tic_qry['e_w3mag'] > 0):
        #            w3mag, e_w3mag = tic_qry['w3mag','e_w3mag']

        #            SED['W3'] = (w3mag,e_w3mag)


        #    if ~np.isnan(tic_qry['w4mag']) & ~np.isnan(tic_qry['e_w4mag']) & ~(young_cluster):
        #        if (tic_qry['e_w3mag'] > 0):
        #            w4mag, e_w4mag = tic_qry['w4mag','e_w4mag']
#
#                    SED['W4'] = (w4mag,e_w4mag)
        if len(irsa_qry_2) != 0:
            irsa_qry_2 = irsa_qry_2[0]
            w3mag,e_w3mag = irsa_qry_2['w3mpro','w3sigmpro']
            w4mag,e_w4mag = irsa_qry_2['w4mpro','w4sigmpro']

            #print(w3mag,e_w3mag)
            #print(w4mag,e_w4mag)

            allwise_cc_flags, allwise_moon_levs, allwise_phot_qual, allwise_ext_flag = irsa_qry_2['cc_flags','moon_lev','ph_qual','ext_flg']

            #if ~np.isnan(w3mag) & ~np.isnan(e_w3mag) & (not young_cluster):
            if (np.ma.is_masked(w3mag)) & (np.ma.is_masked(e_w3mag)) & (not young_cluster):
                if (e_w3mag > 0) & ((allwise_phot_qual[2]=='A') | (allwise_phot_qual[2]=='B')) & (allwise_ext_flag == '0') & (allwise_cc_flags[2]=='0') & (allwise_moon_levs[2]=='0'):
                    #w3mag, e_w3mag = tic_qry['w3mag','e_w3mag']

                    SED['W3'] = (w3mag,e_w3mag)
                #SED['e_W3'] = e_w3mag

            #print(w4mag != '--')
            #print(e_w4mag != '--')
            #print(not young_cluster)
            #if ~np.isnan(w3mag) & ~np.isnan(e_w3mag) & (not young_cluster):
            if (np.ma.is_masked(w4mag)) & (np.ma.is_masked(e_w4mag)) & (not young_cluster):
                if (e_w4mag > 0) & ((allwise_phot_qual[3]=='A') | (allwise_phot_qual[3]=='B')) & (allwise_ext_flag == '0') & (allwise_cc_flags[3]=='0') & (allwise_moon_levs[3]=='0'):
                    #w4mag, e_w4mag = tic_qry['w4mag','e_w4mag']

                    SED['W4'] = (w4mag,e_w4mag)

        #print(irsa_qry.keys())
        if len(irsa_qry) != 0:
            irsa_qry = irsa_qry[0]

            #w1mag,e_w1mag = irsa_qry['w1mag','w1sigm']
            #w2mag,e_w2mag = irsa_qry['w2mag','w2sigm']

            w1mag,e_w1mag = irsa_qry['w1mpro','w1sigmpro']
            w2mag,e_w2mag = irsa_qry['w2mpro','w2sigmpro']

            catwise_cc_flags, catwise_ab_flags = irsa_qry['cc_flags','ab_flags']

            #if ~np.isnan(w1mag) & ~np.isnan(e_w1mag) & (not young_cluster) & (catwise_cc_flags != ''):
            if ~np.ma.is_masked(w1mag) & ~np.ma.is_masked(e_w1mag) & (not young_cluster) & (catwise_cc_flags != ''):
                if (e_w1mag > 0) & (w1mag >= 8) & (catwise_cc_flags[0] == '0') & (catwise_ab_flags[0] == '0'):
                    SED['W1'] = (w1mag,e_w1mag)
            #SED['e_W1'] = e_w1mag

            #print(w2mag)
            #print(e_w2mag)
            #print(catwise_cc_flagss)

            #if ~np.isnan(w2mag) & ~np.isnan(e_w2mag) & (not young_cluster)& (catwise_cc_flags != ''):
            if ~np.ma.is_masked(w2mag) & ~np.ma.is_masked(e_w2mag) & (not young_cluster)& (catwise_cc_flags != ''):
                if (e_w2mag > 0) & (w2mag >= 7) & (catwise_cc_flags[1] == '0') & (catwise_ab_flags[1] == '0'):
                    SED['W2'] = (w2mag,e_w2mag)
            #SED['e_W2'] = e_w2mag

        if len(galex_qry) != 0:
            galex_qry = galex_qry[0]
            nuv_mag = galex_qry['nuv_mag']
            nuv_magerr = galex_qry['nuv_magerr']

            bad_nuv_flag = 0
            if np.ma.is_masked(galex_qry['nuv_artifact']):
                bad_nuv_flag = 1
            else:
                for i in np.array([1,2,3,5,7,8,9])-1:
                    if galex_qry['nuv_artifact'] & 2**(i) != 0:
                        bad_nuv_flag = 1

            fuv_mag = galex_qry['fuv_mag']
            fuv_magerr = galex_qry['fuv_magerr']
            #print(galex_qry['fuv_artifact'])

            bad_fuv_flag = 0
            if np.ma.is_masked(galex_qry['fuv_artifact']):
                bad_fuv_flag = 1
            else:
                for i in np.array([1,2,3,5,7,8,9])-1:
                    if galex_qry['fuv_artifact'] & 2**(i) != 0:
                        bad_fuv_flag = 1

            if (np.ma.is_masked(nuv_mag) == False) & (np.ma.is_masked(nuv_magerr) == False):
                if (nuv_magerr > 0) & (~bad_nuv_flag):
                    SED['GALEX_NUV'] = (nuv_mag,nuv_magerr)
            if (np.ma.is_masked(fuv_mag) == False) & (np.ma.is_masked(fuv_magerr) == False):
                if (fuv_magerr > 0) & (~bad_fuv_flag):
                    SED['GALEX_FUV'] = (fuv_mag,fuv_magerr)

        if min_sep < 1*u.arcsec:
            Bmag = apass_qry['Johnson_B (B)']
            e_Bmag = apass_qry['Berr']
            Bmagflag = 'APASS_DR10'

            if (~np.isnan(Bmag)) & (~np.isnan(e_Bmag)):
                if (Bmag >= 7) & (Bmag <= 17) & (e_Bmag > 0):
                    #print()
                    SED['B'] = (Bmag,e_Bmag)

            #SED['e_B'] = e_Bmag
            #SED['Bmagflag'] = Bmagflag

            Vmag = apass_qry['Johnson_V (V)']
            e_Vmag = apass_qry['Verr']
            Vmagflag = 'APASS_DR10'

            if (~np.isnan(Vmag)) & (~np.isnan(e_Vmag)):
                if (Vmag >= 7) & (Vmag <= 17) & (e_Vmag > 0):
                    #print()
                    SED['V'] = (Vmag,e_Vmag)

            gmag = apass_qry['Sloan_g (SG)']
            e_gmag = apass_qry['SGerr']
            rmag = apass_qry['Sloan_r (SR)']
            e_rmag = apass_qry['SRerr']
            imag = apass_qry['Sloan_i (SI)']
            e_imag = apass_qry['SIerr']
            zmag = apass_qry['Sloan_z (SZ)']
            e_zmag = apass_qry['SZerr']

        if len(skymapper_qry) > 0:
            skm_mags = ['u','v','g','r','i','z']
            for m in skm_mags:
                sm_key = m
                #print(sm_key)
                #print(skymapper_qry[sm_key+'_flags'])
                #print(skymapper_qry[sm_key+'_nimaflags'])
                #print(skymapper_qry[sm_key+'_ngood'])
                #if (~np.isnan(skymapper_qry[sm_key+'_flags'])) & (~np.isnan(skymapper_qry[sm_key+'_nimaflags'])) & (~np.isnan(skymapper_qry[sm_key+'_ngood'])):
                if (skymapper_qry[sm_key+'_flags'] == 0) & (skymapper_qry[sm_key+'_nimaflags'] == 0) & (skymapper_qry[sm_key+'_ngood'] > 1):
                    print('SkyMapper')
                    SED['SkyMapper_'+sm_key] = (skymapper_qry[sm_key+'_psf'], skymapper_qry['e_'+sm_key+'_psf'])

        if len(sdss_qry) > 0:
            print('SDSS')
            if (~np.isnan(sdss_qry['psfMag_u'])) & (~np.isnan(sdss_qry['psfMagErr_u'])):
                SED['SDSS_u'] = (sdss_qry['psfMag_u'], sdss_qry['psfMagErr_u'])
            if (~np.isnan(sdss_qry['psfMag_g'])) & (~np.isnan(sdss_qry['psfMagErr_g'])):
                SED['SDSS_g'] = (sdss_qry['psfMag_g'], sdss_qry['psfMagErr_g'])
            if (~np.isnan(sdss_qry['psfMag_r'])) & (~np.isnan(sdss_qry['psfMagErr_r'])):
                SED['SDSS_r'] = (sdss_qry['psfMag_r'], sdss_qry['psfMagErr_r'])
            if (~np.isnan(sdss_qry['psfMag_i'])) & (~np.isnan(sdss_qry['psfMagErr_i'])):
                SED['SDSS_i'] = (sdss_qry['psfMag_i'], sdss_qry['psfMagErr_i'])
            if (~np.isnan(sdss_qry['psfMag_z'])) & (~np.isnan(sdss_qry['psfMagErr_z'])):
                SED['SDSS_z'] = (sdss_qry['psfMag_z'], sdss_qry['psfMagErr_z'])

        if len(panstarrs_qry) > 0:
            print('Pan-STARRS')
            SED['PS_g'] = (panstarrs_qry['gMeanPSFMag'], panstarrs_qry['gMeanPSFMagErr'])

            #if (~np.isnan(gmag)) & (~np.isnan(e_gmag)):
            #    if (gmag >= 7) & (gmag <= 17) & (e_gmag != 0):
            #        SED['SDSS_g'] = (gmag,e_gmag)
            #if (~np.isnan(rmag)) & (~np.isnan(e_rmag)):
            #    if (rmag >= 7) & (rmag <= 17) & (e_rmag != 0):
            #        SED['SDSS_r'] = (rmag,e_rmag)
            #if (~np.isnan(imag)) & (~np.isnan(e_imag)):
            #    if (imag >= 7) & (imag <= 17) & (e_imag != 0):
            #        SED['SDSS_i'] = (imag,e_imag)
            #if (~np.isnan(Bmag)) & (~np.isnan(e_Bmag)) & (Bmag >= 7) & (Bmag == 17):
            #    SED['SDSS_z'] = (Bmag,e_Bmag)

            #SED['e_V'] = e_Vmag
            #SED['Vmagflag']=Vmagflag

            #if ~np.isnan(apass_qry['PanSTARRS_Y (Y)']):
            #    Ymag = apass_qry['PanSTARRS_Y (Y)']
            #    e_Ymag = apass_qry['Yerr']

            #    SED['PS_y'] = (Ymag,e_Ymag)
                #SED['e_PS_y']=e_Ymag
        #
        if len(tycho_qry) != 0:
            #tycho_qry = tycho_qry['I/259/tyc2'][0]
            tycho_qry = tycho_qry[tycho_qry.keys()[0]][0]

            BTmag = tycho_qry['BTmag']
            VTmag = tycho_qry['VTmag']

            e_BTmag = tycho_qry['e_BTmag']
            e_VTmag = tycho_qry['e_VTmag']

            if ~np.isnan(BTmag) & ~np.isnan(e_BTmag):
                if (e_BTmag > 0):
                    SED['Tycho_B'] = (BTmag.astype(np.float64),e_BTmag.astype(np.float64))
            #SED['e_Tycho_B'] = e_BTmag
            if ~np.isnan(VTmag) & ~np.isnan(e_VTmag):
                if (e_VTmag > 0):
                    SED['Tycho_V'] = (VTmag.astype(np.float64),e_VTmag.astype(np.float64))
            #SED['e_Tycho_V'] = e_VTmag

#        elif len(tic_qry) != 0:
#            Bmag = tic_qry['Bmag']
#            e_Bmag = tic_qry['e_Bmag']
#            Bmagflag = tic_qry['BmagFlag']

            ###SED['B'] = (Bmag,e_Vmag)

            #SED['e_B'] = e_Bmag
            #SED['Bmagflag'] = Bmagflag

#            Vmag = tic_qry['Vmag']
#            e_Vmag = tic_qry['e_Vmag']
#            Vmagflag = tic_qry['VmagFlag']

            ###SED['V'] = (Vmag,e_Vmag)

            #SED['e_V'] = e_Vmag
            #SED['Vmagflag']=Vmagflag

        if cluster == 'Praesepe':
            ps_mags = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/Praesepe_panstarrs_phot.csv',index_col='source_id')
            #tgt_ps = ps_mags.loc[i]
            tgt_ps = ps_mags.loc[sid]

            ps_gmag = tgt_ps['gmag']
            e_ps_gmag = tgt_ps['e_gmag']
            ps_rmag = tgt_ps['rmag']
            e_ps_rmag = tgt_ps['e_rmag']
            ps_imag = tgt_ps['imag']
            e_ps_imag = tgt_ps['e_imag']
            ps_zmag = tgt_ps['zmag']
            e_ps_zmag = tgt_ps['e_zmag']
            ps_ymag = tgt_ps['ymag']
            e_ps_ymag = tgt_ps['e_ymag']

            #if (ps_gmag != 99.0) & (e_ps_gmag > 0):
            #    SED['PS_g'] = (ps_gmag,e_ps_gmag)
            #if (ps_rmag != 99.0) & (e_ps_rmag > 0):
            #    SED['PS_r'] = (ps_rmag,e_ps_rmag)
            #if (ps_imag != 99.0) & (e_ps_imag > 0):
            #    SED['PS_i'] = (ps_imag,e_ps_imag)
            #if (ps_zmag != 99.0) & (e_ps_zmag > 0):
            #    SED['PS_z'] = (ps_zmag,e_ps_zmag)
            #if (ps_ymag != 99.0) & (e_ps_ymag > 0):
            #    SED['PS_y'] = (ps_ymag,e_ps_ymag)

        if cluster == 'M35':
                dance_mags = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/M35_DANCe_phot.csv',index_col = 'source_id')
                wocs_mags = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/M35_wocs_phot_topcat.csv',index_col = 'source_id')
                tgt_dance = dance_mags.loc[sid]
                tgt_wocs = wocs_mags.loc[sid]

                sdss_gmag = tgt_dance['gmag']
                e_sdss_gmag = tgt_dance['e_gmag']
                sdss_rmag = tgt_dance['rmag']
                e_sdss_rmag = tgt_dance['e_rmag']
                sdss_imag = tgt_dance['imag']
                e_sdss_imag = tgt_dance['e_imag']
                sdss_zmag = tgt_dance['zmag']
                e_sdss_zmag = tgt_dance['e_zmag']

                #if (~np.isnan(sdss_gmag)) & (~np.isnan(e_sdss_gmag)):
                #    SED['SDSS_g'] = (sdss_gmag,e_sdss_gmag)
                #if (~np.isnan(sdss_rmag)) & (~np.isnan(e_sdss_rmag)):
                #    SED['SDSS_r'] = (sdss_rmag,e_sdss_rmag)
                #if (~np.isnan(sdss_imag)) & (~np.isnan(e_sdss_imag)):
                #    SED['SDSS_i'] = (sdss_imag,e_sdss_imag)
                #if (~np.isnan(sdss_zmag)) & (~np.isnan(e_sdss_zmag)):
                #    SED['SDSS_z'] = (sdss_zmag,e_sdss_zmag)

                wocs_bmag = tgt_wocs['Bmag']
                e_wocs_bmag = tgt_wocs['e_Bmag']
                wocs_vmag = tgt_wocs['Vmag']
                e_wocs_vmag = tgt_wocs['e_Vmag']
                wocs_rmag = tgt_wocs['Rmag']
                e_wocs_rmag = tgt_wocs['e_Rmag']
                wocs_imag = tgt_wocs['Imag']
                e_wocs_imag = tgt_wocs['e_Imag']

                if (~np.isnan(wocs_bmag)) & (~np.isnan(e_wocs_bmag)):
                    if (e_wocs_bmag > 0):
                        SED['B'] = (wocs_bmag,e_wocs_bmag)
                if (~np.isnan(wocs_vmag)) & (~np.isnan(e_wocs_vmag)):
                    if (e_wocs_vmag > 0):
                        SED['V'] = (wocs_vmag,e_wocs_vmag)
                if (~np.isnan(wocs_rmag)) & (~np.isnan(e_wocs_rmag)):
                    if (e_wocs_rmag > 0):
                        SED['R'] = (wocs_rmag,e_wocs_rmag)
                if (~np.isnan(wocs_imag)) & (~np.isnan(e_wocs_imag)):
                    if (e_wocs_imag > 0):
                        SED['I'] = (wocs_imag,e_wocs_imag)

        if cluster == 'Hyades':
            hy_mags = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/Joner_Hyades_BVRI.csv',index_col='source_id')
            #tgt_ps = ps_mags.loc[i]
            if sid in hy_mags.index:
                tgt = hy_mags.loc[sid]
                if ~np.isnan(tgt['Vmag']):

                    vmag = tgt['Vmag']
                    e_vmag = tgt['e_Vmag']

                    bmag = tgt['B-V'] + vmag
                    e_bmag = np.sqrt(tgt['e_B-V']**2 + e_vmag**2)

                    rmag = vmag - tgt['V-Rc']
                    e_rmag = np.sqrt(tgt['V-Rc']**2 + e_vmag**2)

                    imag = vmag - tgt['V-Ic']
                    e_imag = np.sqrt(tgt['e_V-Ic']**2 + e_vmag**2)

                    if (~np.isnan(bmag)) & (~np.isnan(e_bmag)):
                        if (e_bmag > 0):
                            SED['B'] = (bmag,e_bmag)
                    if (~np.isnan(vmag)) & (~np.isnan(e_vmag)):
                        if (e_vmag > 0):
                            SED['V'] = (vmag,e_vmag)
                    if (~np.isnan(rmag)) & (~np.isnan(e_rmag)):
                        if (e_rmag > 0):
                            SED['R'] = (rmag,e_rmag)
                    if (~np.isnan(imag)) & (~np.isnan(e_imag)):
                        if (e_imag > 0):
                            SED['I'] = (imag,e_imag)


        #k = 1.08
        #if target_0['parallax'].values[0] > 0:
        plx_err = target_0['parallax_error'].values[0]

    #        if target_0['phot_g_mean_mag'].values[0] <= 13:
    #            sig = 0.021
    #        else:
    #            sig = 0.043
    #            plx_err = np.sqrt(k**2*plx_err**2 + sig**2)

            #SED['parallax'] = (target_0['parallax'].values[0] + 0.03, np.sqrt(target_0['parallax_error'].values[0]**2 + 0.03**2))
        SED['parallax'] = (target_0['parallax'].values[0], plx_err)

        #    SED['parallax'] = (target_0['parallax'].values[0] + 0.03, np.sqrt(plx_err**2 + 0.03**2))

        frame_SED = pd.DataFrame.from_dict(SED)

        #frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SEDfiles_' + cluster + '_nogaia_nops/'+np.str(target_0['source_id'].values[0])+'.csv')
        #frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SEDfiles_' + cluster + '_nogaia/'+np.str(target_0['source_id'].values[0])+'.csv')
        #frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SEDfiles_' + cluster + '/'+np.str(target_0['source_id'].values[0])+'.csv')

        #frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+ phase + '/SEDfiles_' + cluster + '_nogaia/'+np.str(target_0['source_id'].values[0])+'.csv')

        ####frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+ phase + '/' + cluster + '/SEDfiles_' + cluster + '_nogaia/'+np.str(target_0['source_id'].values[0])+'.csv')
        frame_SED.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+ phase + '/' + cluster + '/SEDfiles_' + cluster + '_'+suffix+'/'+np.str(target_0['source_id'].values[0])+'.csv')

        if len(SED.keys()) > 1:
            status[i] = 1
    return status
