from isochrones import get_ichrone, SingleStarModel
from isochrones.priors import AgePrior, AVPrior, DistancePrior, EEP_prior, FehPrior, FlatPrior, GaussianPrior, ChabrierPrior

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
    plt.rcParams['font.size']=18
    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-1])
    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-2])
    mist = get_ichrone('mist', bands=[x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')])



    #fig = plt.figure(figsize=fgsz)
    #fig = mist.isochrone(mod1.derived_samples.describe()['age'][1], mod1.derived_samples.describe()['feh'][1],AV=mod1.derived_samples.describe()['AV'][1]).plot('logTeff','logL',zorder=0,color='black',figsize=fgsz)
    #plt.scatter(mod1.derived_samples.describe()['logTeff'][1], mod1.derived_samples.describe()['logL'][1],color='red',zorder=1,s=90)
    #plt.ylabel('LogL')

    fig = mist.isochrone(mod1.derived_samples.describe()['age'][1], mod1.derived_samples.describe()['feh'][1],AV=mod1.derived_samples.describe()['AV'][1]).plot('logTeff','logL',zorder=0,color='black',figsize=fgsz,legend=None)
    plt.scatter(mod1.derived_samples.describe()['logTeff'][1], mod1.derived_samples.describe()['logL'][1],color='red',zorder=1,s=120,edgecolor='black')
    plt.ylabel('Log L')
    fig.set_xlabel(r'Log T$_{\rm eff}$')

    plt.rcParams['font.size']=12
    return fig

def plot_SED_fit(SED,mod1,fgsz=(8,8)):
    plt.rcParams['font.size']=18

    #d = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SED_wavelength_reference.csv')
    d = pd.read_csv('/home/ec2-user/SED_wavelength_reference_updated_angstrom.csv')
    zpts = pd.read_csv('/home/ec2-user/SED_flux_zeropoints_updated.csv')

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
    ##plt.scatter(weff_SED * 1e-4,obs_fluxes,color='black',marker='D',s=50)
    ##plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='black',capsize=5,linestyle='none',zorder=0)

    #plt.plot(df_derived_mags['mag'].index,df_derived_mags['mag'].values,'ro-')
    #plt.plot(df_derived_mags.index * 1e-4,df_derived_mags['flux'],'ro-')
    ##plt.scatter(df_derived_mags.index * 1e-4,df_derived_mags['flux'],color='red',s=35)

    plt.errorbar(weff_SED * 1e-4,obs_fluxes,xerr=width_SED*1e-4,yerr=obs_flux_errs,color='slategrey',capsize=7,linestyle='none',zorder=2,linewidth=2)

    #plt.plot(df_derived_mags['mag'].index,df_derived_mags['mag'].values,'ro-')
    #plt.plot(df_derived_mags.index * 1e-4,df_derived_mags['flux'],'ro-')
    plt.scatter(df_derived_mags.index * 1e-4,df_derived_mags['flux'],color='red',s=120,edgecolor='black')

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
    d = pd.read_csv('/home/ec2-user/SED_wavelength_reference_updated_angstrom.csv')

    mag_mapper = {'BP':'BP_mag', 'RP':'RP_mag', 'J':'J_mag', 'H':'H_mag', 'K':'K_mag', 'W3':'W3_mag',
                'W4':'W4_mag', 'W1':'W1_mag', 'W2':'W2_mag', 'Tycho_B':'Tycho_B_mag', 'Tycho_V':'Tycho_V_mag',
                'B':'B_mag', 'V':'V_mag', 'R':'R_mag', 'I':'I_mag', 'GALEX_FUV':'GALEX_FUV_mag', 'GALEX_NUV':'GALEX_NUV_mag', 'PS_y':'PS_y_mag',
                'PS_g':'PS_g_mag', 'PS_r':'PS_r_mag', 'PS_i':'PS_i_mag', 'PS_z':'PS_z_mag',
                'SDSS_g':'SDSS_g_mag', 'SDSS_r':'SDSS_r_mag', 'SDSS_i':'SDSS_i_mag', 'SDSS_z':'SDSS_z_mag','SDSS_u':'SDSS_u_mag',
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
