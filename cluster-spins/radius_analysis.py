from astropy.coordinates import SkyCoord
from isochrones import SingleStarModel
import numpy as np
import pandas as pd
from dustmaps.bayestar import BayestarWebQuery
from dustmaps.sfd import SFDWebQuery
import astropy.units as u
import glob

def query_dustmaps(cluster, map='bayestar',mode='mean'):

    singlestars_edr3 = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_singlestars_EDR3.csv')

    if map == 'bayestar':
        bayestar = BayestarWebQuery()

        ra_arr = np.arange(np.min(singlestars_edr3['ra']),np.max(singlestars_edr3['ra']),3.4/60)
        dec_arr = np.arange(np.min(singlestars_edr3['dec']),np.max(singlestars_edr3['dec']),3.4/60)

        ra_dec = np.array(np.meshgrid(ra_arr,dec_arr))

        coords = SkyCoord(ra_dec[0]*u.deg, ra_dec[1]*u.deg, distance=1/np.mean(singlestars_edr3['parallax'])*1e3*u.pc, frame='icrs')

        reddening = bayestar(coords, mode=mode)

        av_all_radec = .982*reddening*3.1

    elif map == 'sfd':
        qry = SFDWebQuery()

        ra_arr = np.arange(np.min(singlestars_edr3['ra']),np.max(singlestars_edr3['ra']),3.4/60)
        dec_arr = np.arange(np.min(singlestars_edr3['dec']),np.max(singlestars_edr3['dec']),3.4/60)

        ra_dec = np.array(np.meshgrid(ra_arr,dec_arr))

        coords = SkyCoord(ra_dec[0]*u.deg, ra_dec[1]*u.deg, distance=1/np.mean(singlestars_edr3['parallax'])*1e3*u.pc, frame='icrs')

        reddening = qry(coords)

        av_all_radec = reddening*3.1

    return av_all_radec

def radius_examination(cluster,suffix='nogaia',file_end=''):

    path = '/Users/bhealy/Documents/PhD_Thesis/Phase_3/multinest_ec2/'
    #targets = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48/M48_singlestars_EDR3.csv')
    fullpath = path + 'multinest_runs_' + cluster + '_' + suffix + '_ec2/*.hd5'
    files = glob.glob(fullpath)
    #print(fullpath)

    rad_ng = np.zeros(len(files))
    e_rad_ng = np.zeros(len(files))
    sid_ng = np.zeros(len(files),dtype=np.int64)
    z_ng = np.zeros(len(files))
    d_ng = np.zeros(len(files))
    e_d_ng = np.zeros(len(files))
    teff_ng = np.zeros(len(files))
    e_teff_ng = np.zeros(len(files))
    feh_ng = np.zeros(len(files))
    e_feh_ng = np.zeros(len(files))
    av_ng = np.zeros(len(files))
    e_av_ng = np.zeros(len(files))
    age_ng = np.zeros(len(files))
    e_age_ng = np.zeros(len(files))
    logg_ng = np.zeros(len(files))
    e_logg_ng = np.zeros(len(files))
    m_ng = np.zeros(len(files))
    e_m_ng = np.zeros(len(files))

    for i in range(len(files)):
        print(np.str(i+1)+'/'+np.str(len(files)))
        #print(files[i])

        mod = SingleStarModel.load_hdf(files[i])
        sid_ng[i] = np.int64(files[i].split('/')[-1][:-4].split('_')[0])
        #print(sid_ng[i])

        rad_ng[i] = mod.derived_samples.describe()['radius'].loc['mean']
        e_rad_ng[i] = mod.derived_samples.describe()['radius'].loc['std']

        #z_ng[i] = mod.mnest_analyzer.get_stats()['nested sampling global log-evidence']
        ztbl = pd.read_table(path+'chains/'+np.str(sid_ng[i])+'_'+suffix+'stats.dat',header=None,delimiter=':')
        z_ng[i] = np.float(ztbl.loc[0][1].strip().split('+/-')[0].strip())

        d_ng[i] = mod.derived_samples.describe()['distance'].loc['mean']
        e_d_ng[i] = mod.derived_samples.describe()['distance'].loc['std']

        teff_ng[i] = mod.derived_samples.describe()['Teff'].loc['mean']
        e_teff_ng[i] = mod.derived_samples.describe()['Teff'].loc['std']

        feh_ng[i] = mod.derived_samples.describe()['feh'].loc['mean']
        e_feh_ng[i] = mod.derived_samples.describe()['feh'].loc['std']

        av_ng[i] = mod.derived_samples.describe()['AV'].loc['mean']
        e_av_ng[i] = mod.derived_samples.describe()['AV'].loc['std']

        age_ng[i] = mod.derived_samples.describe()['age'].loc['mean']
        e_age_ng[i] = mod.derived_samples.describe()['age'].loc['std']

        logg_ng[i] = mod.derived_samples.describe()['logg'].loc['mean']
        e_logg_ng[i] = mod.derived_samples.describe()['logg'].loc['std']

        m_ng[i] = mod.derived_samples.describe()['mass'].loc['mean']
        e_m_ng[i] = mod.derived_samples.describe()['mass'].loc['std']

    df_ng = pd.DataFrame({'source_id':sid_ng,'radius_ng':rad_ng, 'e_radius_ng':e_rad_ng,
                      'teff_ng':teff_ng,'e_teff_ng':e_teff_ng, 'feh_ng':feh_ng,
                      'e_feh_ng':e_feh_ng,'av_ng':av_ng, 'e_av_ng':e_av_ng,
                      'age_ng':age_ng, 'e_age_ng':e_age_ng, 'logg_ng':logg_ng, 'e_logg_ng':e_logg_ng, 'd_ng':d_ng,
                      'e_d_ng':e_d_ng, 'm_ng':m_ng, 'e_m_ng':e_m_ng, 'z_ng':z_ng})

    df_ng.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/isochrones_output_dataframe_'+cluster+'_allmems_EDR3'+file_end+'.csv')

    return df_ng
