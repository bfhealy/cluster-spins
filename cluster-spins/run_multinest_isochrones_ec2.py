import sys
sys.path.append('/home/ec2-user/cluster-spins/')

from isochrones import get_ichrone, SingleStarModel
from isochrones.priors import FehPrior,LogNormalPrior,AgePrior, AVPrior, DistancePrior, EEP_prior, FehPrior, FlatPrior, GaussianPrior, ChabrierPrior

import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.size']=18

#Gaia.login(user='bhealy',password='exAtm17$')

import pandas as pd
from astropy.io import fits,ascii
from astropy.table import Table

import make_seds_ec2

import glob

import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.mast import Catalogs

mnest_files = glob.glob('/home/ec2-user/multinest_runs/*.hd5')
mnest_ids = []
for i in range(len(mnest_files)):
    mnest_ids += [mnest_files[i].split('/')[-1].split('.')[0].split('_')[0]]
mnest_ids = np.array(mnest_ids)
#targets=ascii.read('/home/ec2-user/targets/joinedfinalresults_ultimate_revised.dat').to_pandas()
#targets=ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_2/joinedfinalresults_ultimate_revised.dat').to_pandas()
#targets = pd.read_csv('/home/ec2-user/targets/pleiades_mems_periods_allvsini.csv')
#targets = pd.read_csv('/home/ec2-user/targets/praesepe_mems_periods_allvsini_noduplicates.csv')
#targets = pd.read_csv('/home/ec2-user/targets/pleiades_mems_periods_allvsini.csv')
file = glob.glob('/home/ec2-user/targets/*.csv')
targets=pd.read_csv(file[0])

for i in range(len(targets)):
#for i in range(3,4):
    print(np.str(i+1) + '/' + np.str(len(targets)))

    sid = np.str(targets.iloc[i]['source_id'])
    if sid in mnest_ids:
        print('Fit already.')
        continue
    else:
        print('New star - fitting.')
        sedfile = glob.glob('/home/ec2-user/SEDfiles/'+sid+'.csv')[0]
        #sedfile = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SEDfiles_NGC_2516/'+sid+'.csv')[0]

        SED = pd.read_csv(sedfile,index_col=0)
        SED = SED.to_dict(orient='list')

        #SED['Teff'] = [targets.iloc[i]['teff'], 3*targets.iloc[i]['e_teff']]
        #print(SED['Teff'])

        #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-1])
        #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-2])
        mist = get_ichrone('mist', bands=[x for x in SED.keys() if (x != 'parallax') & (x != 'Teff') & (x != 'maxAV')])

        mod1=SingleStarModel(mist)
        mod1.kwargs = SED

        #print(SED)

        maxdist = 1/(SED['parallax'][0] - 5*SED['parallax'][1])*1e3
        dist = 1/SED['parallax'][0] * 1e3
        dist_1sig_lo = 1/(SED['parallax'][0] + SED['parallax'][1]) * 1e3
        dist_1sig_hi = 1/(SED['parallax'][0] - SED['parallax'][1]) * 1e3
        sig_dist = np.mean([dist_1sig_hi - dist, dist -dist_1sig_lo])
        #print(maxdist)

        #Priors
        #mod1.set_prior(age=GaussianPrior(8.097,0.08), feh = GaussianPrior(-0.2,0.4,bounds=[-0.5,0.5]), av = GaussianPrior(0.22,0.4), distance =DistancePrior(maxdist))
        #mod1.set_prior(age=GaussianPrior(8.4,0.08,bounds=[8.25,8.55]), feh = GaussianPrior(-0.2,0.4,bounds=[-0.5,0.5]), av = FlatPrior((0,0.8)), distance = FlatPrior((394,424)))

        #Pleiades
        #Age: 110 - 160 Myr (Gossage et al. 2018)
        #Fe/H: 0.042 (Soderblom et al. 2009)
        #AV: < 0.8 (Bayestar19)
        #Distance: 135.6 pc (CG2018) ± 5 pc (Abramson 2018)
        #mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.95,8.3]), feh = GaussianPrior(0.042,0.1,bounds=[-0.25,0.25]), av = GaussianPrior(0.6,0.1), distance = FlatPrior((120,150)))
        #mod1.set_bounds(AV=(0,0.8))
        #Praesepe
        #Age: 650-800 Myr (multiple)
        #Fe/H: 0.12 - 0.21 - 0.27 (Boesgaard et al. 2013, D'orazi et al. 2019, Pace et al 2008)
        #AV: < 0.1
        #Distance: 185.5 pc (CG18)
        #mod1.set_prior(age=GaussianPrior(8.813,0.1,bounds=[8.7,8.95]), feh = GaussianPrior(0.2,0.2,bounds=[-0.2,0.5]), av = GaussianPrior(0.05,0.1), distance = FlatPrior((170, 200)))
        #mod1.set_bounds(AV=(0,0.3))
        #M35
        #Age: 134-161 Myr (Meibom et al. 2009)
        #Fe/H: -0.21 - -0.15 (Barrado et al. 2001, Anthony-Twarog et al. 2018)
        #AV: 0.71
        #Distance: 862.4 pc (CG18)
        #mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(-0.18,0.1,bounds=[-0.7,0.3]), av = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
        mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(-0.18,0.05,bounds=[-0.5,0.2]), av = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

        #mod1.set_bounds(AV=(0.7,3))
        mod1.set_bounds(AV=(0.,3))

        #mod1.set_bounds(AV=(0,0.8))

        #feh = FlatPrior((-0.4,0.3))
        #av = FlatPrior((0.,0.8))
        #age=FlatPrior((8,8.2))

        #mod1.fit(basename=sid,refit=True)
        mod1.fit(basename=sid+'_nogaia',refit=True)

        #mod1.save_hdf('/home/ec2-user/multinest_runs/'+sid+'.hd5')
        mod1.save_hdf('/home/ec2-user/multinest_runs/'+sid+'_nogaia.hd5')


        cornerfig = mod1.corner_physical()
        plt.show()
        cornerfig.savefig('/home/ec2-user/corner_plots/'+sid+'.pdf',bbox_inches='tight')

        icfig = make_seds_ec2.plot_isochrone_fit(SED,mod1)
        plt.show()
        icfig.get_figure().savefig('/home/ec2-user/ichrone_plots/'+sid+'.pdf',bbox_inches='tight')

        sedfig=make_seds_ec2.plot_SED_fit(SED,mod1)
        plt.show()
        sedfig.savefig('/home/ec2-user/sed_plots/'+sid+'.pdf',bbox_inches='tight')
