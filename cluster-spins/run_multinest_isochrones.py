from isochrones import get_ichrone, SingleStarModel
from isochrones.priors import FehPrior,LogNormalPrior,AgePrior, AVPrior, DistancePrior, EEP_prior, FehPrior, FlatPrior, GaussianPrior, ChabrierPrior

import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.size']=18

#Gaia.login(user='bhealy',password='exAtm17$')

import pandas as pd
from astropy.io import fits,ascii
from astropy.table import Table

import make_seds

import glob

import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.mast import Catalogs

for i in range(len(targets)):
#for i in range(3,4):
    print(np.str(i+1) + '/' + np.str(len(targets)))

    sid = np.str(targets.iloc[i]['source_id'])
    sedfile = glob.glob('~SEDfiles/'+sid+'.csv')[0]
    SED = pd.read_csv(sedfile,index_col=0)
    SED = SED.to_dict(orient='list')

    #SED['Teff'] = [targets.iloc[i]['teff'], 3*targets.iloc[i]['e_teff']]
    #print(SED['Teff'])

    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-1])
    #mist = get_ichrone('mist', bands=[x for x in SED.keys()][:-2])
    mist = get_ichrone('mist', bands=[x for x in SED.keys() if (x != 'parallax') & (x != 'Teff')])

    mod1=SingleStarModel(mist)
    mod1.kwargs = SED

    maxdist = 1/(SED['parallax'][0] - 5*SED['parallax'][1])*1e3
    mod1.set_prior(age=GaussianPrior(8.097,0.08), feh = GaussianPrior(-0.2,0.4,bounds=[-0.6,0.5]), av = GaussianPrior(0.22,0.4), distance =DistancePrior(maxdist))
    #feh = FlatPrior((-0.4,0.3))
    #av = FlatPrior((0.,0.8))
    #age=FlatPrior((8,8.2))

    mod1.fit(basename=sid,refit=True)
    mod1.save_hdf('/Users/bhealy/Documents/PhD_Thesis/Phase_2/NGC_2516_tests/multinest_runs_NGC_2516/'+sid+'.hd5')

    cornerfig = mod1.corner_physical()
    plt.show()
    cornerfig.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_2/NGC_2516_tests/NGC_2516_corner_plots/'+sid+'.pdf',bbox_inches='tight')

    icfig = make_seds.plot_isochrone_fit(SED,mod1)
    plt.show()
    icfig.get_figure().savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_2/NGC_2516_tests/NGC_2516_ichrone_plots/'+sid+'.pdf',bbox_inches='tight')

    sedfig=make_seds.plot_SED_fit(SED,mod1)
    plt.show()
    sedfig.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_2/NGC_2516_tests/NGC_2516_sed_plots/'+sid+'.pdf',bbox_inches='tight')
