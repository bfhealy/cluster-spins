import sys
sys.path.append('/home/ec2-user/cluster-spins/')

#import os
#cmd = "export HDF5_USE_FILE_LOCKING=FALSE"
#os.system(cmd)

from isochrones import get_ichrone, SingleStarModel
from isochrones.priors import FehPrior,LogNormalPrior,AgePrior, AVPrior, DistancePrior, EEP_prior, FehPrior, FlatPrior, GaussianPrior, ChabrierPrior

import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.size']=18
import multiprocessing
#Gaia.login(user='bhealy',password='exAtm17$')

import pandas as pd
from astropy.io import fits,ascii
from astropy.table import Table

import make_seds_ec2

import glob

import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.mast import Catalogs

def run_multinest_isochrones_ec2(cluster):
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
            if cluster == 'Pleiades':
                #mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.875,8.30103]), feh = GaussianPrior(0.042,0.021,bounds=[-0.2,0.3]), av = GaussianPrior(0.578,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.875,8.30103]), feh = GaussianPrior(-0.048743534,0.17925346), AV = GaussianPrior(0.578,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                mod1.set_bounds(AV=(0.,1.5))
                #mod1.set_bounds(AV=(0.,3.0))
            #Praesepe
            #Age: 650-800 Myr (multiple)
            #Fe/H: 0.12 - 0.21 - 0.27 (Boesgaard et al. 2013, D'orazi et al. 2019, Pace et al 2008)
            #AV: < 0.1
            #Distance: 185.5 pc (CG18)
            elif cluster == 'Praesepe':
                #mod1.set_prior(age=GaussianPrior(8.87,0.05,bounds=[8.72,9.02]), feh = GaussianPrior(0.2,0.06,bounds=[-0.2,0.5]), av = GaussianPrior(0.05,0.05), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                mod1.set_prior(age=GaussianPrior(8.87,0.05,bounds=[8.72,9.02]), feh = GaussianPrior(0.135085656,0.16619438), AV = GaussianPrior(0.05,0.045), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                mod1.set_bounds(AV=(0,1))
            #M35
            #Age: 134-161 Myr (Meibom et al. 2009)
            #Fe/H: -0.21 - -0.15 (Barrado et al. 2001, Anthony-Twarog et al. 2018)
            #AV: 0.71
            #Distance: 862.4 pc (CG18)
            elif cluster == 'M35':
                #mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(-0.18,0.05,bounds=[-0.5,0.2]), av = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(0.076366142,0.15111486,bounds=[-0.5,0.2]), AV = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                mod1.set_bounds(AV=(0.,3))

            #M48
            #Age: 450 ± 50 (Barnes et al. 2015)
            #Fe/H: −0.063 ± 0.007 (Sun et al. 2020)
            #AV: 0.18 ± 0.02 (Bayestar19)
            #Distance: 747 pc (EDR3)
            elif cluster == 'M48':
                mod1.set_prior(age=GaussianPrior(8.6532125138,0.05,bounds=[8.4,8.9]), feh = GaussianPrior(-0.063,0.007,bounds=[-0.3,0.3]), AV = GaussianPrior(0.18,0.02), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                mod1.set_bounds(AV=(0.,1.5))

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
    return

def fit_single_star(sid):
        clusterfile = glob.glob('/home/ec2-user/*.txt')
        ###cluster = clusterfile[0].split('/')[3].split('_')[0]
        splitfile = clusterfile[0].split('/')[3].split('_')
        if len(splitfile) < 3:
            cluster = splitfile[0]
        elif len(splitfile) >= 3:
            cluster = splitfile[0]+'_'+splitfile[1]

        suffix = splitfile[-1].split('.')[0]
        #cluster = clusterfile[0].split('/')[3].split('_')[0]
        #print(cluster)
        try:
            sedfile = glob.glob('/home/ec2-user/SEDfiles/'+sid+'.csv')[0]

        #sedfile = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_2/SEDfiles_NGC_2516/'+sid+'.csv')[0]

            SED = pd.read_csv(sedfile,index_col=0)
            SED = SED.to_dict(orient='list')
            #print(len(SED.keys()))

            if len(SED.keys()) <= 1:
                print('SED file missing magnitudes - skipping.')
            else:

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
                if cluster == 'Pleiades':
                    #mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.875,8.30103]), feh = GaussianPrior(0.042,0.021,bounds=[-0.2,0.3]), av = GaussianPrior(0.578,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    ##mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.875,8.30103]), feh = GaussianPrior(-0.048743534,0.17925346,bounds=[-0.5,0.5]), AV = GaussianPrior(0.578,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.13,0.08,bounds=[7.875,8.30103]), feh = GaussianPrior(-0.048743534,0.17925346,bounds=[-0.5,0.5]), AV = GaussianPrior(0.27,0.29), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    #mod1.set_bounds(AV=(0.,1.5))
                    mod1.set_bounds(AV=(-1.,3))

                #Praesepe
                #Age: 650-800 Myr (multiple)
                #Fe/H: 0.12 - 0.21 - 0.27 (Boesgaard et al. 2013, D'orazi et al. 2019, Pace et al 2008)
                #AV: < 0.1
                #Distance: 185.5 pc (CG18)
                elif cluster == 'Praesepe':
                    #mod1.set_prior(age=GaussianPrior(8.87,0.05,bounds=[8.72,9.02]), feh = GaussianPrior(0.2,0.06,bounds=[-0.2,0.5]), av = GaussianPrior(0.05,0.05), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    ##mod1.set_prior(age=GaussianPrior(8.87,0.05,bounds=[8.72,9.02]), feh = GaussianPrior(0.135085656,0.16619438, bounds=[-0.5,0.5]), AV = GaussianPrior(0.05,0.05), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.87,0.05,bounds=[8.72,9.02]), feh = GaussianPrior(0.135085656,0.16619438, bounds=[-0.5,0.5]), AV = GaussianPrior(0.013,0.039), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,1))
                #M35
                #Age: 134-161 Myr (Meibom et al. 2009)
                #Fe/H: -0.21 - -0.15 (Barrado et al. 2001, Anthony-Twarog et al. 2018)
                #AV: 0.71
                #Distance: 862.4 pc (CG18)
                elif cluster == 'M35':
                    #mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(-0.18,0.05,bounds=[-0.5,0.2]), av = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(0.076366142,0.15111486,bounds=[-0.5,0.5]), AV = GaussianPrior(0.713,0.1), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.176,0.08,bounds=[8,8.30103]), feh = GaussianPrior(0.076366142,0.15111486,bounds=[-0.5,0.5]), AV = GaussianPrior(0.69,0.15), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(0.,3))

                #M48*?
                #Age: 450 ± 50 (Barnes et al. 2015)
                #Fe/H: -0.02941617158059184,0.1036785486235826 (LAMOST/Hamer) −0.063 ± 0.007 (Sun et al. 2020)
                #AV: 0.18 ± 0.02 (Bayestar19)
                #Distance: 747 pc (EDR3)
                elif cluster == 'M48':
                    #mod1.set_prior(age=GaussianPrior(8.6532125138,0.05,bounds=[8.4,8.9]), feh = GaussianPrior(-0.063,0.007,bounds=[-0.3,0.3]), av = GaussianPrior(0.18,0.02), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0.,1.5))

                    ###mod1.set_prior(age=GaussianPrior(8.6532125138,0.05,bounds=[8.4,8.9]), feh = GaussianPrior(-0.02941617158059184,0.1036785486235826,bounds=[-0.5,0.5]), AV = GaussianPrior(0.14489202, 0.06206673), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.65,0.1,bounds=[8.,9]), feh = GaussianPrior(-0.02941617158059184,0.1036785486235826,bounds=[-0.5,0.5]), AV = GaussianPrior(0.14489202, 0.06206673), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,2.0))
                    #test
                #Blanco 1*
                #Age: 100-120 Myr (Gaia Collaboration et al. 2018)
                #Fe/H: -0.08661912062692306, 0.12921555365017445 (GALAH DR3)
                #AV: < 0.1 (SFD 2011)
                #Distance
                elif cluster == 'Blanco_1':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    mod1.set_prior(age=GaussianPrior(8.06,0.06,bounds=[7.5, 8.5]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), AV = GaussianPrior(0.05,0.05), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_bounds(AV=(-1,0.3))

                #NGC 2422*
                #Age: 155 ± 20 Myr (Cummings & Kalirai 2018)
                #Fe/H: -0.07508196721311479, 0.20563474662391293 (Bailey 2017)
                #AV: 0.28362706 0.097286016 (Bayestar19)
                #Distance
                elif cluster == 'NGC_2422':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(8.190,0.05,bounds=[7.9, 8.5]), feh = GaussianPrior(-0.07508196721311479, 0.20563474662391293, bounds=[-0.7,0.5]), AV = GaussianPrior(0.28362706, 0.097286016), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.190,0.1,bounds=[7.5, 8.6]), feh = GaussianPrior(-0.07508196721311479, 0.20563474662391293, bounds=[-0.7,0.5]), AV = GaussianPrior(0.28362706, 0.097286016), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,1))

                #NGC 2547*
                #Age: 35 ± 3 Myr (Jeffries & Oliviera 2005)
                #Fe/H: -0.03, 0.06 (Magrini & Randich 2015)
                #AV: 0.372, 0.155 (Naylor & Jeffries 2006)
                #Distance
                elif cluster == 'NGC_2547':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.544,0.05,bounds=[7.3, 7.8]), feh = GaussianPrior(-0.03, 0.06, bounds=[-0.5,0.5]), AV = GaussianPrior(0.372, 0.155), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7.544,0.1,bounds=[7., 8]), feh = GaussianPrior(-0.03, 0.06, bounds=[-0.5,0.5]), AV = GaussianPrior(0.372, 0.155), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,1.5))

                #NGC 2264*
                #Age: 5.5 (Turner 2011) ± 3 Myr (Sung & Bessel 2010)
                    # Flat prior 2-9 Myr
                #Fe/H: -0.09±0.05 (Magrini & Randich 2015)
                #AV: 2.6, 0.76 (Bayestar19), 0.075, 0.06 (Turner 2011)
                #Distance
                elif cluster == 'NGC_2264':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #0.34
                    #0.19

                    #mod1.set_prior(age=GaussianPrior(6.74,0.25,bounds=[6, 7.1]), feh = GaussianPrior(-0.09,0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.075, 0.06), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_prior(age=GaussianPrior(6.74,0.25,bounds=[6, 7.1]), feh = GaussianPrior(-0.09,0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.075, 0.06), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((6.301, 6.954)), feh = GaussianPrior(-0.09,0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.075, 0.06), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(6.74, 0.35, bounds=[5.5, 8]), feh = GaussianPrior(-0.09,0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.075, 0.06), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,4.5))

                #Pozzo 1
                #Age: 18-21 Myr (Jeffries et al. 2017)
                #Fe/H: (-0.04 ± 0.05) (−0.057 ± 0.018) (Spina et al. 2014)
                #AV: 0.131, 0.055 (Jeffries+ 2009)
                #Distance
                elif cluster == 'Pozzo_1':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((7.2553, 7.3222)), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7,0.3,bounds=[6, 8]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,1.))

                #Collinder 69*
                #Age: 5-20 Myr (Bayo et al. 2011) or 5 ± 2 Myr (Barrado y Navascues et al. 2004)
                #Fe/H: -0.1688101428571428, 0.0927522076489856 (SDSS/APOGEE DR17)
                #AV: 0.49162447 0.22306725 (Bayestar19)
                #Distance
                elif cluster == 'Collinder_69':
                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ####mod1.set_prior(age=FlatPrior((6.6, 7.3)), feh = GaussianPrior(-0.1688101428571428, 0.0927522076489856, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(6.7,0.3,bounds=[5.8, 8]), feh = GaussianPrior(-0.1688101428571428, 0.0927522076489856, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    #mod1.set_prior(age=FlatPrior((6.5, 6.85)), feh = GaussianPrior(-0.17049309552238798, .09121834298307405, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,2))

                #NGC 2516
                #Age: 140 Myr (Paper 1)
                #Fe/H:
                #AV: 0.22, 0.4 (Paper 1)
                elif cluster == 'NGC_2516':

                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    ##mod1.set_prior(age=FlatPrior((6.6, 7.1)), feh = GaussianPrior(-0.17049309552238798, .09121834298307405, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(8.146, 0.084, bounds=[7.7, 8.5]), feh = GaussianPrior(-0.2, 0.4, bounds=[-2,0.5]), AV = GaussianPrior(0.22, 0.4), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-2,2))

                #BH 56
                #Age: 17.4 Myr [8-26 Myr] (1 star, Kharchenko et al. 2005)
                #Fe/H: -0.11561123091272728, 0.10982775236706674 (SDSS/APOGEE DR17)
                #AV: 0.20 (Kharchenko et al. 2005)
                elif cluster == 'BH_56':

                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((6.9, 7.4)), feh = GaussianPrior(-0.11561123091272728, 0.10982775236706674, bounds=[-0.7,0.5]), AV = GaussianPrior(0.2, 0.2), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7.24, 0.3, bounds=[6, 8]), feh = GaussianPrior(-0.11561123091272728, 0.10982775236706674, bounds=[-0.7,0.5]), AV = GaussianPrior(0.2, 0.2), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    #mod1.set_prior(age=GaussianPrior(7.24, 0.01, bounds=[6.9, 7.6]), feh = GaussianPrior(-0.11561123091272728, 0.10982775236706674, bounds=[-0.7,0.5]), AV = GaussianPrior(0.2, 0.2), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-2,2.5))

                #ASCC 16
                #Age: ~10 Myr [5-15 Myr] (Kharchenko et al. 2013)
                #Fe/H: -0.09260771381959998, 0.08721131948508637 (SDSS/APOGEE DR17)
                #AV: 0.1879923 0.06489724 (Bayestar19)
                elif cluster == 'ASCC_16':

                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((6.7, 7.2)), feh = GaussianPrior(-0.09260771381959998, 0.08721131948508637, bounds=[-0.7,0.5]), AV = GaussianPrior(0.1879923, 0.06489724), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7., 0.3, bounds=[6., 8]), feh = GaussianPrior(-0.09260771381959998, 0.08721131948508637, bounds=[-0.7,0.5]), AV = GaussianPrior(0.1879923, 0.06489724), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    #mod1.set_prior(age=GaussianPrior(7., 0.2, bounds=[6.7, 7.5]), feh = GaussianPrior(-0.09260771381959998, 0.08721131948508637, bounds=[-0.7,0.5]), AV = GaussianPrior(0.1879923, 0.06489724), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1,1.5))

                #ASCC 19*
                #Age: ~30 Myr [15-45 Myr] (Kharchenko et al. 2013)
                #Fe/H: -0.12924870934883723, 0.1243393536380701 (SDSS/APOGEE DR17)
                #AV: 0.49388105, 0.280083 (Bayestar19)
                elif cluster == 'ASCC_19':

                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    ##mod1.set_prior(age=FlatPrior((6.6, 7.1)), feh = GaussianPrior(-0.17049309552238798, .09121834298307405, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((7.2, 7.65)), feh = GaussianPrior(-0.12924870934883723, 0.1243393536380701, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49388105, 0.280083), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7.5, 0.5, bounds=[6,8]), feh = GaussianPrior(-0.12924870934883723, 0.1243393536380701, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49388105, 0.280083), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1.5,3.5))

                #Gulliver 6*
                #Age: 10-50 Myr (this work)?
                #Fe/H: -0.14590463888888888, 0.11350522580459868 (SDSS/APOGEE DR17)
                #AV: 0.6430089, 0.36486796 (Bayestar19)
                elif cluster == 'Gulliver_6':

                    #print('Blanco_1')
                    #mod1.set_prior(age=GaussianPrior(8.06,0.04,bounds=[7.7, 8.3]), feh = GaussianPrior(-0.08661912062692306, 0.12921555365017445, bounds=[-0.5,0.5]), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    #mod1.set_bounds(AV=(0,0.25))

                    #mod1.set_prior(age=GaussianPrior(7.3,0.05,bounds=[7.1, 7.5]), feh = GaussianPrior(-0.04, 0.05, bounds=[-0.5,0.5]), AV = GaussianPrior(0.131, 0.055), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    ##mod1.set_prior(age=FlatPrior((6.6, 7.1)), feh = GaussianPrior(-0.17049309552238798, .09121834298307405, bounds=[-0.7,0.5]), AV = GaussianPrior(0.49162447, 0.22306725), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    ###mod1.set_prior(age=FlatPrior((7., 7.7)), feh = GaussianPrior(-0.14590463888888888, 0.11350522580459868, bounds=[-0.7,0.5]), AV = GaussianPrior(0.6430089, 0.36486796), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))
                    mod1.set_prior(age=GaussianPrior(7.3, 0.1, bounds=[6,8]), feh = GaussianPrior(-0.14590463888888888, 0.11350522580459868, bounds=[-0.7,0.5]), AV = GaussianPrior(0.6430089, 0.36486796), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1.5,3.5))

                #Alpha Per
                #Age: 50-75 Myr (Basri & Martin 1998)
                #Fe/H: -0.07557962452260272, 0.1440535175579019(SDSS/APOGEE DR17)
                #AV: 0.017124383, 0.062144116 (Bayestar19)
                elif cluster == 'Alpha_Per':

                    mod1.set_prior(age=GaussianPrior(7.8, 0.1, bounds=[6,8.5]), feh = GaussianPrior(-0.07557962452260272, 0.1440535175579019, bounds=[-0.7,0.5]), AV = GaussianPrior(0.017124383, 0.062144116), distance = GaussianPrior(dist, sig_dist, bounds=[dist - 5*sig_dist, dist + 5*sig_dist]))

                    mod1.set_bounds(AV=(-1.5,1.5))


                #mod1.fit(basename=sid,refit=True)
                #mod1.fit(basename=sid+'_nogaia',refit=True)
                mod1.fit(basename=sid+'_'+suffix,refit=True)

                #mod1.save_hdf('/home/ec2-user/multinest_runs/'+sid+'.hd5')
                #mod1.save_hdf('/home/ec2-user/multinest_runs/'+sid+'_nogaia.hd5')
                mod1.save_hdf('/home/ec2-user/multinest_runs/'+sid+'_'+suffix+'.hd5')

                cornerfig = mod1.corner_physical()
                plt.show()
                cornerfig.savefig('/home/ec2-user/corner_plots/'+sid+'.pdf',bbox_inches='tight')
                plt.close()

                icfig = make_seds_ec2.plot_isochrone_fit(SED,mod1)
                plt.show()
                icfig.get_figure().savefig('/home/ec2-user/ichrone_plots/'+sid+'.pdf',bbox_inches='tight')
                plt.close()

                sedfig=make_seds_ec2.plot_SED_fit(SED,mod1)
                plt.show()
                sedfig.savefig('/home/ec2-user/sed_plots/'+sid+'.pdf',bbox_inches='tight')
                plt.close()

        except IndexError:
            print('No SED file found.')

#def multiprocess_run_multinest_isochrones_ec2(cluster):
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

if __name__ == '__main__':
    #processes = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    sids = []
    for i in range(len(targets)):
        target = targets.iloc[i]
        print(np.str(i+1) + '/' + np.str(len(targets)))
        sid = np.str(targets.iloc[i]['source_id'])
        if sid in mnest_ids:
            print('Fit already.')
            continue
        else:
            print('New star - fitting.')
            sids += [sid]
            #p = multiprocessing.Process(target=fit_single_star, args=(sid,cluster,))
    pool.map(fit_single_star, sids)
    pool.close()
    pool.join()

    #processes.append(p)
    #p.start()

    #for process in processes:
    #    process.join()


#    return
