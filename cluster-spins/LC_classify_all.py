import numpy as np
from astropy.io import ascii
from astropy.table import Table
import keyboard
import time
import pandas as pd

def startfile(cluster, lc_source):
    #mems = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_allCGmems.dat')
    #mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/singlestars_68conf_m48.csv')
#M48_singlestars_68conf.csv

    ###mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_'+'singlestars_68conf.csv')

    mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'_selected.csv')
    #mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+cluster+'_ptbl_selected'+lc_source+'.csv')

    #mems = mems[(mems['proba'] > 0.5) & (~np.isnan(mems['bp_rp']))]
    #mems = mems[(mems['proba'] > 0.68) & (~np.isnan(mems['bp_rp']))]

    gaia_ids = mems['source_id'].values

    classifications = np.zeros(len(gaia_ids),dtype=str)
    classifications[:] = 'U'

    Tbl = Table(data=[gaia_ids,classifications],names=['source_id','classification']).to_pandas()

    Tbl.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'.csv')
    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    return #Tbl

def classify(cluster, lc_source, page_n, continuous=False, gaia=None):
    #Tbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat')

    Tbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'.csv')
    #Tbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'_jul08.csv')

    #Tbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat')

    if gaia != None:
        dispos = input('Enter classification for Gaia DR2 '+np.str(gaia)+': ')
        Tbl.set_index('source_id',inplace=True)
        Tbl.loc[gaia,'classification'] = dispos
        Tbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'.csv')
        #Tbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'_jul08.csv')
    else:
        n = page_n - 1
    #dispos = input('Enter clcassification for ' +np.str(n)+' : ')
        print('Gaia DR2 ' + np.str(Tbl['source_id'][n]))
        dispos = input('Enter classification : ')
        for i in range(len(dispos)):
            singlestar = dispos[i]
            #Tbl['classification'][n+i] = singlestar
            if singlestar == 'F':
                print('Last classified page is ' + np.str(n))
                return
            else:
                Tbl.loc[n+i,'classification'] = singlestar

        Tbl.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'.csv')
        #Tbl.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_lc_classifications_'+lc_source+'_jul08.csv')

        print('Last classified page is ' + np.str(n+i+1))#+ ', page ' + np.str(n+i+1))
        print()
    #print('Enter classification for ' +np.str(n)+' : ')
    #time.sleep(5)]


    #Tbl['classification'].dtype = str
    #Tbl['classification'][n] = dispos
    #print(Tbl[n])

    #if (keyboard.is_pressed('C')) or (keyboard.is_pressed('X')) or (eyboard.is_pressed('Z'):
    #key = keyboard.read_key()
    #print (keyboard.is_pressed('C'))

    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat',format='ascii',overwrite=True)

    #print('Gaia DR2 ' + np.str(Tbl['source_id'][n+i]))
    #n += 1
    #classify(n)
    #print(n), print(len(Tbl))
    if gaia == None:
        if (continuous) & (n < len(Tbl)-1):
            classify(cluster, lc_source, page_n+len(dispos), continuous=True)
        else:
            print('Finished classifying.')
            return
    else:
        return #Tbl
