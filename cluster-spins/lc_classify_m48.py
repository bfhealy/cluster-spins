import numpy as np
from astropy.io import ascii
from astropy.table import Table
import keyboard
import time
import pandas as pd

def startfile():
    #mems = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_allCGmems.dat')
    mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/singlestars_68conf_m48.csv')

    #mems = mems[(mems['proba'] > 0.5) & (~np.isnan(mems['bp_rp']))]
    #mems = mems[(mems['proba'] > 0.68) & (~np.isnan(mems['bp_rp']))]

    gaia_ids = mems['source_id'].values

    classifications = np.zeros(len(gaia_ids),dtype=str)
    classifications[:] = 'U'

    Tbl = Table(data=[gaia_ids,classifications],names=['source_id','classification'])

    Tbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    return Table

def classify(n):
    Tbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat')
    #Tbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat')

    #dispos = input('Enter clcassification for ' +np.str(n)+' : ')
    dispos = input('Enter classification : ')
    for i in range(len(dispos)):
        singlestar = dispos[i]
        Tbl['classification'][n+i] = singlestar
    #print('Enter classification for ' +np.str(n)+' : ')
    #time.sleep(5)]


    #Tbl['classification'].dtype = str
    #Tbl['classification'][n] = dispos
    #print(Tbl[n])

    #if (keyboard.is_pressed('C')) or (keyboard.is_pressed('X')) or (eyboard.is_pressed('Z'):
    #key = keyboard.read_key()
    #print (keyboard.is_pressed('C'))

    #Tbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_lc_classifications_cdips.dat',format='ascii',overwrite=True)
    Tbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/M48_lc_classifications_cdips.dat',format='ascii',overwrite=True)

    print('Last classified index is n = ' + np.str(n+i) + ', page ' + np.str(n+i+1))
    print('Gaia DR2 ' + np.str(Tbl['source_id'][n+i]))
    #n += 1
    #classify(n)
    return Tbl
