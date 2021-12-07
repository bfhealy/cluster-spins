import emcee
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,argrelmin,peak_widths
from scipy.stats import median_absolute_deviation
import eleanor
import pandas as pd
import urllib
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import TessTargetPixelFile
import eleanor
import glob
from astropy.io import ascii,fits
from astropy.table import Table,join
import astropy.units as u
import bokeh
import sys
from scipy.signal import savgol_filter
from requests.exceptions import HTTPError
import matplotlib.gridspec as gridspec
from matplotlib import patches
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
import astropy.units as u
from astropy.visualization import SqrtStretch, LogStretch, ZScaleInterval, MinMaxInterval, SquaredStretch, AsinhStretch
from astroquery.mast import Catalogs
from astroquery.exceptions import ResolverError
import PyPDF2
from PyPDF2 import PdfFileMerger,PdfFileReader
from astropy.table import Table,join
from astropy.io import ascii
from astroquery import exceptions
from eleanor.utils import *
#from requests.exceptions import HTTPError

def download_tess_hlsp(singlestars,cluster,lc_source):
    n_lc = np.zeros(len(singlestars))
    for i in range(len(singlestars)):
    #for i in range(2,3):
        print(i)
        for x in range(5):
            if lc_source == 'CDIPS':
                try:
                    obstbl = Observations.query_criteria(objectname='Gaia DR2 '+np.str(singlestars.loc[i]['source_id']),provenance_name='CDIPS',radius=1*u.arcsec)
                    starindx = np.array([j for j in range(len(obstbl)) if np.str(singlestars.loc[i]['source_id']) in obstbl['obs_id'][j]])
                    obstbl = obstbl[starindx]
                except ConnectionError:
                    continue
                except exceptions.ResolverError:
                    n_lc[i] = 0
                    break
            elif lc_source =='PATHOS':
                try:
                    ticid = Catalogs.query_object(objectname='Gaia DR2 '+np.str(singlestars.loc[i]['source_id']),catalog='TIC',radius=1*u.arcsec)[0]['ID']
                    obstbl = Observations.query_criteria(objectname='Gaia DR2 '+np.str(singlestars.loc[i]['source_id']),provenance_name='PATHOS',radius=1*u.arcsec)
                    starmask = ticid == obstbl['target_name']
                    obstbl = obstbl[starmask]
                except ConnectionError:
                    continue
                except exceptions.ResolverError:
                    n_lc[i] = 0
                    break
            if len(obstbl) > 0:
                #print('Downloading products.')
                products = Observations.get_product_list(obstbl)
                Observations.download_products(products,download_dir='/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_'+lc_source)
            n_lc[i] = len(obstbl)
            break

    return

def pypdf_merger_hj(cluster,lc_source,suffix='dvr',num=150,selected=True):
    if selected:
        files = []
        ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'_selected.csv')
        #ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')

        #for x in ptbl['source_id']:
        #print('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf')
            #files += [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+suffix+'.pdf') for i in range(len(ptbl))]
        files = [glob.glob('/Users/bhealy/Documents/PhD_Thesis/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf') for x in ptbl['source_id']]

    else:
        files = glob.glob('/Users/bhealy/Documents/PhD_Thesis/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+suffix+'.pdf')
    files = np.array(files)

    index = np.zeros(len(files),dtype=int)
    sid = np.copy(index)
    for i in range(len(files)):
        #print(files[i])
        index[i] = np.int(files[i][0].split('/')[-1].split('_')[0])
        sid[i] = files[i][0].split('/')[-1].split('_')[1]

    Tbl = Table(data=[files,index],names=['files','index'])
    Tbl.sort(keys='index')
    sortfiles = Tbl['files']
    #print(sortfiles[0])
    #NEW CODE
    merger = PdfFileMerger()
    fcount = 0
    indx = 0
    bins = np.int(np.ceil(len(sortfiles)/num))
    for i in range(len(sortfiles)):
            merger.append(PdfFileReader(sortfiles[i][0],'rb'))
            fcount += 1
            if (fcount == num) or ((len(sortfiles)-i) == 1):
                #merger.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_CDIPS_merged_DVRs_'+np.str(indx)+'_neighbors.pdf')

                #merger.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_PATHOS_merged_DVRs_'+np.str(indx)+'.pdf')
                if selected:
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_selected_'+suffix+'.pdf')
                else:
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_'+suffix+'.pdf')
                indx += 1
                fcount = 0
                merger = PdfFileMerger()

    return indx

def pypdf_merger(cluster,lc_source,suffix='dvr',num=150,selected=True):
    if selected:
        files = []
        ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'_selected.csv')
        #ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')

        #for x in ptbl['source_id']:
        #print('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf')
            #files += [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+suffix+'.pdf') for i in range(len(ptbl))]
        files = [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf') for x in ptbl['source_id']]

    else:
        files = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+suffix+'.pdf')
    files = np.array(files)

    index = np.zeros(len(files),dtype=int)
    sid = np.copy(index)
    for i in range(len(files)):
        #print(files[i])
        index[i] = np.int(files[i][0].split('/')[-1].split('_')[0])
        sid[i] = files[i][0].split('/')[-1].split('_')[1]

    Tbl = Table(data=[files,index],names=['files','index'])
    Tbl.sort(keys='index')
    sortfiles = Tbl['files']
    #print(sortfiles[0])
    #NEW CODE
    merger = PdfFileMerger()
    fcount = 0
    indx = 0
    bins = np.int(np.ceil(len(sortfiles)/num))
    for i in range(len(sortfiles)):
            merger.append(PdfFileReader(sortfiles[i][0],'rb'))
            fcount += 1
            if (fcount == num) or ((len(sortfiles)-i) == 1):
                #merger.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_CDIPS_merged_DVRs_'+np.str(indx)+'_neighbors.pdf')

                #merger.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_PATHOS_merged_DVRs_'+np.str(indx)+'.pdf')
                if selected:
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_selected_'+suffix+'.pdf')
                else:
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_'+suffix+'.pdf')
                indx += 1
                fcount = 0
                merger = PdfFileMerger()

    return indx


def pypdf_merger_new(cluster,lc_source,targets=[],suffix='dvr',num=150,selected=True):
    if (selected) & (len(targets) == 0):
        files = []
        ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'_selected.csv')
        #ptbl = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')

        #for x in ptbl['source_id']:
        #print('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf')
            #files += [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+suffix+'.pdf') for i in range(len(ptbl))]
        files = [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf') for x in ptbl['source_id']]

    elif (not selected) & (len(targets == 0)):
        files = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+suffix+'.pdf')
    elif len(targets) > 0:
        files = [glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/*'+np.str(x)+'_'+suffix+'.pdf') for x in targets['source_id']]


    files = np.array(files)

    index = np.zeros(len(files),dtype=int)
    sid = np.copy(index)
    for i in range(len(files)):
        #print(files[i])
        index[i] = np.int(files[i][0].split('/')[-1].split('_')[0])
        sid[i] = files[i][0].split('/')[-1].split('_')[1]

    Tbl = Table(data=[files,index],names=['files','index'])
    Tbl.sort(keys='index')
    sortfiles = Tbl['files']
    #print(sortfiles[0])
    #NEW CODE
    merger = PdfFileMerger()
    fcount = 0
    indx = 0
    bins = np.int(np.ceil(len(sortfiles)/num))
    for i in range(len(sortfiles)):
            merger.append(PdfFileReader(sortfiles[i][0],'rb'))
            fcount += 1
            if (fcount == num) or ((len(sortfiles)-i) == 1):

                if (selected) & (len(targets) == 0):
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_selected_'+suffix+'.pdf')
                elif (not selected) & (len(targets) == 0):
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_'+suffix+'.pdf')
                else:
                    merger.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+cluster+'_'+lc_source+'_merged_DVRs_'+np.str(indx)+'_'+suffix+'_finalVerify.pdf')
                indx += 1
                fcount = 0
                merger = PdfFileMerger()

    return indx

def identify_neighbors(mems,cluster,npix=2):
    plt.rcParams['font.size']=12

    #ngc2516mems = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_allCGmems.dat')
    #mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/singlestars_68conf_m48.csv')
    #mems = mems[(mems['proba'] > 0.68) & (~np.isnan(mems['bp_rp']))]
    gaia_ids = mems.index
    #start_index = mems.index[0]
    closeids = np.array([],dtype='int64')
    closemags = np.array([])
    closeseps = np.array([])
    closebprp = np.array([])

        #neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gaia_ids[i])+'.csv') #'*.csv')
        #if len(neighborfiles) != 0:

    #for i in range(len(mems)):
    for gid in gaia_ids:
        #i += start_index
        #print(i)
        #neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gaia_ids[i])+'.csv') #'*.csv')
        neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv') #'*.csv')

        if len(neighborfiles) != 0:
            #print('Found star.',gid)
        #for i in range(496,len(SC)):
        #for i in range(93,94):

            #####
            df = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv')
            closeids = np.append(closeids,df['closeids'].values.tolist())
            closemags = np.append(closemags,df['closemags'].values.tolist()) #[df['closemags'].values.tolist()]
            closeseps = np.append(closeseps,df['closeseps'].values.tolist())
            closebprp = np.append(closebprp,df['closebprp'].values.tolist())

        #return closeids, closemags, closeseps, closebprp

        else:
            closeids = np.array([],dtype='int64')
            closemags = np.array([])
            closeseps = np.array([])
            closebprp = np.array([])

            gmags = mems['phot_g_mean_mag'].values
            bprp = mems['bp_rp'].values
            ra = mems.loc[gid,'ra']*u.deg
            dec = mems.loc[gid,'dec']*u.deg
            SC = SkyCoord(ra,dec)

            #close = np.zeros(len(SC),dtype=bool)
            count = [] #np.zeros(len(SC))
            #inds = np.zeros(len(SC),dtype=int)

    #closeinds = []s

        #####

    #'''
            for attempt in range(10):
                try:
                    qry = Gaia.query_object_async(SC,radius=21*npix*u.arcsec)
                    break
                except:
                    print('Continuing.')
                    continue
            qry['dist'] = qry['dist'] / (21./3600)
            nearby_bright = qry[qry['phot_g_mean_mag'] <= 17][1:]
            #sep = SC[i].separation(SC).arcsec
            #indx = sep != 0
            #close = (sep <= 21*2) & (sep != 0)
            count = len(nearby_bright)
            #print(count[i],'nearby')
            if count != 0:
                #inds[i] = ngc2516mems['source_id'][close].data
                #closeids += nearby_bright['source_id'].data.tolist() #[ngc2516mems['source_id'][close].data]
                closeids = np.append(closeids, nearby_bright['source_id'].data.tolist()) #[ngc2516mems['source_id'][close].data]

                cm1=np.array([])
                cbprp1=np.array([])
                for j in range(len(nearby_bright)):
                    cm1 = np.append(cm1, np.float(format(nearby_bright['phot_g_mean_mag'].data[j],'.1f')))
                    if nearby_bright['bp_rp'].data[j] != '--':
                        cbprp1 = np.append(cbprp1, np.float(format(nearby_bright['bp_rp'].data[j],'.2f')))
                    else:
                        cbprp1 = np.append(cbprp1, -99.)
                #closemags += [np.round(nearby_bright['phot_g_mean_mag'].data,0).tolist()]
                closemags = np.append(closemags, cm1)
                closebprp = np.append(closebprp, cbprp1)
                closeseps = np.append(closeseps, np.round(nearby_bright['dist'].data,1).tolist())

                #closeinds += [i]
                #print(i,ngc2516mems['source_id'][close].data)
            else:
                closeids = np.append(closeids, -1)
                closemags = np.append(closemags, -1)
                closeseps = np.append(closeseps, -1)
                closebprp = np.append(closebprp, -99)
            #print(i)
            #print(closemags)
            #print(closemags[i])
        #####
            #df = pd.DataFrame(data={'closeids':closeids[i],'closemags':closemags[i],'closeseps':closeseps[i]})
            df = pd.DataFrame(data={'closeids':closeids,'closemags':closemags,'closeseps':closeseps,'closebprp':closebprp})
            df.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv',index=False)

    return closeids, closemags, closeseps, closebprp
        #####
    #print(closeids, closemags, closeseps)
    #'''


def identify_neighbors(mems,cluster=None,directory=None,npix=2):
    plt.rcParams['font.size']=12

    #ngc2516mems = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_allCGmems.dat')
    #mems = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/singlestars_68conf_m48.csv')
    #mems = mems[(mems['proba'] > 0.68) & (~np.isnan(mems['bp_rp']))]
    gaia_ids = mems.index
    #start_index = mems.index[0]
    closeids = np.array([],dtype='int64')
    closemags = np.array([])
    closeseps = np.array([])
    closebprp = np.array([])

        #neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gaia_ids[i])+'.csv') #'*.csv')
        #if len(neighborfiles) != 0:

    #for i in range(len(mems)):
    for gid in gaia_ids:
        #print(cluster)
        #i += start_index
        #print(i)
        #neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gaia_ids[i])+'.csv') #'*.csv')
        if cluster==None:
            neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/close_stars/'+np.str(gid)+'.csv') #'*.csv')

        else:
            neighborfiles = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv') #'*.csv')

        if len(neighborfiles) != 0:
            #print('Found star.',gid)
        #for i in range(496,len(SC)):
        #for i in range(93,94):

            #####
            if cluster==None:
                df = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/close_stars/'+np.str(gid)+'.csv')
            else:
                df = pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv')
            closeids = np.append(closeids,df['closeids'].values.tolist())
            closemags = np.append(closemags,df['closemags'].values.tolist()) #[df['closemags'].values.tolist()]
            closeseps = np.append(closeseps,df['closeseps'].values.tolist())
            closebprp = np.append(closebprp,df['closebprp'].values.tolist())

        #return closeids, closemags, closeseps, closebprp

        else:
            closeids = np.array([],dtype='int64')
            closemags = np.array([])
            closeseps = np.array([])
            closebprp = np.array([])

            gmags = mems['phot_g_mean_mag'].values
            bprp = mems['bp_rp'].values
            ra = mems.loc[gid,'ra']*u.deg
            dec = mems.loc[gid,'dec']*u.deg
            SC = SkyCoord(ra,dec)

            #close = np.zeros(len(SC),dtype=bool)
            count = [] #np.zeros(len(SC))
            #inds = np.zeros(len(SC),dtype=int)

    #closeinds = []s

        #####

    #'''
            qry=[]
            for attempt in range(10):
                try:
                    qry = Gaia.query_object_async(SC,radius=21*npix*u.arcsec)
                    break
                except:
                    print('Continuing.')
                    continue
            if len(qry) != 0:
                qry['dist'] = qry['dist'] / (21./3600)
                nearby_bright = qry[qry['phot_g_mean_mag'] <= 17][1:]
            #sep = SC[i].separation(SC).arcsec
            #indx = sep != 0
            #close = (sep <= 21*2) & (sep != 0)
                count = len(nearby_bright)
            else:
                count=0

            #print(count[i],'nearby')
            if count != 0:
                #inds[i] = ngc2516mems['source_id'][close].data
                #closeids += nearby_bright['source_id'].data.tolist() #[ngc2516mems['source_id'][close].data]
                closeids = np.append(closeids, nearby_bright['source_id'].data.tolist()) #[ngc2516mems['source_id'][close].data]

                cm1=np.array([])
                cbprp1=np.array([])
                for j in range(len(nearby_bright)):
                    cm1 = np.append(cm1, np.float(format(nearby_bright['phot_g_mean_mag'].data[j],'.1f')))
                    if nearby_bright['bp_rp'].data[j] != '--':
                        cbprp1 = np.append(cbprp1, np.float(format(nearby_bright['bp_rp'].data[j],'.2f')))
                    else:
                        cbprp1 = np.append(cbprp1, -99.)
                #closemags += [np.round(nearby_bright['phot_g_mean_mag'].data,0).tolist()]
                closemags = np.append(closemags, cm1)
                closebprp = np.append(closebprp, cbprp1)
                closeseps = np.append(closeseps, np.round(nearby_bright['dist'].data,1).tolist())

                #closeinds += [i]
                #print(i,ngc2516mems['source_id'][close].data)
            else:
                closeids = np.append(closeids, -1)
                closemags = np.append(closemags, -1)
                closeseps = np.append(closeseps, -1)
                closebprp = np.append(closebprp, -99)
            #print(i)
            #print(closemags)
            #print(closemags[i])
        #####
            #df = pd.DataFrame(data={'closeids':closeids[i],'closemags':closemags[i],'closeseps':closeseps[i]})
            df = pd.DataFrame(data={'closeids':closeids,'closemags':closemags,'closeseps':closeseps,'closebprp':closebprp})

            if cluster==None:
                df.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/close_stars/'+np.str(gid)+'.csv',index=False)
            else:
                df.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/close_stars/'+np.str(gid)+'.csv',index=False)

    return closeids, closemags, closeseps, closebprp


def get_toi_lc(n,mems,size=(17,17),sigma_thresh=5,tpfplot=False,do_psf=False):

    #srch = lk.search_tesscut('TIC '+np.str(ticids[n]))

    #tpfcoll = srch.download_all(cutout_size=size)
    #apers = np.zeros((len(tpfcoll),size[0],size[1]))
    #for i in range(len(tpfcoll)):
    #    apers[i,:,:] = tpfcoll[i].create_threshold_mask(sigma_thresh)
    #    if tpfplot:
    #        tpfcoll[i].plot(aperture_mask=apers[i])
    gaia_ids = mems['source_id']
    #S = eleanor.multi_sectors('all',gaia = bsourceids[n], tc=True)
    #S = eleanor.multi_sectors('all',gaia = 5290720867621791872, tc=True)
    for attempt in range(10):
        try:
            S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False, post_dir = '/Users/bhealy/Downloads/eleanor_files')
            break
        except:
            print('Continuing.')
            continue
    #S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)

    t = np.array([])
    f = np.array([])
    e = np.array([])

    pca_t = np.array([])
    pca_f = np.array([])
    pca_e = np.array([])

    psf_t = np.array([])
    psf_f = np.array([])
    psf_e = np.array([])

    raw_t = np.array([])
    raw_f = np.array([])
    raw_e = np.array([])
    for i in range(len(S)):
        #if S[i].position_on_chip[0] > 0 and S[i].position_on_chip[1] > 0:
        try:
            if do_psf==False:
                tempdata = eleanor.TargetData(S[i], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True)
            elif do_psf:
                tempdata = eleanor.TargetData(S[i], height=size[0], width=size[1], bkg_size=31, do_psf=True, do_pca=False)
            TPF = TessTargetPixelFile(S[i].cutout)
            #apers = TPF.create_threshold_mask(5)
            apers = np.zeros((31,31))
            apers[9:22,9:22]=tempdata.aperture

            #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
            #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

            #tempdata.get_lightcurve(apers)

            q0 = tempdata.quality == 0
            #t = np.append(t,tempdata.time[q0])
            #f = np.append(f,tempdata.corr_flux[q0]/np.nanmedian(tempdata.corr_flux[q0]))
            #e = np.append(e,tempdata.flux_err[q0]/tempdata.corr_flux[q0])
            pca_t = np.append(pca_t,tempdata.time[q0])
            if do_psf==False:
                pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
                pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])
            elif do_psf:
                pca_f = np.append(pca_f,tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]))
                pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.psf_flux[q0])

            #psf_t = np.append(psf_t,tempdata.time[q0])
            #psf_f = np.append(psf_f,tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]))
            #psf_e = np.append(psf_e,tempdata.flux_err[q0]/tempdata.psf_flux[q0])
            #raw_t = np.append(raw_t,tempdata.time[q0])
            #raw_f = np.append(raw_f,tempdata.raw_flux[q0]/np.nanmedian(tempdata.raw_flux[q0]))


        #else:
        except ValueError:
            print('Skipping sector - not observed here.')
        #except urllib.error.HTTPError:
        #    print('Skipping sector - timeout')

    #lc=lk.LightCurve(time=t,flux=f,flux_err=e)
    lc_pca=lk.LightCurve(time=pca_t,flux=pca_f,flux_err=pca_e)
    #lc_psf=lk.LightCurve(time=psf_t,flux=psf_f,flux_err=psf_e)

    #lc_raw=lk.LightCurve(time=raw_t,flux=raw_f)

    #tpf_p=TPF.plot(aperture_mask=apers)
    #tpf_p.figure.savefig('/Users/bhealy/Documents/PhD_Thesis/TOI_Figs/TPFs/'+np.str(n)+'_'+np.str(Xmatch[n]['TOI'])+'_tpf.pdf')

    #return lc, TPF, apers
    return lc_pca, TPF, apers

def get_acf_period(lc,smth=100,min_prominence=0.1,min_horiz_dist=10,use_first_peak=False, use_second_peak=False):
    #t = data['time']
    #y = (data['flux'] - 1) * 1e3
    #yerr = (data['flux_err']) * 1e3

    t = lc.time.value
    y = (lc.flux.value-1) * 1e3
    #print(y)
    yerr = lc.flux_err.value * 1e3

    delta_t = np.median(np.diff(t))
    new_t = np.arange(t.min(), t.max(), delta_t)
    y_interp = np.interp(new_t, t, y)
    emp_acorr = emcee.autocorr.function_1d(y_interp) #* np.var(y_interp)

    #emp_acorr = emcee.autocorr.function_1d(y) * np.var(y)

    new_f = np.zeros(len(new_t))

    for i in range(len(lc.time)):
        t_diffs = np.abs(new_t - t[i])
        minindx = np.argmin(t_diffs)
        new_f[minindx] = y[i]

    #emp_acorr = emcee.autocorr.function_1d(y_interp)
    emp_acorr = emcee.autocorr.function_1d(new_f)

    #tau = np.arange(len(emp_acorr)) * delta_t

    #print(smth)

    emp_acorr_smooth = gaussian_filter1d(emp_acorr,smth*delta_t)
    #emp_acorr_smooth = gaussian_filter1d(emp_acorr,150*delta_t)

    #print(emp_acorr_smooth)
    #peakinds = find_peaks(emp_acorr_smooth)[0]
    pks = find_peaks(emp_acorr_smooth, prominence=min_prominence, distance=min_horiz_dist)
    peakinds = pks[0]
    peakvals = emp_acorr_smooth[peakinds]

    prominences=pks[1]['prominences']
    peakheights = prominences

    valinds_left = pks[1]['left_bases']
    valinds_right = pks[1]['right_bases']

    valvals_left = emp_acorr_smooth[valinds_left]
    valvals_right = emp_acorr_smooth[valinds_right]

    valinds = [valinds_left, valinds_right]
    #print(valinds)
    #print(peakinds)
    #valinds = argrelmin(emp_acorr_smooth)[0]
    #valinds = find_peaks(-1*emp_acorr_smooth, prominence=min_prominence, distance=min_horiz_dist)[0]
    #print(valinds)

    tau = np.arange(len(emp_acorr_smooth)) * delta_t

    #emp_acorr_smooth[peakinds[0]] - emp_acorr_smooth[peakinds[1]] > 0
    #if len(peakinds) == 0:
    #    peakheights = []
    #else:
    #    peakheights = np.zeros(len(peakinds)-1)
    #    for i in range(len(peakinds)-1):
    #        peakheights[i] = np.mean((emp_acorr_smooth[peakinds[i]]-emp_acorr_smooth[valinds[i]], emp_acorr_smooth[peakinds[i]]-emp_acorr_smooth[valinds[i+1]]))

    #print('Peak heights', peakheights)
    #print('Peak indices', peakinds)

    if len(peakheights) != 0:

        maxindx = np.argmax(peakheights)

        if maxindx == 1:
            maxindx = 1
            #print('!')

        elif maxindx != 1:
            maxindx = 0

        if use_first_peak:
            maxindx = 0

        if use_second_peak:
            maxindx=1

        if use_first_peak and use_second_peak:
            maxindx=0

        period=tau[peakinds[maxindx]]
        #print(period)
        maxheight=peakheights[maxindx]
        maxpeakval=peakvals[maxindx]

        integerinds = np.array([2,3,4,5],dtype=int)
        integermults = [period * n for n in range(2,6)]

        tau_prev = period
        periodmults = np.array([])
        linfit_peakinds = np.array([],dtype=int)
        peakdiffs = np.zeros(len(peakinds)-1)
        n = 1
        #for i in range(len(integermults)):
        #    for j in range(len(peakinds)):
        #        if (np.abs(integermults[i] - tau[peakinds[j]])/period <= 0.1) & ((tau[peakinds[j]]-tau_prev) > 0.5*period):
        #            #print(i,j)
        #            periodmults = np.concatenate((periodmults,[tau[peakinds[j]]]))
        #            tau_prev = tau[peakinds[j]]
        #            linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(j)]))
        #            n += 1
        #print(peakdiffs)
        if len(peakdiffs) > 0:
            for i in range(len(integermults)):
                for j in range(len(peakinds)-1):
                #for j in range(len(peakinds)):
                    ###peakdiffs[j] = np.abs(integermults[i] - tau[peakinds[j]])/period
                    peakdiffs[j] = np.abs(integermults[i] - tau[peakinds[j+1]])/period
                    #print(tau[peakinds[j+1]])
                    #print(j,integermults[i] - tau[peakinds[j+1]])
                #print(peakdiffs)
                if np.min(peakdiffs) <= 0.2: # change to 0.2?
                    #print('!')
                    jj = np.argmin(peakdiffs)
                    #if peakheights[jj]>= 0.5 * maxheight #0.1:
                    if (peakheights[jj]>= 0.5 * maxheight): #| (peakheights[jj] >= min_peakheight):
                        #print('!!')
                        #print(tau[peakinds[jj]])
                        ###periodmults = np.concatenate((periodmults,[tau[peakinds[jj+1]]]))
                        periodmults = np.concatenate((periodmults,[tau[peakinds[jj+1]]]))
                    #linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(jj)]))
                        linfit_peakinds = np.concatenate((linfit_peakinds,[integerinds[i]]))
                        n += 1
                #if (np.abs(integermults[i] - tau[peakinds[j]])/period <= 0.1) & ((tau[peakinds[j]]-tau_prev) > 0.5*period):
                    #print(i,j)
                    #periodmults = np.concatenate((periodmults,[tau[peakinds[j]]]))
                    #tau_prev = tau[peakinds[j]]
                    #linfit_peakinds = np.concatenate((linfit_peakinds,[np.int(j)]))
        #print('integer mults', integermults)
        #print('tau peakinds', tau[peakinds])
        #print(period)
        #print(peakinds)
        #print(peakdiffs)
        #print(periodmults)
        #print(linfit_peakinds)
        sigma_from_fwhm = peak_widths(emp_acorr_smooth, [peakinds[maxindx]])[0][0] * delta_t / 2

        #if (minval1 == minval2) | (minval1 == minval2+1) | (minval1 == minval2-1):
                #print('!!!')
                #sigma_from_fwhm = (np.abs(period - tauslice[minval1]) * 2)/2 #2.35482004503
                #print(period)
                ##print(tauslice[minval1])
                #print(sigma_from_fwhm)

        #print(sigma_from_fwhm)

        linfit_peakinds = np.concatenate(([maxindx],linfit_peakinds))
        #linfit_peakinds += 1
        linfit_peakinds = np.concatenate(([0],linfit_peakinds))
        #if linfit_peakinds[1] == 2:
        #    linfit_peakinds = linfit_peakinds / 2
        #print(linfit_peakinds)
        if linfit_peakinds[1] == 0:
            linfit_peakinds[1] = 1

        e_period_hwhm = sigma_from_fwhm
        #print(np.str(n))
        #print(linfit_peakinds)
        if (n == 1) | (np.sum(np.diff(np.diff(linfit_peakinds)))) != 0:
        #if (n == 1) | (np.sum(np.diff(linfit_peakinds))) != 0:

            #print('X')
            periodmults = np.array([0,period])
            period_unc = e_period_hwhm #sigma_from_fwhm
            finalperiod = period
            e_period_mad = -1
            e_period_std = -1

        elif n > 1:
            #print('!!!')
            periodmults = np.concatenate(([0,period],periodmults))
            period_diffs = np.diff(periodmults)
            #print(period_diffs)
            #print(period_diffs)
            #finalperiod = np.median(period_diffs)
            finalperiod = np.polyfit(linfit_peakinds,periodmults,1)[0]
            #print(linfit_peakinds)
            #print(periodmults)


            #print(median_absolute_deviation(period_diffs))
            #print(np.std(period_diffs))
            #print(n)
            e_period_mad = 1.483* median_absolute_deviation(period_diffs) / np.sqrt(n-1)
            e_period_std = np.std(period_diffs)

            #print(median_absolute_deviation(period_diffs))
            #period_unc = 1.483* median_absolute_deviation(period_diffs) / np.sqrt(n-1)

            ###period_unc = e_period_mad
            period_unc = e_period_std

            if period_unc == 0:
                #print('Period unc too small')
                period_unc = e_period_hwhm #sigma_from_fwhm #np.std(period_diffs) / np.sqrt(n-1)
    else:
        #print('!')
        finalperiod = -1
        period_unc = -1
        e_period_hwhm = -1
        e_period_mad = -1
        e_period_std = -1
        #peakinds = [-1]
        #valinds = [[-1],[-1]]
        maxheight = -1
        periodmults = []


    return finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults

def get_lc_and_period(id,cluster=None,lc_source='PATHOS',use_eleanor=True,smth=100,min_prominence=0.1,min_horiz_dist=10,size=(31,31),sigma_thresh=5,tpfplot=False,sec='all',disp_sec=None,do_psf=False):
    #size=(31,31)
    #srch = lk.search_tesscut('TIC '+np.str(ticids[n]))
    if cluster==None:
        AM = 'normal'
    else:
        AM = 'small'
    #print('len mems', len(mems))
    #tpfcoll = srch.download_all(cutout_size=size)
    #apers = np.zeros((len(tpfcoll),size[0],size[1]))
    #for i in range(len(tpfcoll)):
    #    apers[i,:,:] = tpfcoll[i].create_threshold_mask(sigma_thresh)
    #    if tpfplot:
    #        tpfcoll[i].plot(aperture_mask=apers[i])

    #if isinstance(mems, np.int):
    #    gaia_ids = [mems]
        #print('!')
    #else:
    #    gaia_ids = mems['source_id']
        #print('x')

    #print(n)
    #print(gaia_ids)
    #####obstbl = Observations.query_criteria(objectname='Gaia DR2 '+np.str(gaia_ids[i]),provenance_name='CDIPS',radius=1*u.arcsec)
    for attempt in range(10):
        try:
            ticid, tmag = Catalogs.query_object(objectname='Gaia DR2 '+np.str(id),catalog='TIC',radius=1*u.arcsec)[0]['ID','Tmag']
        #lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_PATHOS/mastDownload/HLSP/*'+ticid+'*')
        #print(np.str(gaia_ids[n]))
        #print(ticid, cluster, lc_source)
            if lc_source == 'PATHOS':
                lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_'+lc_source+'/mastDownload/HLSP/*'+ticid+'*')
            elif lc_source == 'CDIPS':
                lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_'+lc_source+'/mastDownload/HLSP/*'+np.str(id)+'*')
            else:
                lcdirs = []
            break
        except exceptions.ResolverError:
            lcdirs = []
            break
        except ConnectionError:
            print('Continuing.')
            continue
        except ConnectionResetError:
            print('ConnectionResetError - Continuing.')
            continue

    #lcdirs = glob.glob('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/PATHOS/mastDownload/HLSP/*'+np.str(gaia_ids[n])+'*')

    pca_t = np.array([])
    pca_f = np.array([])
    pca_e = np.array([])
    secs = []
    hlsp_secs = []

    all_maxheights = [] #np.zeros(len(secs))
    all_smths = [] #np.copy(all_maxheights)
    all_periods = [] #np.copy(all_maxheights)
    all_period_uncs = [] #np.copy(all_maxheights)
    all_e_period_hwhm = [] #np.copy(all_maxheights)
    all_e_period_mad = [] #np.copy(all_maxheights)

    if sec == 0:
        sec = 'all'

    #print(sec)
    #print(gaia_ids)
    #print(gaia_ids[n])
    #print(id)
    for attempt in range(10):
        try:
            #S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)
            S = eleanor.multi_sectors(sec,gaia=id, tc=False, post_dir = '/Users/bhealy/Downloads/eleanor_files')
            break
        except SearchError:
            print('Not observed yet.')
            S = []
            pos=[-1,-1]
            label='x'
            bestsector=-1
            break
        except ConnectionError:
            print('ConnectionError - Continuing.')
            continue
        except ConnectionResetError:
            print('ConnectionResetError - Continuing.')
            continue
        else:
            print('Continuing.')
            continue
    try:
        all_secs = np.array([S[i].sector for i in range(len(S))])
        print(len(lcdirs), len(all_secs))
        #print(all_secs)
        #not_covered = ~np.isin(E_secs, secs)
        #S_el = S[not_covered]

        if (len(lcdirs) != 0) & (lc_source=='CDIPS'): #PROCEED WITH CDIPS
            print('CDIPS')
            for d in lcdirs:
                secs += [np.int(d.split('-')[1][2:])]

            secs = np.array(secs)
            hlsp_secs=secs
            #print(secs)

            lctbl = Table(data=[lcdirs,secs],names=['lcdirs','secs'])

            ###lctbl = lctbl[(lctbl['secs'] != 4) & (lctbl['secs'] != 1)]
            #Possibly exclude sectors 1 and 4?

            lctbl.sort('secs')
            secs = lctbl['secs'].data
            #print(secs)

            for i in range(len(lctbl)):
                lcfile = glob.glob(lctbl['lcdirs'][i]+'/*.fits')
                hdul=fits.open(lcfile[0])

                cdips_time = hdul[1].data['TMID_BJD'] - 2457000

                cdips_flux = 2.512**(-hdul[1].data['PCA1'])
                cdips_flux /= np.nanmedian(cdips_flux)

                cdips_iflflux = hdul[1].data['IFL1']
                #cdips_flux = cdips_iflflux

                cdips_err = hdul[1].data['IFE1']
                cdips_fracerr = cdips_err/cdips_iflflux

                cdips_err = cdips_fracerr * cdips_flux
                cdips_qual = hdul[1].data['IRQ1']
                goodqual = cdips_qual != 'X'

                single_lc_pca=lk.LightCurve(time=cdips_time[goodqual],flux=cdips_flux[goodqual],flux_err=cdips_err[goodqual])
                single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

                #pca_t = np.append(pca_t,cdips_time[goodqual])
                #pca_f = np.append(pca_f,cdips_flux[goodqual])
                #pca_e = np.append(pca_e,cdips_err[goodqual])

                pca_t = np.append(pca_t,single_lc_pca.time)
                pca_f = np.append(pca_f,single_lc_pca.flux)
                pca_e = np.append(pca_e,single_lc_pca.flux_err)

                if len(single_lc_pca) == 0:
                    all_maxheights += [-1]
                    all_periods += [-1]
                    all_period_uncs += [-1]
                    all_e_period_hwhm += [-1]
                    all_e_period_mad += [-1]
                else:
                    maxpow_period = single_lc_pca.to_periodogram().period_at_max_power.value
                    #smth = 500
                    #if (maxpow_period < 1) & (maxpow_period >= 0.1):
                #        smth = 150
                #    elif maxpow_period > 3:
                #        smth = 800
                #    elif maxpow_period > 6:
                #        smth = 1200

                    all_smths += [smth]
                    finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)
                    all_maxheights += [maxheight]
                    all_periods += [finalperiod]
                    all_period_uncs += [period_unc]
                    all_e_period_hwhm += [e_period_hwhm]
                    all_e_period_mad += [e_period_mad]

                #calculate peakheights etc. here

        elif (len(lcdirs) != 0) & (lc_source=='PATHOS'): #PROCEED WITH PATHOS
            print('PATHOS')
            #print(lcdirs)
            for d in lcdirs:
                #secs += [np.int(d.split('-')[1][2:])]
                #secs += [np.int(d.split('-')[2].split('_')[0][-2:])]
                secs += [np.int(d.split('-')[2].split('_')[0][1:])]

            secs = np.array(secs)
            hlsp_secs = secs

            lctbl = Table(data=[lcdirs,secs],names=['lcdirs','secs'])

            ###lctbl = lctbl[(lctbl['secs'] != 4) & (lctbl['secs'] != 1)]
            #Possibly exclude sectors 1 and 4?

            lctbl.sort('secs')
            secs = lctbl['secs'].data
            #print(secs)

            #CHANGE KEYWORDS LATER
            for i in range(len(lctbl)):
                #lcfile = glob.glob(lctbl['lcdirs'][i]+'/*.fits')
                lcfile = glob.glob(lctbl['lcdirs'][i]+'/*.txt')

                pathos_data = ascii.read(lcfile[0],format='commented_header',header_start=-1)

                #hdul=fits.open(lcfile[0])

                #cdips_time = hdul[1].data['TMID_BJD'] - 2457000
                pathos_time = pathos_data['TIME[d]']

                #print(tmag)
                if tmag <= 7:
                    flux_kywrd = 'AP4_FLUX_COR[e-/s]'
                elif (tmag > 7) & (tmag <= 9):
                    flux_kywrd = 'AP3_FLUX_COR[e-/s]'
                elif (tmag > 9) & (tmag <= 10.5):
                    flux_kywrd = 'AP2_FLUX_COR[e-/s]'
                elif (tmag > 10.5) & (tmag <= 13.5):
                    flux_kywrd = 'PSF_FLUX_COR[e-/s]'
                elif tmag > 13.5:
                    flux_kywrd = 'AP1_FLUX_COR[e-/s]'


                #cdips_flux = 2.512**(-hdul[1].data['PCA1'])
                #cdips_flux /= np.nanmedian(cdips_flux)
                pathos_flux = pathos_data[flux_kywrd]
                pathos_err = np.sqrt(pathos_flux)/pathos_flux

                pathos_bkg = pathos_data['SKY_LOCAL[e-/s]']
                pathos_bkglc = lk.LightCurve(pathos_time,pathos_bkg)
                badsky = pathos_bkglc.remove_outliers(5,return_mask=True)[1]

                pathos_flux /= np.nanmedian(pathos_flux)

                #cdips_iflflux = hdul[1].data['IFL1']
                #cdips_flux = cdips_iflflux

                #cdips_err = hdul[1].data['IFE1']
                #cdips_fracerr = cdips_err/cdips_iflflux

                #cdips_err = cdips_fracerr * cdips_flux
                #cdips_qual = hdul[1].data['IRQ1']
                pathos_qual = pathos_data['DQUALITY']

                #goodqual = cdips_qual != 'X'
                goodqual = (pathos_qual == 0) & (~badsky)

                single_lc_pca=lk.LightCurve(time=pathos_time[goodqual],flux=pathos_flux[goodqual],flux_err = pathos_err[goodqual])
                #print(np.min(single_lc_pca.flux),np.max(single_lc_pca.flux))
                if (np.min(single_lc_pca.flux) != np.max(single_lc_pca.flux)) & (np.min(single_lc_pca.flux) != 1.0):
                    single_lc_pca = single_lc_pca.remove_nans().remove_outliers()#.flatten(481)

                #pca_t = np.append(pca_t,pathos_time[goodqual])
                #pca_f = np.append(pca_f,pathos_flux[goodqual])
                #pca_e = np.append(pca_e,pathos_err[goodqual])

                pca_t = np.append(pca_t,single_lc_pca.time)
                pca_f = np.append(pca_f,single_lc_pca.flux)
                pca_e = np.append(pca_e,single_lc_pca.flux_err)

                if len(single_lc_pca) == 0:
                    all_maxheights += [-1]
                    all_periods += [-1]
                    all_period_uncs += [-1]
                    all_e_period_hwhm += [-1]
                    all_e_period_mad += [-1]
                else:
                    maxpow_period = single_lc_pca.to_periodogram().period_at_max_power.value
                    #smth = 500
                    #if (maxpow_period < 1) & (maxpow_period >= 0.1):
                    #    smth = 150
                    #elif maxpow_period > 3:
                #        smth = 800
                #    elif maxpow_period > 6:
                #        smth = 1200

                    all_smths += [smth]
                    finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)
                    all_maxheights += [maxheight]
                    all_periods += [finalperiod]
                    all_period_uncs += [period_unc]
                    all_e_period_hwhm += [e_period_hwhm]
                    all_e_period_mad += [e_period_mad]
        #all_smths[i] = smth

    #print(all_maxheights)
        secs = np.array(secs,dtype=np.int)
        badsecs = np.array([])
        not_covered = ~np.isin(all_secs, secs)
        #print(not_covered)
        S_el = np.array(S)[not_covered]

        #S = eleanor.multi_sectors('all',gaia = bsourceids[n], tc=True)
        #S = eleanor.multi_sectors('all',gaia = 5290720867621791872, tc=True)
        #ELSE.......eleanor
        #else:

        #except requests.exceptions.HTTPError:
            #if sec == 0:
            #S = eleanor.multi_sectors('all',gaia=gaia_ids[n], tc=False)

        #    else:
        #        S = eleanor.Source(gaia=gaia_ids[n],tc=False,sector=sec)

            #5290723204083832448
            #S = eleanor.multi_sectors([2,3,4,5,6,7,8,9,10,11,12,13],gaia=gaia_ids[n], tc=True)
            #if S[0].sector == 1:
            #    S = S[1:]

        #t = np.array([])
        #f = np.array([])
        #e = np.array([])

        #pca_t = np.array([])
        #pca_f = np.array([])
        #pca_e = np.array([])

        #psf_t = np.array([])
        #psf_f = np.array([])
        #psf_e = np.array([])

        #raw_t = np.array([])
        #raw_f = np.array([])
        #raw_e = np.array([])

        #all_maxheights = np.zeros(len(S))
        #all_periods = np.copy(all_maxheights)
        #all_period_uncs = np.copy(all_maxheights)
        #all_e_period_hwhm = np.copy(all_maxheights)
        #all_e_period_mad = np.copy(all_maxheights)
        #all_smths = np.copy(all_maxheights)

            #sectors = np.zeros(len(S)-1,dtype=int)
        if (len(S_el) != 0) & (use_eleanor == True):
            print('eleanor')

            for i in range(len(S_el)):
                #secs += [S_el[i].sector]
                secs = np.append(secs,S_el[i].sector)
                #print(i)
                #if S[i].position_on_chip[0] > 0 and S[i].position_on_chip[1] > 0:
                try:
                    if do_psf==False:
                        tempdata = eleanor.TargetData(S_el[i], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True, aperture_mode=AM)
                    elif do_psf:
                        tempdata = eleanor.TargetData(S_el[i], height=size[0], width=size[1], bkg_size=31, do_psf=True, do_pca=False, aperture_mode=AM)

                    #TPF = TessTargetPixelFile(S[i].cutout)
                    #apers = TPF.create_threshold_mask(5)
                    #apers = np.zeros((31,31))
                    #apers[9:22,9:22]=tempdata.aperture

                    #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
                    #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

                    #tempdata.get_lightcurve(apers)
                    #sectors[i] = S[i].sector
                    q0 = tempdata.quality == 0

                    #pca_t = np.append(pca_t,tempdata.time[q0])
                    #pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
                    #pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])

                    if do_psf==False:
                        single_lc_pca=lk.LightCurve(time=tempdata.time[q0],flux=tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]),flux_err=tempdata.flux_err[q0]/tempdata.pca_flux[q0])
                    elif do_psf:
                        single_lc_pca=lk.LightCurve(time=tempdata.time[q0],flux=tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]),flux_err=tempdata.flux_err[q0]/tempdata.psf_flux[q0])

                    single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

                    pca_t = np.append(pca_t,single_lc_pca.time)
                    pca_f = np.append(pca_f,single_lc_pca.flux)
                    pca_e = np.append(pca_e,single_lc_pca.flux_err)


                    middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                    goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                    single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])
                    maxpow_period = single_lc_pca.to_periodogram().period_at_max_power.value

                    #smth = 500
                    #if (maxpow_period < 1) & (maxpow_period >= 0.1):
                #        smth = 150
                #    elif maxpow_period > 3:
                #        smth = 800
                #    elif maxpow_period > 6:
                #        smth = 1200
                    all_smths += [smth]

                    #for i in range(1,16):
                    #    smth = 100 * i
                    #    finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth)
                    #    if tau[peakinds][2] / tau[peakinds][0] > 2.5:
                    #        break

                    #finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth)
                    finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)

                    all_maxheights += [maxheight]
                    all_periods += [finalperiod]
                    all_period_uncs += [period_unc]
                    all_e_period_hwhm += [e_period_hwhm]
                    all_e_period_mad += [e_period_mad]

                    #print(all_maxheights)
                    #t = np.append(t,tempdata.time[q0])
                    #f = np.append(f,tempdata.corr_flux[q0]/np.nanmedian(tempdata.corr_flux[q0]))
                    #e = np.append(e,tempdata.flux_err[q0]/tempdata.corr_flux[q0])
                    #pca_t = np.append(pca_t,tempdata.time[q0])
                    #pca_f = np.append(pca_f,tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]))
                    #pca_e = np.append(pca_e,tempdata.flux_err[q0]/tempdata.pca_flux[q0])
                    #psf_t = np.append(psf_t,tempdata.time[q0])
                    #psf_f = np.append(psf_f,tempdata.psf_flux[q0]/np.nanmedian(tempdata.psf_flux[q0]))
                    #psf_e = np.append(psf_e,tempdata.flux_err[q0]/tempdata.psf_flux[q0])
                    #raw_t = np.append(raw_t,tempdata.time[q0])
                    #raw_f = np.append(raw_f,tempdata.raw_flux[q0]/np.nanmedian(tempdata.raw_flux[q0]))


                #else:
                except ValueError:
                    print('ValueError - skipping sector.')
                    all_maxheights += [-2]
                    all_smths += [smth]
                    badsecs = np.append(badsecs,S_el[i].sector)
                except IndexError:
                    print('IndexError - skipping sector.')
                    all_maxheights += [-2]
                    all_smths += [smth]
                    badsecs = np.append(badsecs,S_el[i].sector)
                except AttributeError:
                    print('AttributeError - skipping sector.')
                    all_maxheights += [-2]
                    all_smths += [smth]
                    badsecs = np.append(badsecs,S_el[i].sector)
                except HTTPError:
                    print('Skipping sector - HTTPError.')
                    continue

        #print(all_maxheights)

        bestindx = np.argmax(all_maxheights)

        timemasks = {
                's1' : np.where((pca_t >= 1325.293) & (pca_t <= 1353.178))[0],
                's2' : np.where((pca_t >= 1354.101) & (pca_t <= 1381.515))[0],
                's3' : np.where((pca_t >= 1385.897) & (pca_t <= 1406.292))[0],
                's4' : np.where((pca_t >= 1410.900) & (pca_t <= 1436.849))[0],
                's5' : np.where((pca_t >= 1437.826) & (pca_t <= 1464.400))[0],
                's6' : np.where((pca_t >= 1468.270) & (pca_t <= 1490.044))[0],
                's7' : np.where((pca_t >= 1491.626) & (pca_t <= 1516.085))[0],
                's8' : np.where((pca_t >= 1517.342) & (pca_t <= 1542.000))[0],
                's9' : np.where((pca_t >= 1543.216) & (pca_t <= 1568.475))[0],
                's10' : np.where((pca_t >= 1569.432) & (pca_t <= 1595.680))[0],
                's11' : np.where((pca_t >= 1596.772) & (pca_t <= 1623.891))[0],
                's12' : np.where((pca_t >= 1624.950) & (pca_t <= 1652.891))[0],
                's13' : np.where((pca_t >= 1653.915) & (pca_t <= 1682.357))[0],
                's14' : np.where((pca_t >= 1683.34838) & (pca_t <= 1710.20392))[0],
                's15' : np.where((pca_t >= 1711.35947) & (pca_t <= 1737.40946))[0],
                's16' : np.where((pca_t >= 1738.64697) & (pca_t <= 1763.31918))[0],
                's17' : np.where((pca_t >= 1764.67891) & (pca_t <= 1789.69417))[0],
                's18' : np.where((pca_t >= 1790.65111) & (pca_t <= 1815.03026))[0],
                's19' : np.where((pca_t >= 1816.07749) & (pca_t <= 1841.14831))[0],
                's20' : np.where((pca_t >= 1842.49831) & (pca_t <= 1868.82191))[0],
                's21' : np.where((pca_t >= 1870.42885) & (pca_t <= 1897.78023))[0],
                's22' : np.where((pca_t >= 1899.30103) & (pca_t <= 1926.49269))[0],
                's23' : np.where((pca_t >= 1928.09965) & (pca_t <= 1954.87464))[0],
                's24' : np.where((pca_t >= 1955.78990) & (pca_t <= 1982.28017))[0],
                's25' : np.where((pca_t >= 1983.62738) & (pca_t <= 2009.30515))[0],
                's26' : np.where((pca_t >= 2010.26209) & (pca_t <= 2035.13430))[0],
                's27' : np.where((pca_t >= 2036.27320) & (pca_t <= 2060.64125))[0],
                's28' : np.where((pca_t >= 2061.84540) & (pca_t <= 2087.09678))[0],
                's29' : np.where((pca_t >= 2088.23429) & (pca_t <= 2114.43289))[0],
                's30' : np.where((pca_t >= 2115.88011) & (pca_t <= 2143.22177))[0],
                's31' : np.where((pca_t >= 2144.50927) & (pca_t <= 2169.94398))[0],
                's32' : np.where((pca_t >= 2174.21898) & (pca_t <= 2200.23147))[0],
                's33' : np.where((pca_t >= 2201.72730) & (pca_t <= 2227.57173))[0],
                's34' : np.where((pca_t >= 2228.74533) & (pca_t <= 2254.06476))[0],
                's35' : np.where((pca_t >= 2254.98421) & (pca_t <= 2279.97864))[0],
                's36' : np.where((pca_t >= 2280.89808) & (pca_t <= 2305.98835))[0],
                's37' : np.where((pca_t >= 2307.23418) & (pca_t <= 2332.57862))[0],
                's38' : np.where((pca_t >= 2333.84945) & (pca_t <= 2360.55083))[0],
                's39' : np.where((pca_t >= 2361.76612) & (pca_t <= 2389.71750))[0]}

        max_sec = np.int([x for x in timemasks.keys()][-1][1:]) #37

            #print(all_maxheights)
            #print(bestsector)
            #print(bestindx)
            #print(S[bestindx].cutout)

        #if sec == 0:
        #    bestsector = S[bestindx].sector
        #else:
        #    bestsector = sec
        #print(bestindx)
        print(secs)
        if (disp_sec != None) & (disp_sec in secs):
            bestsector = disp_sec
            #key_prev = 's' + np.str(bestsector-1)
            #key_next = 's' + np.str(bestsector+1)
        else:
            bestsector = secs[bestindx]

        key = 's' + np.str(bestsector)
        print(bestsector)
        #print(gaia_ids[n])

        if np.int(bestsector) in hlsp_secs:
            label=1
        else:
            label=0

        #print(secs.dtype)
        #print(bestsector.dtype)
        #print(gaia_ids[n].dtype)
        for attempt in range(10):
            try:
                #qry = Gaia.query_object_async(SC,radius=21*npix*u.arcsec)
                Src = eleanor.Source(gaia=id, sector=np.int(bestsector), post_dir = '/Users/bhealy/Downloads/eleanor_files')
                pos = Src.position_on_chip
                break
            except SearchError:
                print('Not observed yet.')
                Src = []
                pos=[-1,-1]
            except ConnectionResetError:
                print('ConnectionResetError - Continuing.')
                continue
            except ConnectionError:
                print('ConnectionResetError - Continuing.')
                continue
            else:
                print('Continuing.')
                continue
        ###Src = eleanor.Source(gaia=id, sector=np.int(bestsector), post_dir = '/Users/bhealy/Downloads')


        #Src = eleanor.Source(gaia=5290024533163062144, sector=31)
        #print(Src.sector)


        #tempdata = eleanor.TargetData(S[bestindx], height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True, aperture_mode='small')

        tempdata = eleanor.TargetData(Src, height=size[0], width=size[1], bkg_size=31, do_psf=False, do_pca=True, aperture_mode=AM)

            #TPF = TessTargetPixelFile(S[bestindx].cutout)
        TPF = tempdata.tpf[0,:,:]
        apers = tempdata.aperture

        #print('bestsector')
                #apers = TPF.create_threshold_mask(5)
        #apers = np.zeros((31,31))
        #apers[9:22,9:22]=tempdata.aperture
        #print(S[bestindx].gaia)
                #tempdata.save('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')
                #TPF = TessTargetPixelFile('/Users/bhealy/Documents/PhD_Thesis/TPFs/TPF_TIC_'+np.str(ticids[n])+'.fits')

                #tempdata.get_lightcurve(apers)
        #sectors[i] = S[i].sector
        q0 = tempdata.quality == 0

        final_time = np.array([])
        final_flux = np.array([])
        final_flux_err = np.array([])

        prev_sec = bestsector
        for i in range(bestsector):
            test_sec = bestsector - i
            if (test_sec in secs) & (np.abs(test_sec - prev_sec) <= 1) & (test_sec not in badsecs):
                print(test_sec)
                key = 's' + np.str(test_sec)
                single_lc_pca=lk.LightCurve(time=pca_t[timemasks[key]],flux=pca_f[timemasks[key]],flux_err=pca_e[timemasks[key]])
                single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

                middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

                final_time = np.append(final_time, single_lc_pca.time)
                final_flux = np.append(final_flux, single_lc_pca.flux)
                final_flux_err = np.append(final_flux_err, single_lc_pca.flux_err)

                prev_sec = test_sec

        prev_sec = bestsector
        for i in range(max_sec - bestsector):
            test_sec = bestsector + i + 1
            if (test_sec in secs) & (np.abs(test_sec - prev_sec) <= 1) & (test_sec not in badsecs):
                print(test_sec)
                key = 's' + np.str(test_sec)
                single_lc_pca=lk.LightCurve(time=pca_t[timemasks[key]],flux=pca_f[timemasks[key]],flux_err=pca_e[timemasks[key]])
                single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

                middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

                final_time = np.append(final_time, single_lc_pca.time)
                final_flux = np.append(final_flux, single_lc_pca.flux)
                final_flux_err = np.append(final_flux_err, single_lc_pca.flux_err)

                prev_sec = test_sec

        single_lc_pca = lk.LightCurve(time = final_time, flux = final_flux, flux_err = final_flux_err)
        #print(single_lc_pca.time, single_lc_pca.time.shape)
        #if len(pca_t[timemasks[key_prev]]) > 0:
        #    prev_lc_pca = lk.LightCurve(time=pca_t[timemasks[key_prev]],flux=pca_f[timemasks[key_prev]],flux_err=pca_e[timemasks[key_prev]])

        #if len(pca_t[timemasks[key_next]]) > 0:
        #    next_lc_pca = lk.LightCurve(time=pca_t[timemasks[key_next]],flux=pca_f[timemasks[key_next]],flux_err=pca_e[timemasks[key_next]])

    ####################################################################################################

        #single_lc_pca=lk.LightCurve(time=pca_t[q0],flux=pca_f[q0],flux_err=pca_e[q0])

        ###single_lc_pca=lk.LightCurve(time=tempdata.time[q0],flux=tempdata.pca_flux[q0]/np.nanmedian(tempdata.pca_flux[q0]),flux_err=tempdata.flux_err[q0]/tempdata.pca_flux[q0])
        #single_lc_pca = single_lc_pca.remove_nans().remove_outliers()

    ########## previous
        #single_lc_pca=lk.LightCurve(time=pca_t[timemasks[key]],flux=pca_f[timemasks[key]],flux_err=pca_e[timemasks[key]])
        #single_lc_pca = single_lc_pca.remove_nans().remove_outliers()
        #middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
        #goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
        #single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])
    ###########
        #if 50 not in all_smths:
        #    final_smth = 350
        #elif 350 not in all_smths:
        #    final_smth = 50
        #else:
        #    final_smth = 350
        final_smth = all_smths[bestindx]


        t_start = np.min(single_lc_pca.time) - 0.5
        t_end = np.max(single_lc_pca.time) + 0.5
        #finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=final_smth)
        finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, periodmults = get_acf_period(single_lc_pca,smth=final_smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)

        flat_lc_401 = single_lc_pca.flatten(401)
        alt_values = get_acf_period(flat_lc_401,smth=final_smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)
        alt_period = alt_values[0]
        alt_period_unc = alt_values[1]

        flat_lc_601 = single_lc_pca.flatten(601)
        alt_values_b = get_acf_period(flat_lc_601,smth=final_smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist)
        alt_period_b = alt_values_b[0]
        alt_period_unc_b = alt_values_b[1]

        #lc=lk.LightCurve(time=t,flux=f,flux_err=e)
        lc_pca=lk.LightCurve(time=pca_t,flux=pca_f,flux_err=pca_e)
        return lc_pca, TPF, apers, finalperiod, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label, bestsector, alt_period, alt_period_unc, alt_period_b, alt_period_unc_b

    except ValueError:
        print('ValueError: bad LC.')
        return [-1], -1, -1, -1, -1, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, pos, label, bestsector, -1, -1, -1, -1

    #lc_psf=lk.LightCurve(time=psf_t,flux=psf_f,flux_err=psf_e)
    #label=0
    #lc_raw=lk.LightCurve(time=raw_t,flux=raw_f)

    #tpf_p=TPF.plot(aperture_mask=apers)
    #tpf_p.figure.savefig('/Users/bhealy/Documents/PhD_Thesis/TOI_Figs/TPFs/'+np.str(n)+'_'+np.str(Xmatch[n]['TOI'])+'_tpf.pdf')

    #return lc, TPF, apers
    #return lc_pca, TPF, apers, finalperiod, period_unc, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos

#periods = np.zeros(len(toi_unique))
def perform_period_analysis(mems,cluster=None,npix=2,lc_source='PATHOS',use_eleanor=True,generate_neighbor_lcs=True,smth=100,min_prominence=0.1,min_horiz_dist=10,size=(31,31),sigma_thresh=5,tpfplot=False,sec='all',disp_sec=None):

    start_index = mems.index[0]
    n_id = mems[['source_id']]
    n_id['num'] = n_id.index

    mems = mems.set_index('source_id')
    n_id = n_id.set_index('source_id')
    #print(n_id)

    gmags = mems['phot_g_mean_mag'].values

    #perTbl = Table(data=[gaia_ids,periods,period_uncs,e_period_hwhm,e_period_mad,e_period_std,label],names=['source_id','period','period_unc','e_period_hwhm','e_period_mad','e_period_std','PATHOS'])
    #perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat')
    gaia_ids = mems.index

    #UNCOMMENT WHEN CREATED
    filetest = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

    if len(filetest) == 0:
        periods = np.zeros(len(mems))
        period_uncs = np.copy(periods)
        alt_periods = np.zeros(len(mems))
        alt_period_uncs = np.copy(periods)
        alt_periods_b = np.zeros(len(mems))
        alt_period_uncs_b = np.copy(periods)
        e_period_hwhm = np.copy(periods)
        e_period_mad = np.copy(periods)
        e_period_std = np.copy(periods)
        label = np.zeros(len(mems),dtype=int)
        maxheights = np.copy(periods)
        n_periodmults = np.copy(label)
        bestsectors = np.copy(label)
        pdgm_peaks = np.copy(periods)
        xpixel_positions = np.copy(periods)
        ypixel_positions = np.copy(periods)

    #if initialize:
        #perTbl = Table(data=[gaia_ids,periods,period_uncs,e_period_hwhm,e_period_mad,e_period_std,label],names=['source_id','period','period_unc','e_period_hwhm','e_period_mad','e_period_std','PATHOS'])

        #perTbl = pd.DataFrame({'source_id':gaia_ids, 'period':periods, 'period_unc':period_uncs, 'e_period_hwhm':e_period_hwhm, 'e_period_mad':e_period_mad, 'e_period_std':e_period_std, 'HLSP_best':label,
        #                        'maxheight':maxheights, 'n_periodmults': n_periodmults, 'bestsector':bestsectors, 'pdgm_peak':pdgm_peaks})

        perTbl = pd.DataFrame({'source_id':np.array([],dtype='int64'), 'period':np.array([]), 'period_unc':np.array([]), 'alt_period':np.array([]), 'alt_period_unc':np.array([]), 'alt_period_b':np.array([]), 'alt_period_unc_b':np.array([]), 'e_period_hwhm':np.array([]), 'e_period_mad':np.array([]), 'e_period_std':np.array([]), 'HLSP_best':np.array([],dtype=np.int),
                                'maxheight':np.array([]), 'n_periodmults': np.array([],dtype=np.int), 'bestsector':np.array([],dtype=np.int), 'pdgm_peak':np.array([]), 'xpixel_pos':np.array([]), 'ypixel_pos':np.array([])})
        perTbl = perTbl.set_index('source_id')
        perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

    else:
        #perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/M48_ptbl_pathos_neighbors.dat')
        perTbl = pd.read_csv(filetest[0], index_col='source_id')
        #perTbl = perTbl.set_index('source_id')
        #pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

    #import OpenSSL
    #import astroquery.mast.core

    except_count = 0
    total_failures = 0
    #for n in range(371,372):
    #for n in range(496,len(ngc2516mems)):

    #for n in range(len(mems)):
    #    n= n+start_index

    for id in gaia_ids:
    #for n in mems.index:

    #for n in range(93,94):


    #for n in range(256,257):
    #for n in range(58,len(mems)):

        #for attempt in range(10):
        #    try:

    #requests.exceptions.SSLError: HTTPSConnectionPool(host='mast.stsci.edu', port=443): Max retries exceeded with url: /api/v0.1/Download/file?uri=mast:HLSP/eleanor/postcards/s0010/3-3/hlsp_eleanor_tess_ffi_postcard-s0010-3-3-cal-1000-0052_tess_v2_pc.fits (Caused by SSLError(SSLError("bad handshake: SysCallError(-1, 'Unexpected EOF')",),))
    #ssl.SSLError
        #for n in range(38,len(ngc2516mems)):
        #for n in range(28,29):
        #for n in range(0,3):
        #for n in range(165,168):

        gaia_id = id
        num = n_id.loc[gaia_id,'num']
        print('Star', num, gaia_id)
        gmag = np.str(np.round(mems.loc[id,'phot_g_mean_mag'],2))
        bprp = np.str(np.round(mems.loc[id,'bp_rp'],2))
        #print(n, gaia_id)
        #toi = toi_unique[n]
        closeids, closemags, closeseps, closebprp = identify_neighbors(mems.loc[[gaia_id]], cluster, npix)

        #planets = toi_dict[toi_unique[n]]

        #lc, TPF, apers = get_toi_lc(n,ticids)
        #lc, TPF, apers = get_toi_lc(n,ngc2516mems)

        #lc_pca, TPF, apers, periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos = get_lc_and_period(n,ngc2516mems)
        lc_pca, TPF, apers, period, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label, bestsec, alt_period, alt_period_unc, alt_period_b, alt_period_unc_b = get_lc_and_period(gaia_id,cluster=cluster,lc_source=lc_source,use_eleanor=use_eleanor,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist,size=size,sigma_thresh=sigma_thresh,tpfplot=tpfplot,sec=sec,disp_sec=disp_sec)

        if len(lc_pca) > 1:
            try:
                pdg = lc_pca.to_periodogram()
            except IndexError:
                lc_pca = [-1]

        if len(lc_pca) == 1:
            print('Finished with star.')

            perTbl.loc[id,'period'] = period
            perTbl.loc[id,'period_unc'] = period_unc
            perTbl.loc[id,'alt_period'] = alt_period
            perTbl.loc[id,'alt_period_unc'] = alt_period_unc
            perTbl.loc[id,'alt_period_b'] = alt_period_b
            perTbl.loc[id,'alt_period_unc_b'] = alt_period_unc_b
            perTbl.loc[id,'e_period_hwhm'] = e_period_hwhm
            perTbl.loc[id,'e_period_mad'] = e_period_mad
            perTbl.loc[id,'e_period_std'] = e_period_std
            perTbl.loc[id,'maxheight'] = maxheight
            perTbl.loc[id,'n_periodmults'] = 0
            perTbl.loc[id,'bestsector'] = bestsec
            perTbl.loc[id,'pdgm_peak'] = -1
            perTbl.loc[id,'HLSP_best'] = label
            perTbl.loc[id, 'xpixel_pos'] = pos[0]
            perTbl.loc[id, 'ypixel_pos'] = pos[1]

            fig2 = plt.figure(figsize=(9,12))


            fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')


            perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')
            perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')


        else:

            lc_pca = lc_pca.remove_nans().remove_outliers()
            single_indx = (lc_pca.time > t_start) & (lc_pca.time < t_end)
            single_lc_pca = lk.LightCurve(time=lc_pca.time[single_indx], flux = lc_pca.flux[single_indx], flux_err = lc_pca.flux_err[single_indx])

            ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
            ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
            ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

            #repl_lc = cleanlc


            #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
            #periods[n], period_uncs[n] = get_acf_period(repl_lc)

            #for i in range(len(planets)):
            #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
            #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
                #tpf_p=TPF.plot(aperture_mask=apers)
            #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)
            #print(lc_pca.time)


            #lc_pca.scatter()

            #pdg = lc_pca.to_periodogram()

            #maxheights[n] = maxheight
            n_periodmults = len(periodmults)
            #bestsectors[n] = bestsec
            pdgm_peak = pdg.period_at_max_power.value

            lctbl = lc_pca.to_table()#.to_pandas()
            lctbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)
            #lctbl.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Light_Curves/'+np.str(n)+'_'+gaia_id+'_lc.csv')


            pdgtbl = pdg.to_table()
            #Commented to save space
            ###pdgtbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Periodograms/'+np.str(num)+'_'+np.str(gaia_id)+'_pdg.txt',format='ascii',overwrite=True)

            #fig2 = plt.figure(figsize=(15,20))
            #fig2 = plt.figure(figsize=(9,12))
            fig2 = plt.figure(figsize=(9,12))

            gs=gridspec.GridSpec(ncols=3, nrows=4,hspace=.3,wspace=.3)
            gss=gridspec.GridSpec(ncols=3, nrows=16,hspace=.5,wspace=.3)

            #gs=gridspec.GridSpec(ncols=7, nrows=4,hspace=.3,wspace=.3)

            ax1=plt.subplot(gs[0:1,0:])

            if (label == 1) & (lc_source=='PATHOS'):
                lbl = 'PT'
            elif (label == 1) & (lc_source=='CDIPS'):
                lbl = 'CD'
            else:
                lbl = 'el'
            #ax1.set_title('P = '+np.str(np.round(periods[n],2))+'$\pm$'+np.str(np.round(period_uncs[n],2))+  '        $G$ = '+gmag,loc='left')
            ax1.set_title('P = '+np.str(np.round(period,2))+'$\pm$'+np.str(np.round(period_unc,2))+  '    $G$ = '+np.str(gmag) +'    '+lbl,loc='left')

            ax1.set_title('Gaia DR2 '+ np.str(gaia_id),loc='right')
            #ax[0].text(.2,.9,gaia_id,horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
            single_lc_pca.scatter(ax=ax1,c='black',s=3)

            #print(peakinds)
            #print(valinds)

            ax2=plt.subplot(gs[1:2,0:])

            ax2.set_title('AltP = '+np.str(np.round(alt_period,2))+'$\pm$'+np.str(np.round(alt_period_unc,2)),loc='left')
            ax2.set_title('AltP B = '+np.str(np.round(alt_period_b,2))+'$\pm$'+np.str(np.round(alt_period_unc_b,2)),loc='right')

            ax2.plot(tau, emp_acorr_smooth,color='blue',zorder=0)
            ax2.scatter(tau[peakinds],emp_acorr_smooth[peakinds],color='blue',s=50)
            ax2.scatter(tau[valinds[0]],emp_acorr_smooth[valinds[0]],color='orange',s=50)
            ax2.scatter(tau[valinds[1]],emp_acorr_smooth[valinds[1]],color='orange',s=50)
            ax2.axvline(period, color="green", alpha=0.9)
            if len(periodmults) != 0:
                for xx in range(1,6):
                    ax2.axvline(periodmults[1]*xx,color='k',alpha=.75)#,ls='dashed')
                    ax2.axvline(periodmults[1]*.8, color="k", alpha=0.5,ls="dashed")
                    ax2.axvline(periodmults[1]*1.2, color="k", alpha=0.5,ls="dashed")
            for yy in range(len(periodmults)):
                ax2.axvline(periodmults[yy],color='k',alpha=.5,ls='dashed')
            ax2.set_xlim(0,25)
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Acorr')

            ax3=plt.subplot(gs[2:3,0:])
            lc_pca.scatter(ax=ax3,s=.3,c='black')
            ax3.axvline(t_start,color='k',ls='dashed',alpha=.5)
            ax3.axvline(t_end,color='k',ls='dashed',alpha=.5)

            ax4=plt.subplot(gs[3:4,0:1])
            #tpf_p=TPF.plot(aperture_mask=apers,ax=ax4,show_colorbar=False)
            interval = MinMaxInterval() #ZScaleInterval()
            stretch = SqrtStretch() #LogStretch()
            #ax4.imshow(stretch(interval(TPF)),origin='lower')
            ax4.imshow(stretch(interval(TPF)),origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)])
            #ax4.imshow(apers,origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)],alpha=.5, cmap=binary_r)

            #ax4.imshow(apers,alpha=0.5,cmap='spring_r')
            for i in range(apers.shape[0]):
                for j in range(apers.shape[1]):
                    if apers[i, j]:
                        #print('!')
                        #print(j+np.int(pos[1]-TPF.shape[1]/2)-.5)
                        #print(i+np.int(pos[0]-TPF.shape[0]/2)-.5)
                        #plt.scatter(i,j,color='pink',alpha=.8,marker='s',s=50,facecolors='None')
                        ax4.scatter(i+np.int(pos[0]-TPF.shape[0]/2)+.5,j+np.int(pos[1]- TPF.shape[1]/2)+.5,color='pink',alpha=.8,marker='s',s=20,facecolors='None')

                        #ax4.add_patch(patches.Rectangle((j-.5, i-.5),
                        #                                1, 1, color='pink', fill=False,
                        #                                alpha=.75))
                        #ax4.add_patch(patches.Rectangle((j+np.int(pos[1]- TPF.shape[1]/2)-.5, i+np.int(pos[0]-TPF.shape[0]/2)-.5),
                        #                                1, 1, color='pink', fill=False,
                        #                                alpha=.75))

            ax4.set_title('')

            ax5=plt.subplot(gs[3:4,1:])
            pdg.plot(view='period',ax=ax5,color='k',ylabel='')
            ax5.set_xlim(0,25)
            #ax5.text(.85,.9,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
            if np.float(bprp) < 1:
                bprp_clr = 'blue'
            else:
                bprp_clr = 'red'
            ax5.text(.5,1.05,'BP-RP = '+bprp,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color=bprp_clr)
            ax5.text(.85,1.05,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
            ax5.axvline(pdg.period_at_max_power.value, color="brown", alpha=0.75,ls="dashed")
            #if closeids[n] != 0:

            if -1 not in closeids:

                for m in range(len(closeids)):
                    if closeids[m] in mems.index.values:
                       clr = 'green'
                   #elif (np.abs(gmags[n]-closemags[n][m]) < 1) & (closeseps[n][m] < 1):
                    elif (closemags[m] - mems.loc[gaia_id,'phot_g_mean_mag'] < 1) & (closeseps[m] < 1):
                       clr = 'red'
                    else:
                       clr = 'darkorange'

                    ax5.text(.6,.9 - (m*.075), np.str(closeids[m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                    ax5.text(.85,.9 - (m*.075), np.str(closemags[m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                   #ax5.text(.95,.9 - (m*.075), np.str(closeseps[n][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                    ax5.text(.95,.9 - (m*.075), np.str(closebprp[m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)

                    ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                    ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                    ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

                    #repl_lc = cleanlc


                    #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
                    #periods[n], period_uncs[n] = get_acf_period(repl_lc)

                    #for i in range(len(planets)):
                    #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
                    #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
                    #tpf_p=TPF.plot(aperture_mask=apers)
                    #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)

            #if 0 not in closeids[0]:
            #    for m in range(len(closeids[0])):
            #        if closeids[0][m] in ngc2516mems['source_id']:
            #            clr = 'green'
            #        else:
            #            clr = 'darkorange'
            #        ax5.text(.6,.9 - (m*.075), np.str(closeids[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
            #        ax5.text(.85,.9 - (m*.075), np.str(closemags[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
            #        ax5.text(.95,.9 - (m*.075), np.str(closeseps[0][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)

            #fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/Figs/DVRs/'+np.str(n)+'_'+gaia_id+'_dvr.pdf',overwrite=True,bbox_inches='tight')
            fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')
            plt.close('all')

            if generate_neighbor_lcs:
                fig3 = plt.figure(figsize=(8,24))
                if (-1 not in closeids):
                    if len(closeids) <= 8:
                        rng = range(len(closeids))
                    else:
                        rng = range(8)
                    #for m in range(len(closeids)):
                    for m in rng:
                        #print(closeids[m])
                   #lc_pca, TPF, apers, periods[n], period_uncs[n], e_period_hwhm[n], e_period_mad[n], e_period_std[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label[n], bstsc = get_lc_and_period(0,[closeids[n][m]],sec=bestsec)
                   #print(n,m)
                        lc_pca_neighbor, TPF_neighbor, apers_neighbor, finalperiod_neighbor, period_unc_neighbor, e_period_hwhm_neighbor, e_period_mad_neighbor, e_period_std_neighbor, tau_neighbor, emp_acorr_smooth_neighbor, peakinds_neighbor, valinds_neighbor, maxheight_neighbor, t_start_neighbor, t_end_neighbor, periodmults_neighbor, pos_neighbor, label_neighbor, bestsector_neighbor, alt_period_neighbor, alt_period_unc_neighbor, alt_period_neighbor_b, alt_period_unc_neighbor_b = get_lc_and_period(closeids[m], cluster=cluster,lc_source=lc_source,use_eleanor=use_eleanor,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist,size=size,sigma_thresh=sigma_thresh,tpfplot=tpfplot,sec=sec,disp_sec=disp_sec)

                        single_indx_neighbor = (lc_pca_neighbor.time > t_start_neighbor) & (lc_pca_neighbor.time < t_end_neighbor)
                        single_lc_pca_neighbor = lk.LightCurve(time=lc_pca_neighbor.time[single_indx_neighbor], flux = lc_pca_neighbor.flux[single_indx_neighbor], flux_err = lc_pca_neighbor.flux_err[single_indx_neighbor])

                        lctbl_neighbor = single_lc_pca_neighbor.to_table()
                        lctbl_neighbor.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Neighbor_Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)

                        pdg_neighbor = lc_pca_neighbor.to_periodogram()
                        pdgtbl_neighbor = pdg_neighbor.to_table()
                        #Commented to save space
                        ###pdgtbl_neighbor.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Neighbor_Periodograms/'+np.str(num)+'_'+np.str(gaia_id)+'_pdg.txt',format='ascii',overwrite=True)

                        ax = plt.subplot(gss[0+m*2:1+m*2,0:])
                        axx = plt.subplot(gss[0+m*2+1:1+m*2+1,0:])

                        # ax.set_title(np.str(closeids[n][m]))
                        #smth = 500
                        #if (periods[n] < 1) & (periods[n] >= 0.1):
                        #       smth = 150
                        #  elif periods[n] > 3:
                        #       smth = 800
                        #  elif periods[n] > 6:
                        #       smth = 1200
                        #finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(single_lc_pca,smth=smth)
                        #finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(lc_pca,smth=smth)


                        #single_lc_pca.scatter(ax=plt.subplot(gs[0+m:1+m,0:]),c='black',s=3)
                        single_lc_pca_neighbor.scatter(ax=ax,c='black',s=1)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        axx.plot(tau_neighbor, emp_acorr_smooth_neighbor,color='blue',zorder=0)
                        axx.scatter(tau_neighbor[peakinds_neighbor],emp_acorr_smooth_neighbor[peakinds_neighbor],color='blue',s=10)
                        axx.scatter(tau_neighbor[valinds_neighbor[0]],emp_acorr_smooth_neighbor[valinds_neighbor[0]],color='orange',s=10)
                        axx.scatter(tau_neighbor[valinds_neighbor[1]],emp_acorr_smooth_neighbor[valinds_neighbor[1]],color='orange',s=10)

                        axx.axvline(finalperiod_neighbor, color="green", alpha=0.9)
                        if len(periodmults_neighbor) != 0:
                            for xx in range(1,6):
                                axx.axvline(periodmults_neighbor[1]*xx,color='k',alpha=.75)#,ls='dashed')
                                axx.axvline(periodmults_neighbor[1]*.8, color="k", alpha=0.5,ls="dashed")
                                axx.axvline(periodmults_neighbor[1]*1.2, color="k", alpha=0.5,ls="dashed")
                        for yy in range(len(periodmults_neighbor)):
                           axx.axvline(periodmults_neighbor[yy],color='k',alpha=.5,ls='dashed')
                        axx.set_xlim(0,10)
                        #axx.text(0.2,0.8, np.str(closeids[n][m]),transform=axx.transAxes)
                        axx.text(0.3,0.8, np.str(closeids[m]),transform=axx.transAxes)
                        #axx.text(0.65,0.8, np.str(np.round(gaia_bp_rp,2)),transform=axx.transAxes)
                        axx.text(0.75,0.8, np.str(np.round(finalperiod_neighbor,2)),transform=axx.transAxes,color='blue')
                        axx.text(0.85,0.8, np.str(np.round(single_lc_pca_neighbor.to_periodogram().period_at_max_power.value,2)),color='brown',transform=axx.transAxes)


                fig3.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_neighbors.pdf',overwrite=True,bbox_inches='tight')
                plt.close('all')
            #plt.subplot(gs[1:2])
            plt.close('all')
            #perTbl['period'][n] = periods[n]
            #perTbl['period_unc'][n] = period_uncs[n]
            #perTbl['e_period_hwhm'][n] = e_period_hwhm[n]
            #perTbl['e_period_mad'][n] = e_period_mad[n]
            #perTbl['e_period_std'][n] = e_period_std[n]
            #perTbl['PATHOS'][n] = label[n]
            #print(perTbl)

            perTbl.loc[id,'period'] = period
            perTbl.loc[id,'period_unc'] = period_unc
            perTbl.loc[id,'alt_period'] = alt_period
            perTbl.loc[id,'alt_period_unc'] = alt_period_unc
            perTbl.loc[id,'alt_period_b'] = alt_period_b
            perTbl.loc[id,'alt_period_unc_b'] = alt_period_unc_b
            perTbl.loc[id,'e_period_hwhm'] = e_period_hwhm
            perTbl.loc[id,'e_period_mad'] = e_period_mad
            perTbl.loc[id,'e_period_std'] = e_period_std
            perTbl.loc[id,'maxheight'] = maxheight
            perTbl.loc[id,'n_periodmults'] = n_periodmults
            perTbl.loc[id,'bestsector'] = bestsec
            perTbl.loc[id,'pdgm_peak'] = pdgm_peak
            perTbl.loc[id,'HLSP_best'] = label
            perTbl.loc[id, 'xpixel_pos'] = pos[0]
            perTbl.loc[id, 'ypixel_pos'] = pos[1]

            #print(period)

            #perTbl.reset_index(inplace=True)
            #+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs
            #perTbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat',format='ascii',overwrite=True)

            perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

        perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')
        #perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')

    return

def sectormasks(lc, sector):
    pca_t = lc.time
    timemasks = {
            's1' : np.where((pca_t >= 1325.293) & (pca_t <= 1353.178))[0],
            's2' : np.where((pca_t >= 1354.101) & (pca_t <= 1381.515))[0],
            's3' : np.where((pca_t >= 1385.897) & (pca_t <= 1406.292))[0],
            's4' : np.where((pca_t >= 1410.900) & (pca_t <= 1436.849))[0],
            's5' : np.where((pca_t >= 1437.826) & (pca_t <= 1464.400))[0],
            's6' : np.where((pca_t >= 1468.270) & (pca_t <= 1490.044))[0],
            's7' : np.where((pca_t >= 1491.626) & (pca_t <= 1516.085))[0],
            's8' : np.where((pca_t >= 1517.342) & (pca_t <= 1542.000))[0],
            's9' : np.where((pca_t >= 1543.216) & (pca_t <= 1568.475))[0],
            's10' : np.where((pca_t >= 1569.432) & (pca_t <= 1595.680))[0],
            's11' : np.where((pca_t >= 1596.772) & (pca_t <= 1623.891))[0],
            's12' : np.where((pca_t >= 1624.950) & (pca_t <= 1652.891))[0],
            's13' : np.where((pca_t >= 1653.915) & (pca_t <= 1682.357))[0],
            's14' : np.where((pca_t >= 1683.34838) & (pca_t <= 1710.20392))[0],
            's15' : np.where((pca_t >= 1711.35947) & (pca_t <= 1737.40946))[0],
            's16' : np.where((pca_t >= 1738.64697) & (pca_t <= 1763.31918))[0],
            's17' : np.where((pca_t >= 1764.67891) & (pca_t <= 1789.69417))[0],
            's18' : np.where((pca_t >= 1790.65111) & (pca_t <= 1815.03026))[0],
            's19' : np.where((pca_t >= 1816.07749) & (pca_t <= 1841.14831))[0],
            's20' : np.where((pca_t >= 1842.49831) & (pca_t <= 1868.82191))[0],
            's21' : np.where((pca_t >= 1870.42885) & (pca_t <= 1897.78023))[0],
            's22' : np.where((pca_t >= 1899.30103) & (pca_t <= 1926.49269))[0],
            's23' : np.where((pca_t >= 1928.09965) & (pca_t <= 1954.87464))[0],
            's24' : np.where((pca_t >= 1955.78990) & (pca_t <= 1982.28017))[0],
            's25' : np.where((pca_t >= 1983.62738) & (pca_t <= 2009.30515))[0],
            's26' : np.where((pca_t >= 2010.26209) & (pca_t <= 2035.13430))[0],
            's27' : np.where((pca_t >= 2036.27320) & (pca_t <= 2060.64125))[0],
            's28' : np.where((pca_t >= 2061.84540) & (pca_t <= 2087.09678))[0],
            's29' : np.where((pca_t >= 2088.23429) & (pca_t <= 2114.43289))[0],
            's30' : np.where((pca_t >= 2115.88011) & (pca_t <= 2143.22177))[0],
            's31' : np.where((pca_t >= 2144.50927) & (pca_t <= 2169.94398))[0],
            's32' : np.where((pca_t >= 2174.21898) & (pca_t <= 2200.23147))[0],
            's33' : np.where((pca_t >= 2201.72730) & (pca_t <= 2227.57173))[0],
            's34' : np.where((pca_t >= 2228.74533) & (pca_t <= 2254.06476))[0],
            's35' : np.where((pca_t >= 2254.98421) & (pca_t <= 2279.97864))[0],
            's36' : np.where((pca_t >= 2280.89808) & (pca_t <= 2305.98835))[0],
            's37' : np.where((pca_t >= 2307.23418) & (pca_t <= 2332.57862))[0],
            's38' : np.where((pca_t >= 2333.84945) & (pca_t <= 2360.55083))[0],
            's39' : np.where((pca_t >= 2361.76612) & (pca_t <= 2389.71750))[0]}

    stitched_lc = lk.LightCurve(time=[],flux=[],flux_err=[])
    for s in sector:
        singlesector_lc = lc[timemasks[s]]
        stitched_lc = stitched_lc.append(singlesector_lc)

    return stitched_lc

def perform_period_analysis_hj(mems,cluster=None,directory=None,npix=2,lc_source='PATHOS',use_eleanor=True,generate_neighbor_lcs=True,smth=100,min_prominence=0.1,min_horiz_dist=10,size=(31,31),sigma_thresh=5,tpfplot=False,sec='all',disp_sec=None,do_psf=False):

    start_index = mems.index[0]
    n_id = mems[['source_id']]
    n_id['num'] = n_id.index

    mems = mems.set_index('source_id')
    n_id = n_id.set_index('source_id')
    #print(n_id)

    gmags = mems['phot_g_mean_mag'].values

    #perTbl = Table(data=[gaia_ids,periods,period_uncs,e_period_hwhm,e_period_mad,e_period_std,label],names=['source_id','period','period_unc','e_period_hwhm','e_period_mad','e_period_std','PATHOS'])
    #perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat')
    gaia_ids = mems.index

    #UNCOMMENT WHEN CREATED
    if cluster==None:
        filetest = glob.glob('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_ptbl_'+lc_source+'.csv')
    else:
        filetest = glob.glob('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')


    if len(filetest) == 0:
        periods = np.zeros(len(mems))
        period_uncs = np.copy(periods)
        alt_periods = np.zeros(len(mems))
        alt_period_uncs = np.copy(periods)
        alt_periods_b = np.zeros(len(mems))
        alt_period_uncs_b = np.copy(periods)
        e_period_hwhm = np.copy(periods)
        e_period_mad = np.copy(periods)
        e_period_std = np.copy(periods)
        label = np.zeros(len(mems),dtype=int)
        maxheights = np.copy(periods)
        n_periodmults = np.copy(label)
        bestsectors = np.copy(label)
        pdgm_peaks = np.copy(periods)
        xpixel_positions = np.copy(periods)
        ypixel_positions = np.copy(periods)

    #if initialize:
        #perTbl = Table(data=[gaia_ids,periods,period_uncs,e_period_hwhm,e_period_mad,e_period_std,label],names=['source_id','period','period_unc','e_period_hwhm','e_period_mad','e_period_std','PATHOS'])

        #perTbl = pd.DataFrame({'source_id':gaia_ids, 'period':periods, 'period_unc':period_uncs, 'e_period_hwhm':e_period_hwhm, 'e_period_mad':e_period_mad, 'e_period_std':e_period_std, 'HLSP_best':label,
        #                        'maxheight':maxheights, 'n_periodmults': n_periodmults, 'bestsector':bestsectors, 'pdgm_peak':pdgm_peaks})

        perTbl = pd.DataFrame({'source_id':np.array([],dtype='int64'), 'period':np.array([]), 'period_unc':np.array([]), 'alt_period':np.array([]), 'alt_period_unc':np.array([]), 'alt_period_b':np.array([]), 'alt_period_unc_b':np.array([]), 'e_period_hwhm':np.array([]), 'e_period_mad':np.array([]), 'e_period_std':np.array([]), 'HLSP_best':np.array([],dtype=np.int),
                                'maxheight':np.array([]), 'n_periodmults': np.array([],dtype=np.int), 'bestsector':np.array([],dtype=np.int), 'pdgm_peak':np.array([]), 'xpixel_pos':np.array([]), 'ypixel_pos':np.array([])})
        perTbl = perTbl.set_index('source_id')

#directory='HJ_hosts/HJ_periods'
        if cluster==None:
            perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_ptbl_'+lc_source+'.csv')
        else:
            perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

    else:
        #perTbl = ascii.read('/Users/bhealy/Documents/PhD_Thesis/Phase_3/M48_periods/pathos_lcgen/M48_ptbl_pathos_neighbors.dat')
        perTbl = pd.read_csv(filetest[0], index_col='source_id')
        #perTbl = perTbl.set_index('source_id')
        #pd.read_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')

    #import OpenSSL
    #import astroquery.mast.core

    except_count = 0
    total_failures = 0
    #for n in range(371,372):
    #for n in range(496,len(ngc2516mems)):

    #for n in range(len(mems)):
    #    n= n+start_index

    for id in gaia_ids:
    #for n in mems.index:

    #for n in range(93,94):


    #for n in range(256,257):
    #for n in range(58,len(mems)):

        #for attempt in range(10):
        #    try:

    #requests.exceptions.SSLError: HTTPSConnectionPool(host='mast.stsci.edu', port=443): Max retries exceeded with url: /api/v0.1/Download/file?uri=mast:HLSP/eleanor/postcards/s0010/3-3/hlsp_eleanor_tess_ffi_postcard-s0010-3-3-cal-1000-0052_tess_v2_pc.fits (Caused by SSLError(SSLError("bad handshake: SysCallError(-1, 'Unexpected EOF')",),))
    #ssl.SSLError
        #for n in range(38,len(ngc2516mems)):
        #for n in range(28,29):
        #for n in range(0,3):
        #for n in range(165,168):

        gaia_id = id
        num = n_id.loc[gaia_id,'num']
        print('Star', num, gaia_id)
        gmag = np.str(np.round(mems.loc[id,'phot_g_mean_mag'],2))
        bprp = np.str(np.round(mems.loc[id,'bp_rp'],2))
        #print(n, gaia_id)
        #toi = toi_unique[n]

        if cluster==None:
            closeids, closemags, closeseps, closebprp = identify_neighbors(mems.loc[[gaia_id]], npix=npix, directory=directory)
        else:
            closeids, closemags, closeseps, closebprp = identify_neighbors(mems.loc[[gaia_id]], cluster, npix=npix)

        #planets = toi_dict[toi_unique[n]]

        #lc, TPF, apers = get_toi_lc(n,ticids)
        #lc, TPF, apers = get_toi_lc(n,ngc2516mems)

        #lc_pca, TPF, apers, periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos = get_lc_and_period(n,ngc2516mems)
        lc_pca, TPF, apers, period, period_unc, e_period_hwhm, e_period_mad, e_period_std, tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label, bestsec, alt_period, alt_period_unc, alt_period_b, alt_period_unc_b = get_lc_and_period(gaia_id,cluster=cluster,lc_source=lc_source,use_eleanor=use_eleanor,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist,size=size,sigma_thresh=sigma_thresh,tpfplot=tpfplot,sec=sec,disp_sec=disp_sec,do_psf=do_psf)

        if len(lc_pca) == 1:
            print('Finished with star.')

            perTbl.loc[id,'period'] = period
            perTbl.loc[id,'period_unc'] = period_unc
            perTbl.loc[id,'alt_period'] = alt_period
            perTbl.loc[id,'alt_period_unc'] = alt_period_unc
            perTbl.loc[id,'alt_period_b'] = alt_period_b
            perTbl.loc[id,'alt_period_unc_b'] = alt_period_unc_b
            perTbl.loc[id,'e_period_hwhm'] = e_period_hwhm
            perTbl.loc[id,'e_period_mad'] = e_period_mad
            perTbl.loc[id,'e_period_std'] = e_period_std
            perTbl.loc[id,'maxheight'] = maxheight
            perTbl.loc[id,'n_periodmults'] = 0
            perTbl.loc[id,'bestsector'] = bestsec
            perTbl.loc[id,'pdgm_peak'] = -1
            perTbl.loc[id,'HLSP_best'] = label
            perTbl.loc[id, 'xpixel_pos'] = pos[0]
            perTbl.loc[id, 'ypixel_pos'] = pos[1]

            fig2 = plt.figure(figsize=(9,12))

            if cluster==None:
                fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')
            else:
                fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')

            if cluster!=None:
                perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')
                perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')
            else:
                perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_ptbl_'+lc_source+'.csv')
                perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_selected_ptbl_'+lc_source+'.csv')


        else:

            lc_pca = lc_pca.remove_nans().remove_outliers()
            single_indx = (lc_pca.time > t_start) & (lc_pca.time < t_end)
            single_lc_pca = lk.LightCurve(time=lc_pca.time[single_indx], flux = lc_pca.flux[single_indx], flux_err = lc_pca.flux_err[single_indx])

            ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
            ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
            ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

            #repl_lc = cleanlc


            #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
            #periods[n], period_uncs[n] = get_acf_period(repl_lc)

            #for i in range(len(planets)):
            #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
            #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
                #tpf_p=TPF.plot(aperture_mask=apers)
            #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)
            #print(lc_pca.time)

            pdg = lc_pca.to_periodogram()

            #maxheights[n] = maxheight
            n_periodmults = len(periodmults)
            #bestsectors[n] = bestsec
            pdgm_peak = pdg.period_at_max_power.value

            lctbl = lc_pca.to_table()#.to_pandas()

            if cluster==None:
                lctbl.write('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/Figs/Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)
            else:
                lctbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)


            #lctbl.set_index('source_id').to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Light_Curves/'+np.str(n)+'_'+gaia_id+'_lc.csv')


            pdgtbl = pdg.to_table()
            #Commented to save space
            ###pdgtbl.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Periodograms/'+np.str(num)+'_'+np.str(gaia_id)+'_pdg.txt',format='ascii',overwrite=True)

            #fig2 = plt.figure(figsize=(15,20))
            #fig2 = plt.figure(figsize=(9,12))
            fig2 = plt.figure(figsize=(9,12))

            gs=gridspec.GridSpec(ncols=3, nrows=4,hspace=.3,wspace=.3)
            gss=gridspec.GridSpec(ncols=3, nrows=16,hspace=.5,wspace=.3)

            #gs=gridspec.GridSpec(ncols=7, nrows=4,hspace=.3,wspace=.3)

            ax1=plt.subplot(gs[0:1,0:])

            if (label == 1) & (lc_source=='PATHOS'):
                lbl = 'PT'
            elif (label == 1) & (lc_source=='CDIPS'):
                lbl = 'CD'
            else:
                lbl = 'el'
            #ax1.set_title('P = '+np.str(np.round(periods[n],2))+'$\pm$'+np.str(np.round(period_uncs[n],2))+  '        $G$ = '+gmag,loc='left')
            ax1.set_title('P = '+np.str(np.round(period,2))+'$\pm$'+np.str(np.round(period_unc,2))+  '    $G$ = '+np.str(gmag) +'    '+lbl,loc='left')

            ax1.set_title('Gaia DR2 '+ np.str(gaia_id),loc='right')
            #ax[0].text(.2,.9,gaia_id,horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
            single_lc_pca.scatter(ax=ax1,c='black',s=3)

            #print(peakinds)
            #print(valinds)

            ax2=plt.subplot(gs[1:2,0:])

            ax2.set_title('AltP = '+np.str(np.round(alt_period,2))+'$\pm$'+np.str(np.round(alt_period_unc,2)),loc='left')
            ax2.set_title('AltP B = '+np.str(np.round(alt_period_b,2))+'$\pm$'+np.str(np.round(alt_period_unc_b,2)),loc='right')

            ax2.plot(tau, emp_acorr_smooth,color='blue',zorder=0)
            ax2.scatter(tau[peakinds],emp_acorr_smooth[peakinds],color='blue',s=50)
            ax2.scatter(tau[valinds[0]],emp_acorr_smooth[valinds[0]],color='orange',s=50)
            ax2.scatter(tau[valinds[1]],emp_acorr_smooth[valinds[1]],color='orange',s=50)
            ax2.axvline(period, color="green", alpha=0.9)
            if len(periodmults) != 0:
                for xx in range(1,6):
                    ax2.axvline(periodmults[1]*xx,color='k',alpha=.75)#,ls='dashed')
                    ax2.axvline(periodmults[1]*.8, color="k", alpha=0.5,ls="dashed")
                    ax2.axvline(periodmults[1]*1.2, color="k", alpha=0.5,ls="dashed")
            for yy in range(len(periodmults)):
                ax2.axvline(periodmults[yy],color='k',alpha=.5,ls='dashed')
            ax2.set_xlim(0,25)
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Acorr')

            ax3=plt.subplot(gs[2:3,0:])
            lc_pca.scatter(ax=ax3,s=.3,c='black')
            ax3.axvline(t_start,color='k',ls='dashed',alpha=.5)
            ax3.axvline(t_end,color='k',ls='dashed',alpha=.5)

            ax4=plt.subplot(gs[3:4,0:1])
            #tpf_p=TPF.plot(aperture_mask=apers,ax=ax4,show_colorbar=False)
            interval = MinMaxInterval() #ZScaleInterval()
            stretch = SqrtStretch() #LogStretch()
            #ax4.imshow(stretch(interval(TPF)),origin='lower')
            ax4.imshow(stretch(interval(TPF)),origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)])
            #ax4.imshow(apers,origin='lower',extent=[np.int(pos[0] - TPF.shape[0]/2), np.int(pos[0] + TPF.shape[0]/2), np.int(pos[1] - TPF.shape[1]/2), np.int(pos[1] + TPF.shape[1]/2)],alpha=.5, cmap=binary_r)

            #ax4.imshow(apers,alpha=0.5,cmap='spring_r')
            for i in range(apers.shape[0]):
                for j in range(apers.shape[1]):
                    if apers[i, j]:
                        #print('!')
                        #print(j+np.int(pos[1]-TPF.shape[1]/2)-.5)
                        #print(i+np.int(pos[0]-TPF.shape[0]/2)-.5)
                        #plt.scatter(i,j,color='pink',alpha=.8,marker='s',s=50,facecolors='None')
                        ax4.scatter(i+np.int(pos[0]-TPF.shape[0]/2)+.5,j+np.int(pos[1]- TPF.shape[1]/2)+.5,color='pink',alpha=.8,marker='s',s=20,facecolors='None')

                        #ax4.add_patch(patches.Rectangle((j-.5, i-.5),
                        #                                1, 1, color='pink', fill=False,
                        #                                alpha=.75))
                        #ax4.add_patch(patches.Rectangle((j+np.int(pos[1]- TPF.shape[1]/2)-.5, i+np.int(pos[0]-TPF.shape[0]/2)-.5),
                        #                                1, 1, color='pink', fill=False,
                        #                                alpha=.75))

            ax4.set_title('')

            ax5=plt.subplot(gs[3:4,1:])
            pdg.plot(view='period',ax=ax5,color='k',ylabel='')
            ax5.set_xlim(0,25)
            #ax5.text(.85,.9,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
            if np.float(bprp) < 1:
                bprp_clr = 'blue'
            else:
                bprp_clr = 'red'
            ax5.text(.5,1.05,'BP-RP = '+bprp,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color=bprp_clr)
            ax5.text(.85,1.05,'P$_{max} = $'+np.str(np.round(pdg.period_at_max_power.value,2)),horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,color='brown')
            ax5.axvline(pdg.period_at_max_power.value, color="brown", alpha=0.75,ls="dashed")
            #if closeids[n] != 0:

            if -1 not in closeids:

                for m in range(len(closeids)):
                    if closeids[m] in mems.index.values:
                       clr = 'green'
                   #elif (np.abs(gmags[n]-closemags[n][m]) < 1) & (closeseps[n][m] < 1):
                    elif (closemags[m] - mems.loc[gaia_id,'phot_g_mean_mag'] < 1) & (closeseps[m] < 1):
                       clr = 'red'
                    else:
                       clr = 'darkorange'

                    ax5.text(.6,.9 - (m*.075), np.str(closeids[m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                    ax5.text(.85,.9 - (m*.075), np.str(closemags[m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                   #ax5.text(.95,.9 - (m*.075), np.str(closeseps[n][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
                    ax5.text(.95,.9 - (m*.075), np.str(closebprp[m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)

                    ###middletime = (single_lc_pca.time[0] + single_lc_pca.time[-1])/2
                    ###goodindx = (single_lc_pca.time < (middletime - 1.)) | (single_lc_pca.time > (middletime + 2.))
                    ###single_lc_pca = lk.LightCurve(time = single_lc_pca.time[goodindx], flux = single_lc_pca.flux[goodindx], flux_err = single_lc_pca.flux_err[goodindx])

                    #repl_lc = cleanlc


                    #periods[n], period_uncs[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight = get_acf_period(repl_lc)
                    #periods[n], period_uncs[n] = get_acf_period(repl_lc)

                    #for i in range(len(planets)):
                    #    fulltoi = toi_unique[n] +'.'+ toi_dict[toi_unique[n]][i]
                    #    xmatchindx = np.where(np.float(fulltoi) == Xmatch['TOI'])[0][0]
                    #tpf_p=TPF.plot(aperture_mask=apers)
                    #    repl_lc = replace_transits(Xmatch,xmatchindx,repl_lc)

            #if 0 not in closeids[0]:
            #    for m in range(len(closeids[0])):
            #        if closeids[0][m] in ngc2516mems['source_id']:
            #            clr = 'green'
            #        else:
            #            clr = 'darkorange'
            #        ax5.text(.6,.9 - (m*.075), np.str(closeids[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
            #        ax5.text(.85,.9 - (m*.075), np.str(closemags[0][m])+ ' ',color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)
            #        ax5.text(.95,.9 - (m*.075), np.str(closeseps[0][m]),color=clr,fontsize=10,horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,zorder=3)

            #fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/Figs/DVRs/'+np.str(n)+'_'+gaia_id+'_dvr.pdf',overwrite=True,bbox_inches='tight')
            #fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')

            if cluster==None:
                fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')
            else:
                fig2.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_dvr.pdf',overwrite=True,bbox_inches='tight')

            plt.close('all')

            if generate_neighbor_lcs:
                fig3 = plt.figure(figsize=(8,24))
                if (-1 not in closeids):
                    if len(closeids) <= 8:
                        rng = range(len(closeids))
                    else:
                        rng = range(8)
                    #for m in range(len(closeids)):
                    for m in rng:
                        #print(closeids[m])
                   #lc_pca, TPF, apers, periods[n], period_uncs[n], e_period_hwhm[n], e_period_mad[n], e_period_std[n], tau, emp_acorr_smooth, peakinds, valinds, maxheight, t_start, t_end, periodmults, pos, label[n], bstsc = get_lc_and_period(0,[closeids[n][m]],sec=bestsec)
                   #print(n,m)
                        lc_pca_neighbor, TPF_neighbor, apers_neighbor, finalperiod_neighbor, period_unc_neighbor, e_period_hwhm_neighbor, e_period_mad_neighbor, e_period_std_neighbor, tau_neighbor, emp_acorr_smooth_neighbor, peakinds_neighbor, valinds_neighbor, maxheight_neighbor, t_start_neighbor, t_end_neighbor, periodmults_neighbor, pos_neighbor, label_neighbor, bestsector_neighbor, alt_period_neighbor, alt_period_unc_neighbor, alt_period_neighbor_b, alt_period_unc_neighbor_b = get_lc_and_period(closeids[m], cluster=cluster,lc_source=lc_source,use_eleanor=use_eleanor,smth=smth,min_prominence=min_prominence,min_horiz_dist=min_horiz_dist,size=size,sigma_thresh=sigma_thresh,tpfplot=tpfplot,sec=sec,disp_sec=disp_sec,do_psf=do_psf)

                        single_indx_neighbor = (lc_pca_neighbor.time > t_start_neighbor) & (lc_pca_neighbor.time < t_end_neighbor)
                        single_lc_pca_neighbor = lk.LightCurve(time=lc_pca_neighbor.time[single_indx_neighbor], flux = lc_pca_neighbor.flux[single_indx_neighbor], flux_err = lc_pca_neighbor.flux_err[single_indx_neighbor])

                        lctbl_neighbor = single_lc_pca_neighbor.to_table()

                        if cluster!=None:
                            lctbl_neighbor.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Neighbor_Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)
                        else:
                            lctbl_neighbor.write('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/Figs/Neighbor_Light_Curves/'+np.str(num)+'_'+np.str(gaia_id)+'_lc.txt',format='ascii',overwrite=True)

                        pdg_neighbor = lc_pca_neighbor.to_periodogram()
                        pdgtbl_neighbor = pdg_neighbor.to_table()
                        #Commented to save space
                        ###pdgtbl_neighbor.write('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs/Neighbor_Periodograms/'+np.str(num)+'_'+np.str(gaia_id)+'_pdg.txt',format='ascii',overwrite=True)

                        ax = plt.subplot(gss[0+m*2:1+m*2,0:])
                        axx = plt.subplot(gss[0+m*2+1:1+m*2+1,0:])

                        # ax.set_title(np.str(closeids[n][m]))
                        #smth = 500
                        #if (periods[n] < 1) & (periods[n] >= 0.1):
                        #       smth = 150
                        #  elif periods[n] > 3:
                        #       smth = 800
                        #  elif periods[n] > 6:
                        #       smth = 1200
                        #finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(single_lc_pca,smth=smth)
                        #finalperio, period_un, e_period_hwh, e_period_ma, e_period_st, ta, emp_acorr_smoot, peakind, valind, maxheigh, periodmult = get_acf_period(lc_pca,smth=smth)


                        #single_lc_pca.scatter(ax=plt.subplot(gs[0+m:1+m,0:]),c='black',s=3)
                        single_lc_pca_neighbor.scatter(ax=ax,c='black',s=1)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        axx.plot(tau_neighbor, emp_acorr_smooth_neighbor,color='blue',zorder=0)
                        axx.scatter(tau_neighbor[peakinds_neighbor],emp_acorr_smooth_neighbor[peakinds_neighbor],color='blue',s=10)
                        axx.scatter(tau_neighbor[valinds_neighbor[0]],emp_acorr_smooth_neighbor[valinds_neighbor[0]],color='orange',s=10)
                        axx.scatter(tau_neighbor[valinds_neighbor[1]],emp_acorr_smooth_neighbor[valinds_neighbor[1]],color='orange',s=10)

                        axx.axvline(finalperiod_neighbor, color="green", alpha=0.9)
                        if len(periodmults_neighbor) != 0:
                            for xx in range(1,6):
                                axx.axvline(periodmults_neighbor[1]*xx,color='k',alpha=.75)#,ls='dashed')
                                axx.axvline(periodmults_neighbor[1]*.8, color="k", alpha=0.5,ls="dashed")
                                axx.axvline(periodmults_neighbor[1]*1.2, color="k", alpha=0.5,ls="dashed")
                        for yy in range(len(periodmults_neighbor)):
                           axx.axvline(periodmults_neighbor[yy],color='k',alpha=.5,ls='dashed')
                        axx.set_xlim(0,10)
                        #axx.text(0.2,0.8, np.str(closeids[n][m]),transform=axx.transAxes)
                        axx.text(0.3,0.8, np.str(closeids[m]),transform=axx.transAxes)
                        #axx.text(0.65,0.8, np.str(np.round(gaia_bp_rp,2)),transform=axx.transAxes)
                        axx.text(0.75,0.8, np.str(np.round(finalperiod_neighbor,2)),transform=axx.transAxes,color='blue')
                        axx.text(0.85,0.8, np.str(np.round(single_lc_pca_neighbor.to_periodogram().period_at_max_power.value,2)),color='brown',transform=axx.transAxes)

                if cluster!=None:
                    fig3.savefig('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_neighbors.pdf',overwrite=True,bbox_inches='tight')
                else:
                    fig3.savefig('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/'+lc_source+'_LC_DVRs_neighbors/'+np.str(num)+'_'+np.str(gaia_id)+'_neighbors.pdf',overwrite=True,bbox_inches='tight')

                plt.close('all')
            #plt.subplot(gs[1:2])
            plt.close('all')
            #perTbl['period'][n] = periods[n]
            #perTbl['period_unc'][n] = period_uncs[n]
            #perTbl['e_period_hwhm'][n] = e_period_hwhm[n]
            #perTbl['e_period_mad'][n] = e_period_mad[n]
            #perTbl['e_period_std'][n] = e_period_std[n]
            #perTbl['PATHOS'][n] = label[n]
            #print(perTbl)

            perTbl.loc[id,'period'] = period
            perTbl.loc[id,'period_unc'] = period_unc
            perTbl.loc[id,'alt_period'] = alt_period
            perTbl.loc[id,'alt_period_unc'] = alt_period_unc
            perTbl.loc[id,'alt_period_b'] = alt_period_b
            perTbl.loc[id,'alt_period_unc_b'] = alt_period_unc_b
            perTbl.loc[id,'e_period_hwhm'] = e_period_hwhm
            perTbl.loc[id,'e_period_mad'] = e_period_mad
            perTbl.loc[id,'e_period_std'] = e_period_std
            perTbl.loc[id,'maxheight'] = maxheight
            perTbl.loc[id,'n_periodmults'] = n_periodmults
            perTbl.loc[id,'bestsector'] = bestsec
            perTbl.loc[id,'pdgm_peak'] = pdgm_peak
            perTbl.loc[id,'HLSP_best'] = label
            perTbl.loc[id, 'xpixel_pos'] = pos[0]
            perTbl.loc[id, 'ypixel_pos'] = pos[1]

            #print(period)

            #perTbl.reset_index(inplace=True)
            #+cluster+'/'+cluster+'_periods/'+lc_source+'_lcgen/Figs
            #perTbl.write('/Users/bhealy/Documents/PhD_Thesis/NGC_2516/NGC_2516_ptbl_newf.dat',format='ascii',overwrite=True)

            if cluster!=None:
                perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_ptbl_'+lc_source+'.csv')
                perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/Phase_3/'+cluster+'/'+cluster+'_periods'+'/'+lc_source+'_lcgen/'+cluster+'_selected_ptbl_'+lc_source+'.csv')
            else:
                perTbl.to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_ptbl_'+lc_source+'.csv')
                perTbl[perTbl['period'] != -1].to_csv('/Users/bhealy/Documents/PhD_Thesis/'+directory+'/'+lc_source+'_lcgen/HJ_selected_ptbl_'+lc_source+'.csv')

    return
