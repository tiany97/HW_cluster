#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 21:23:59 2023

@author: tianyl
"""



import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import pyplot as plt
import copy
from netCDF4 import Dataset, num2date, date2num
import more_itertools as mit
from scipy import arange, cos, exp
from scipy.interpolate import RegularGridInterpolator
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from matplotlib import pyplot
import matplotlib.patches as patches
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from haversine import haversine, Unit
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import rankdata

import scipy as scipy
from matplotlib import colors as c
import csv
import pandas as pd

import matplotlib as mpl

import cc3d
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl
import matplotlib.cm as cm
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from matplotlib.colors import from_levels_and_colors
import random
import cmaps
from haversine import haversine, Unit
def remove_seasonal(var,yearnumber):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(yearnumber)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(yearnumber):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend 

#%%

#%%
def SEB_ano(var,skt,yearnumber):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(yearnumber)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        temp1 =  np.array([skt[365*i_year+i_box] for i_year in range(yearnumber)])
        temp1 = np.array(len(temp1)*[np.nanmean(temp1,axis=0)])
        temp = temp/(4*5.68*1e-8*temp1**3)
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(yearnumber):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend
    #%%




#%%
def read_cmip6_data(modelname,senarioname,variablename,yearrange,md_cmip6_all):
    # yearstart = 1979
    # yearend = 2014
    # modelname = 'CanESM5/'
    # senarioname = 'historical'
    # variablename = 'tasmax'

    filelist_cmip6 = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname)
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[:len(variablename)+1]==variablename+'_'])
    temp = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filelist_cmip6[0]
    grid_label = temp[temp.find('_g')+1:temp.find('_g')+1+temp[temp.find('_g')+1:].find('_')]
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[-31-len(grid_label)-len(senarioname):-31-len(grid_label)]==senarioname])
    filelist_cmip6_2 = np.array([int(i[-11:-7]) for i in filelist_cmip6])
    filelist_cmip6_1 = np.array([int(i[-20:-16]) for i in filelist_cmip6])
    filelist_cmip6 = np.array(filelist_cmip6)[~((filelist_cmip6_2<yearstart)|(filelist_cmip6_1>yearend))]   
    filelist_cmip6 = np.array(filelist_cmip6)[np.argsort(np.array([int(i[-20:-16]) for i in filelist_cmip6]))]

    var_all= []
    for filename in filelist_cmip6:
        
        filename = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filename
        NCData = Dataset( filename)
        lon_temp= NCData.variables['lon'][:]
        lat_temp = NCData.variables['lat'][:][::-1]


        time = NCData.variables['time']
        dates = list(num2date(time[:], time.units, time.calendar))
        year = np.array([date.year for date in dates])
        year_where = np.where(np.in1d(year,yearrange))[0]
        if len(year_where)==0:
            continue
        print(filename)
        var_temp = NCData.variables[variablename][year_where][:,::-1,:]
        var_temp[var_temp>1000000]=np.nan
        var_all.extend(var_temp)

    temp = int(len(lon_temp)*0.5)
    lon_temp = np.concatenate((lon_temp[temp:]-360 ,lon_temp[:temp]) )
    var_all = np.squeeze(np.array(var_all))
    var_all = np.squeeze(var_all[md_cmip6_all!=229])
    var_all = np.concatenate((var_all[:,:,temp:] ,var_all[:,:,:temp]),axis=2 )

    lon_unin = np.arange(-180,180,2.5)
    lat_unin = np.arange(-90,90,2.5)[::-1]
    LON_unin, LAT_unin = np.meshgrid(lon_unin, lat_unin)
    LAT_unin2 = copy.deepcopy(LAT_unin)
    LON_unin2 = copy.deepcopy(LON_unin)
    LAT_unin2[LAT_unin2>np.nanmax(lat_temp)] = np.nanmax(lat_temp)
    LAT_unin2[LAT_unin2<np.nanmin(lat_temp)] = np.nanmin(lat_temp)
    LON_unin2[LON_unin2>np.nanmax(lon_temp)] = np.nanmax(lon_temp)
    LON_unin2[LON_unin2<np.nanmin(lon_temp)] = np.nanmin(lon_temp)
    var_all_new =[]
    for i in var_all:

        my_interpolating_function = RegularGridInterpolator((lat_temp[::-1],lon_temp), i[::-1])
        var_all_new.append(my_interpolating_function((LAT_unin2.ravel(), LON_unin2.ravel())).reshape((len(LAT_unin2[:,0]),len(LON_unin2[0]))))
    return np.array(var_all_new)


HW_ssr_all1_all = []
HW_slfh_all1_all = []
HW_adv_all1_all = []
HW_strd_all1_all = []

HW_ssr_all2_all = []
HW_slfh_all2_all = []
HW_adv_all2_all = []
HW_strd_all2_all = []
for modelname in ['CanESM5/','INM-CM4-8/','INM-CM5-0/','MRI-ESM2-0/','CMCC-ESM2/','IPSL-CM6A-LR/','MPI-ESM1-2-LR/']:
    yearstart=2015
    yearend=2060
    vername = '_235'+'_'+str(yearstart)+'_'+str(yearend)
    var = 'skt'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_skt_all = copy.deepcopy(all) 

    var = 'ssr'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_ssr_all = copy.deepcopy(all)

    #

    var = 'strd'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_strd_all = copy.deepcopy(all)

    #

    var = 'sshf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ -float(i) for i in row]))
    HW_sshf_all = copy.deepcopy(all)

    #

    var = 'slhf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ -float(i) for i in row]))
    HW_slhf_all = copy.deepcopy(all)

    var = 'adv'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adv_all = copy.deepcopy(all)

    var = 'adiabatic'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adiabatic_all = copy.deepcopy(all)


    HW_q_all = []  
    for iii in range(len(HW_slhf_all)):
        HW_q_all.append(HW_skt_all[iii]-HW_ssr_all[iii]-HW_strd_all[iii]-HW_slhf_all[iii]-HW_sshf_all[iii])

    #%%

    HW_SEB_all = []
    HW_SEB_all1 = []
    for iii in range(len(HW_slhf_all)):
        SEB_event = np.array([HW_adv_all[iii],HW_adiabatic_all[iii], HW_ssr_all[iii],HW_strd_all[iii],HW_slhf_all[iii],HW_sshf_all[iii],HW_q_all[iii]]).T
        for ii in  SEB_event:
            if np.isnan(ii/np.sum([np.abs(k) for k in ii])).any():
                #HW_SEB_all.append(np.array([np.nan,np.nan,np.nan,np.nan]))
                continue
            else:
                HW_SEB_all.append(np.array([ii/np.sum([np.abs(k) for k in ii])]))
                HW_SEB_all1.append(np.array(ii))
    HW_SEB_all = np.squeeze( np.array(HW_SEB_all))
    HW_SEB_all1 = np.squeeze( np.array(HW_SEB_all1))

    var = 'ssr'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_ssr_all_re = copy.deepcopy(all) 
    var = 'slhf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ 1-float(i) for i in row]))
    HW_slhf_all_re = copy.deepcopy(all) 
    var = 'adv'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adv_all_re = copy.deepcopy(all) 
    var = 'strd'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_strd_all_re = copy.deepcopy(all) 
    
    HW_SEB_all_re = []
    for iii in range(len(HW_slhf_all)):
        SEB_event = np.array([HW_adv_all[iii],HW_adiabatic_all[iii], HW_ssr_all[iii],HW_strd_all[iii],HW_slhf_all[iii],HW_sshf_all[iii],HW_q_all[iii]]).T
        SEB_event1 = np.array([HW_ssr_all_re[iii],HW_slhf_all_re[iii], HW_adv_all_re[iii],HW_strd_all_re[iii]]).T
        for ii,jj in  zip(SEB_event,SEB_event1 ):
            if np.isnan(ii/np.sum([np.abs(k) for k in ii])).any():
                #HW_SEB_all.append(np.array([np.nan,np.nan,np.nan,np.nan]))
                continue
            else:
                HW_SEB_all_re.append(np.array(jj))
    HW_SEB_all_re = np.squeeze( np.array(HW_SEB_all_re))



    vername = '_235_'+str(yearstart)+'_'+str(yearend)
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_labels'+vername+'.csv'
    HW_labels_all1 = []

    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            temp = []
            for i_i,i in enumerate(row):
                if np.isnan(float(i)):
                    temp.append(np.nan)

                else:
                    temp.append(round(float(i)))
                    HW_labels_all1.append(round(float(i)))

            all.append(np.array(temp))
    HW_labels_all = copy.deepcopy(all)
    HW_labels_all1 = np.array(HW_labels_all1)
    SEB1=np.load(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_2.5_degree/SEB_era5_2.5_degree', allow_pickle=True)
    SEB_model = []
    for i in range(4):
        SEB_model.append(np.nanmean(HW_SEB_all[HW_labels_all1==i],axis=0)*100)
    dis_all = []
    for i in SEB1:
        temp = []
        for j in SEB_model:
            temp.append(np.sum([ii*ii for ii in i-j]))
        dis_all.append(temp)
    label_name = [np.argmin(i) for i in dis_all]

    HW_ssr_all1=np.array(HW_SEB_all_re)[:,0][np.in1d(HW_labels_all1 ,label_name [1])]
    HW_slfh_all1=np.array(HW_SEB_all_re)[:,1][np.in1d(HW_labels_all1 ,label_name [1])]
    HW_adv_all1=np.array(HW_SEB_all_re)[:,2][np.in1d(HW_labels_all1 ,label_name [2])]
    HW_strd_all1=np.array(HW_SEB_all_re)[:,3][np.in1d(HW_labels_all1 ,label_name [2])]


    yearstart=2055
    yearend=2100
    vername = '_235'+'_'+str(yearstart)+'_'+str(yearend)
    var = 'skt'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_skt_all = copy.deepcopy(all) 

    var = 'ssr'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_ssr_all = copy.deepcopy(all)

    #

    var = 'strd'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_strd_all = copy.deepcopy(all)

    #

    var = 'sshf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ -float(i) for i in row]))
    HW_sshf_all = copy.deepcopy(all)

    #

    var = 'slhf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ -float(i) for i in row]))
    HW_slhf_all = copy.deepcopy(all)

    var = 'adv'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adv_all = copy.deepcopy(all)

    var = 'adiabatic'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adiabatic_all = copy.deepcopy(all)


    HW_q_all = []  
    for iii in range(len(HW_slhf_all)):
        HW_q_all.append(HW_skt_all[iii]-HW_ssr_all[iii]-HW_strd_all[iii]-HW_slhf_all[iii]-HW_sshf_all[iii])

    #%%

    HW_SEB_all = []
    HW_SEB_all1 = []
    for iii in range(len(HW_slhf_all)):
        SEB_event = np.array([HW_adv_all[iii],HW_adiabatic_all[iii], HW_ssr_all[iii],HW_strd_all[iii],HW_slhf_all[iii],HW_sshf_all[iii],HW_q_all[iii]]).T
        for ii in  SEB_event:
            if np.isnan(ii/np.sum([np.abs(k) for k in ii])).any():
                #HW_SEB_all.append(np.array([np.nan,np.nan,np.nan,np.nan]))
                continue
            else:
                HW_SEB_all.append(np.array([ii/np.sum([np.abs(k) for k in ii])]))
                HW_SEB_all1.append(np.array(ii))
    HW_SEB_all = np.squeeze( np.array(HW_SEB_all))
    HW_SEB_all1 = np.squeeze( np.array(HW_SEB_all1))
    var = 'ssr'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_ssr_all_re = copy.deepcopy(all) 
    var = 'slhf'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ 1-float(i) for i in row]))
    HW_slhf_all_re = copy.deepcopy(all) 
    var = 'adv'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_adv_all_re = copy.deepcopy(all) 
    var = 'strd'
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_return'+var+'_ano_6days_all'+vername+'.csv'
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            all.append(np.array([ float(i) for i in row]))
    HW_strd_all_re = copy.deepcopy(all) 
    
    HW_SEB_all_re = []
    for iii in range(len(HW_slhf_all)):
        SEB_event = np.array([HW_adv_all[iii],HW_adiabatic_all[iii], HW_ssr_all[iii],HW_strd_all[iii],HW_slhf_all[iii],HW_sshf_all[iii],HW_q_all[iii]]).T
        SEB_event1 = np.array([HW_ssr_all_re[iii],HW_slhf_all_re[iii], HW_adv_all_re[iii],HW_strd_all_re[iii]]).T
        for ii,jj in  zip(SEB_event,SEB_event1 ):
            if np.isnan(ii/np.sum([np.abs(k) for k in ii])).any():
                #HW_SEB_all.append(np.array([np.nan,np.nan,np.nan,np.nan]))
                continue
            else:
                HW_SEB_all_re.append(np.array(jj))
    HW_SEB_all_re = np.squeeze( np.array(HW_SEB_all_re))

    vername = '_235_'+str(yearstart)+'_'+str(yearend)
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'HW_labels'+vername+'.csv'
    HW_labels_all1 = []

    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            temp = []
            for i_i,i in enumerate(row):
                if np.isnan(float(i)):
                    temp.append(np.nan)

                else:
                    temp.append(round(float(i)))
                    HW_labels_all1.append(round(float(i)))

            all.append(np.array(temp))
    HW_labels_all = copy.deepcopy(all)
    HW_labels_all1 = np.array(HW_labels_all1)
    SEB1=np.load(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_2.5_degree/SEB_era5_2.5_degree', allow_pickle=True)
    SEB_model = []
    for i in range(4):
        SEB_model.append(np.nanmean(HW_SEB_all[HW_labels_all1==i],axis=0)*100)
    dis_all = []
    for i in SEB1:
        temp = []
        for j in SEB_model:
            temp.append(np.sum([ii*ii for ii in i-j]))
        dis_all.append(temp)
    label_name = [np.argmin(i) for i in dis_all]

    HW_ssr_all2=np.array(HW_SEB_all_re)[:,0][np.in1d(HW_labels_all1 ,label_name [1])]
    HW_slfh_all2=np.array(HW_SEB_all_re)[:,1][np.in1d(HW_labels_all1 ,label_name [1])]
    HW_adv_all2=np.array(HW_SEB_all_re)[:,2][np.in1d(HW_labels_all1 ,label_name [2])]
    HW_strd_all2=np.array(HW_SEB_all_re)[:,3][np.in1d(HW_labels_all1 ,label_name [2])]

    HW_ssr_all1_all.append(np.nanmean(HW_ssr_all1))
    HW_slfh_all1_all.append(np.nanmean(HW_slfh_all1))
    HW_adv_all1_all.append(np.nanmean(HW_adv_all1))
    HW_strd_all1_all.append(np.nanmean(HW_strd_all1))

    HW_ssr_all2_all.append(np.nanmean(HW_ssr_all2))
    HW_slfh_all2_all.append(np.nanmean(HW_slfh_all2))
    HW_adv_all2_all.append(np.nanmean(HW_adv_all2))
    HW_strd_all2_all.append(np.nanmean(HW_strd_all2))
    # fig= plt.figure(figsize=(2.5, 2))
    # #fig,axs = plt.subplots(figsize=(4, 2),ncols=4,nrows=1)
    # ax = plt.axes([0.3, 0.2, 0.65, 0.75])

    # all_data = []
    # for box_temp,pos in zip([[HW_ssr_all1,HW_ssr_all2],[HW_slfh_all1,HW_slfh_all2+0.05],[HW_adv_all1,HW_adv_all2],[HW_strd_all1,HW_strd_all2+0.05]],range(4)):
    #     #ax=axs[pos]
    #     box_temp1 = box_temp[0]
    #     bplot = ax.boxplot(box_temp1,widths=0.3, positions=[pos+1-0.2],whis = [10, 90],showfliers=False)
    #     for element in ['whiskers','caps','medians','boxes']:
    #         for patch in bplot[element] :
    #             plt.setp(patch, color='blue')
    #     box_temp1 = box_temp[1]
    #     bplot = ax.boxplot(box_temp1,widths=0.3, positions=[pos+1+0.2],whis = [10, 90],showfliers=False)
    #     for element in ['whiskers','caps','medians','boxes']:
    #         for patch in bplot[element] :
    #             plt.setp(patch, color='red')
    # #ax.axhline(y = 0, color = 'grey', linestyle = '--')
    # ax.axvline(x = 2.5, color = 'k', linestyle = '-')
    # ax.set_ylabel('Contribution (%)',labelpad=-0.01,fontsize=9) 
    # for bnd in ['bottom','top','right','left']:
    #     ax.spines[bnd].set_color('k')
    # ax.grid(b=False)

    # ax.set_xticks([1,2,3,4])
    # ax.set_xticklabels(['$R_{s}$','-LE','ADV','$R_{ld}$'],fontsize=10)

    # plt.subplots_adjust(wspace=0.6,bottom=0.25,left=0.1,right=0.9)
    #fig.savefig(r'/Net/Groups/BGI/scratch/yt/figure/Heat_wave_3D_99th_connect6/cmip6_com_'+modelname[:-1]+'.jpg', dpi=300)


delta_strd_po = []
delta_strd_ne = []
delta_adv_po = []
delta_adv_ne = []

delta_ADVcorr_po = []
delta_ADVcorr_ne = []
delta_ADVslope_po = []
delta_ADVslope_ne = []

for i_model,modelname in enumerate(['CanESM5/','INM-CM4-8/','INM-CM5-0/','MPI-ESM1-2-LR/','MRI-ESM2-0/','CMCC-ESM2/','IPSL-CM6A-LR/']):
    per_strd1 = (1-HW_strd_all1_all[i_model])*100
    per_strd2 = (1-HW_strd_all2_all[i_model]    )*100
    per_adv1 = (1-HW_adv_all1_all[i_model])*100
    per_adv2 = (1-HW_adv_all2_all[i_model] )*100

    senarioname = 'ssp585'

    print(modelname)

    yearstart=2015
    yearend=2060
    yearrange = np.arange(yearstart,yearend+1)

    lon_unin = np.arange(-180,180,2.5)
    lat_unin = np.arange(-90,90,2.5)[::-1]
    LON_unin, LAT_unin = np.meshgrid(lon_unin, lat_unin)
    ##
    NCname = '/Net/Groups/BGI/scratch/yt/data/era5_daily_1x1/land_mask.nc'
    NCData = Dataset(NCname)
    land_mask = np.squeeze(NCData.variables['lsm'][:])
    NCData.close()
    NCname = r'/Net/Groups/BGI/scratch/yt/data/era5_daily_1x1/t2m_1979_2020_dailymax_1x1_delFeb29.nc'
    NCData = Dataset(NCname)
    lon_era5 = NCData.variables['lon'][:]
    lat_era5 = NCData.variables['lat'][:]
    LON_era5, LAT_era5 = np.meshgrid(lon_era5, lat_era5)
    NCData.close()
    land_mask = np.concatenate((land_mask[:,180:] ,land_mask[:,:180]),axis=1 )
    lon_era5 = np.concatenate((lon_era5[180:]-360 ,lon_era5[:180]) )
    LON_era5 = np.concatenate((LON_era5[:,180:]-360 ,LON_era5[:,:180]),axis=1 )
    ##
    LAT_unin1 = copy.deepcopy(LAT_unin)
    LON_unin1 = copy.deepcopy(LON_unin)
    LAT_unin1[LAT_unin1>np.nanmax(lat_era5)] = np.nanmax(lat_era5)
    LAT_unin1[LAT_unin1<np.nanmin(lat_era5)] = np.nanmin(lat_era5)
    LON_unin1[LON_unin1>np.nanmax(lon_era5)] = np.nanmax(lon_era5)
    LON_unin1[LON_unin1<np.nanmin(lon_era5)] = np.nanmin(lon_era5)

    my_interpolating_function = RegularGridInterpolator((lat_era5,lon_era5), land_mask)
    land_mask =  my_interpolating_function((LAT_unin1.ravel(), LON_unin1.ravel())).reshape((len(LAT_unin1[:,0]),len(LON_unin1[0])))

    #%% file info
    filelist_cmip6 = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname)
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[:6]=='tasmax'])
    temp = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filelist_cmip6[0]
    grid_label = temp[temp.find('_g')+1:temp.find('_g')+1+temp[temp.find('_g')+1:].find('_')]
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[-31-len(grid_label)-len(senarioname):-31-len(grid_label)]==senarioname])
    filelist_cmip6_2 = np.array([int(i[-11:-7]) for i in filelist_cmip6])
    filelist_cmip6_1 = np.array([int(i[-20:-16]) for i in filelist_cmip6])
    filelist_cmip6 = np.array(filelist_cmip6)[~((filelist_cmip6_2<yearstart)|(filelist_cmip6_1>yearend))]   
    filelist_cmip6 = np.array(filelist_cmip6)[np.argsort(np.array([int(i[-20:-16]) for i in filelist_cmip6]))]


    #%%%%%%%%%%%%%%%%%%%%%%% spatial info
    filename = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filelist_cmip6[0]
    NCData = Dataset( filename)
    lon_cmip6 = NCData.variables['lon'][:]
    lat_cmip6 = NCData.variables['lat'][:][::-1]
    LON_cmip6, LAT_cmip6 = np.meshgrid(lon_cmip6, lat_cmip6)
    print(lon_cmip6[:10])
    print(lat_cmip6[:10])
    temp = int(len(lon_cmip6)*0.5)
    lon_cmip6 = np.concatenate((lon_cmip6[temp:]-360 ,lon_cmip6[:temp]) )
    LON_cmip6 = np.concatenate((LON_cmip6[:,temp:]-360 ,LON_cmip6[:,:temp]),axis=1 )
    LAT_unin2 = copy.deepcopy(LAT_unin)
    LON_unin2 = copy.deepcopy(LON_unin)
    LAT_unin2[LAT_unin2>np.nanmax(lat_cmip6)] = np.nanmax(lat_cmip6)
    LAT_unin2[LAT_unin2<np.nanmin(lat_cmip6)] = np.nanmin(lat_cmip6)
    LON_unin2[LON_unin2>np.nanmax(lon_cmip6)] = np.nanmax(lon_cmip6)
    LON_unin2[LON_unin2<np.nanmin(lon_cmip6)] = np.nanmin(lon_cmip6)

    #%%%%%%%%%%%%%%%%%%%%%%% temporal info
    ######time
    year_cmip6_all = []
    mon_cmip6_all = []
    day_cmip6_all = []
    for filename in filelist_cmip6:
        
        filename = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filename
        NCData = Dataset( filename)
        # lon_cmip6 = NCData.variables['lon'][:]
        # lat_cmip6 = NCData.variables['lat'][:][::-1]
        # LON_cmip6, LAT_cmip6 = np.meshgrid(lon_cmip6, lat_cmip6)
        time = NCData.variables['time']
        dates = list(num2date(time[:], time.units, time.calendar))
        year = np.array([date.year for date in dates])
        year_where = np.where(np.in1d(year,yearrange))[0]
        if len(year_where)==0:
            continue
        print (filename)
        year_temp = np.array([date.year for date in dates])[year_where ]
        mon_temp = np.array([date.month for date in dates])[year_where ]
        day_temp = np.array([date.day for date in dates])[year_where ]

        tmax_temp = NCData.variables['tasmax'][year_where][:,::-1,:]

        year_cmip6_all.extend(year_temp)
        mon_cmip6_all.extend(mon_temp)
        day_cmip6_all.extend(day_temp)
    md_cmip6_all = 100*np.array(mon_cmip6_all)+np.array(day_cmip6_all)
    year_cmip6_all = np.squeeze(np.array(year_cmip6_all)[md_cmip6_all!=229])
    mon_cmip6_all = np.squeeze(np.array(mon_cmip6_all)[md_cmip6_all!=229])
    day_cmip6_all = np.squeeze(np.array(day_cmip6_all)[md_cmip6_all!=229])

    md_cmip6_all = np.squeeze(md_cmip6_all )
    print(len(md_cmip6_all))
    #%%
    strd1 = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'strd'+'_ano_'+str(yearstart)+'_'+str(yearend)+'_dailymean_2.5x2.5_del29Feb.nc') ['__xarray_dataarray_variable__'][:][:])# HW_vrh_ano = get_HW_ano_mean(vrh_ano, HW_date,HW_lat,HW_lon)
    adv1 = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'adv'+'_ano_'+str(yearstart)+'_'+str(yearend)+'_dailymean_2.5x2.5_del29Feb.nc') ['__xarray_dataarray_variable__'][:][:])# HW_vrh_ano = get_HW_ano_mean(vrh_ano, HW_date,HW_lat,HW_lon)

    strd1_sum = np.zeros((strd1.shape[1],strd1.shape[2]))
    strd1_sum[lat_unin>23.5,:] = np.nanpercentile(strd1[np.in1d(mon_cmip6_all,[6,7,8])],per_strd1,axis=0)[lat_unin>23.5,:]
    strd1_sum[np.where(lat_unin<-23.5)[0],:] = np.nanpercentile(strd1[np.in1d(mon_cmip6_all,[12,1,2])],per_strd1,axis=0)[lat_unin<-23.5,:]
    strd1_sum[np.where((lat_unin>-23.5)&(lat_unin<23.5))[0],:] = np.nanpercentile(strd1,per_strd1,axis=0)[(lat_unin>-23.5)&(lat_unin<23.5)]

    adv1_sum = np.zeros((adv1.shape[1],adv1.shape[2]))
    adv1_sum[lat_unin>23.5,:] = np.nanpercentile(adv1[np.in1d(mon_cmip6_all,[6,7,8])],per_adv1,axis=0)[lat_unin>23.5,:]
    adv1_sum[np.where(lat_unin<-23.5)[0],:] = np.nanpercentile(adv1[np.in1d(mon_cmip6_all,[12,1,2])],per_adv1,axis=0)[lat_unin<-23.5,:]
    adv1_sum[np.where((lat_unin>-23.5)&(lat_unin<23.5))[0],:] = np.nanpercentile(adv1,per_adv1,axis=0)[(lat_unin>-23.5)&(lat_unin<23.5)]

    ADV_corr1 = np.load( r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'Corr_strd_adv_'+str(yearstart)+'_'+str(yearend) ,allow_pickle=True)
    ADV_corr1 = np.array([np.abs(i) for i in ADV_corr1.ravel()]).reshape((ADV_corr1.shape[0],ADV_corr1.shape[1]))
    ADV_slope1 = np.load( r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'slope_strd_adv_'+str(yearstart)+'_'+str(yearend) ,allow_pickle=True)

    yearstart=2055
    yearend=2100
    yearrange = np.arange(yearstart,yearend+1)

    lon_unin = np.arange(-180,180,2.5)
    lat_unin = np.arange(-90,90,2.5)[::-1]
    LON_unin, LAT_unin = np.meshgrid(lon_unin, lat_unin)
    ##
    NCname = '/Net/Groups/BGI/scratch/yt/data/era5_daily_1x1/land_mask.nc'
    NCData = Dataset(NCname)
    land_mask = np.squeeze(NCData.variables['lsm'][:])
    NCData.close()
    NCname = r'/Net/Groups/BGI/scratch/yt/data/era5_daily_1x1/t2m_1979_2020_dailymax_1x1_delFeb29.nc'
    NCData = Dataset(NCname)
    lon_era5 = NCData.variables['lon'][:]
    lat_era5 = NCData.variables['lat'][:]
    LON_era5, LAT_era5 = np.meshgrid(lon_era5, lat_era5)
    NCData.close()
    land_mask = np.concatenate((land_mask[:,180:] ,land_mask[:,:180]),axis=1 )
    lon_era5 = np.concatenate((lon_era5[180:]-360 ,lon_era5[:180]) )
    LON_era5 = np.concatenate((LON_era5[:,180:]-360 ,LON_era5[:,:180]),axis=1 )
    ##
    LAT_unin1 = copy.deepcopy(LAT_unin)
    LON_unin1 = copy.deepcopy(LON_unin)
    LAT_unin1[LAT_unin1>np.nanmax(lat_era5)] = np.nanmax(lat_era5)
    LAT_unin1[LAT_unin1<np.nanmin(lat_era5)] = np.nanmin(lat_era5)
    LON_unin1[LON_unin1>np.nanmax(lon_era5)] = np.nanmax(lon_era5)
    LON_unin1[LON_unin1<np.nanmin(lon_era5)] = np.nanmin(lon_era5)

    my_interpolating_function = RegularGridInterpolator((lat_era5,lon_era5), land_mask)
    land_mask =  my_interpolating_function((LAT_unin1.ravel(), LON_unin1.ravel())).reshape((len(LAT_unin1[:,0]),len(LON_unin1[0])))

    #%% file info
    filelist_cmip6 = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname)
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[:6]=='tasmax'])
    temp = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filelist_cmip6[0]
    grid_label = temp[temp.find('_g')+1:temp.find('_g')+1+temp[temp.find('_g')+1:].find('_')]
    filelist_cmip6 = np.array([i for i in filelist_cmip6 if i[-31-len(grid_label)-len(senarioname):-31-len(grid_label)]==senarioname])
    filelist_cmip6_2 = np.array([int(i[-11:-7]) for i in filelist_cmip6])
    filelist_cmip6_1 = np.array([int(i[-20:-16]) for i in filelist_cmip6])
    filelist_cmip6 = np.array(filelist_cmip6)[~((filelist_cmip6_2<yearstart)|(filelist_cmip6_1>yearend))]   
    filelist_cmip6 = np.array(filelist_cmip6)[np.argsort(np.array([int(i[-20:-16]) for i in filelist_cmip6]))]


    #%%%%%%%%%%%%%%%%%%%%%%% spatial info
    filename = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filelist_cmip6[0]
    NCData = Dataset( filename)
    lon_cmip6 = NCData.variables['lon'][:]
    lat_cmip6 = NCData.variables['lat'][:][::-1]
    LON_cmip6, LAT_cmip6 = np.meshgrid(lon_cmip6, lat_cmip6)
    print(lon_cmip6[:10])
    print(lat_cmip6[:10])
    temp = int(len(lon_cmip6)*0.5)
    lon_cmip6 = np.concatenate((lon_cmip6[temp:]-360 ,lon_cmip6[:temp]) )
    LON_cmip6 = np.concatenate((LON_cmip6[:,temp:]-360 ,LON_cmip6[:,:temp]),axis=1 )
    LAT_unin2 = copy.deepcopy(LAT_unin)
    LON_unin2 = copy.deepcopy(LON_unin)
    LAT_unin2[LAT_unin2>np.nanmax(lat_cmip6)] = np.nanmax(lat_cmip6)
    LAT_unin2[LAT_unin2<np.nanmin(lat_cmip6)] = np.nanmin(lat_cmip6)
    LON_unin2[LON_unin2>np.nanmax(lon_cmip6)] = np.nanmax(lon_cmip6)
    LON_unin2[LON_unin2<np.nanmin(lon_cmip6)] = np.nanmin(lon_cmip6)

    #%%%%%%%%%%%%%%%%%%%%%%% temporal info
    ######time
    year_cmip6_all = []
    mon_cmip6_all = []
    day_cmip6_all = []
    for filename in filelist_cmip6:
        
        filename = r'/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+filename
        NCData = Dataset( filename)
        # lon_cmip6 = NCData.variables['lon'][:]
        # lat_cmip6 = NCData.variables['lat'][:][::-1]
        # LON_cmip6, LAT_cmip6 = np.meshgrid(lon_cmip6, lat_cmip6)
        time = NCData.variables['time']
        dates = list(num2date(time[:], time.units, time.calendar))
        year = np.array([date.year for date in dates])
        year_where = np.where(np.in1d(year,yearrange))[0]
        if len(year_where)==0:
            continue
        print (filename)
        year_temp = np.array([date.year for date in dates])[year_where ]
        mon_temp = np.array([date.month for date in dates])[year_where ]
        day_temp = np.array([date.day for date in dates])[year_where ]

        tmax_temp = NCData.variables['tasmax'][year_where][:,::-1,:]

        year_cmip6_all.extend(year_temp)
        mon_cmip6_all.extend(mon_temp)
        day_cmip6_all.extend(day_temp)
    md_cmip6_all = 100*np.array(mon_cmip6_all)+np.array(day_cmip6_all)
    year_cmip6_all = np.squeeze(np.array(year_cmip6_all)[md_cmip6_all!=229])
    mon_cmip6_all = np.squeeze(np.array(mon_cmip6_all)[md_cmip6_all!=229])
    day_cmip6_all = np.squeeze(np.array(day_cmip6_all)[md_cmip6_all!=229])

    md_cmip6_all = np.squeeze(md_cmip6_all )
    print(len(md_cmip6_all))
    #%%
    strd2 = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'strd'+'_ano_'+str(yearstart)+'_'+str(yearend)+'_dailymean_2.5x2.5_del29Feb.nc') ['__xarray_dataarray_variable__'][:][:])# HW_vrh_ano = get_HW_ano_mean(vrh_ano, HW_date,HW_lat,HW_lon)
    adv2 = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'adv'+'_ano_'+str(yearstart)+'_'+str(yearend)+'_dailymean_2.5x2.5_del29Feb.nc') ['__xarray_dataarray_variable__'][:][:])# HW_vrh_ano = get_HW_ano_mean(vrh_ano, HW_date,HW_lat,HW_lon)

    strd2_sum = np.zeros((strd2.shape[1],strd2.shape[2]))
    strd2_sum[lat_unin>23.5,:] = np.nanpercentile(strd2[np.in1d(mon_cmip6_all,[6,7,8])],per_strd2,axis=0)[lat_unin>23.5,:]
    strd2_sum[np.where(lat_unin<-23.5)[0],:] = np.nanpercentile(strd2[np.in1d(mon_cmip6_all,[12,1,2])],per_strd2,axis=0)[lat_unin<-23.5,:]
    strd2_sum[np.where((lat_unin>-23.5)&(lat_unin<23.5))[0],:] = np.nanpercentile(strd2,per_strd2,axis=0)[(lat_unin>-23.5)&(lat_unin<23.5)]

    adv2_sum = np.zeros((adv2.shape[1],adv2.shape[2]))
    adv2_sum[lat_unin>23.5,:] = np.nanpercentile(adv2[np.in1d(mon_cmip6_all,[6,7,8])],per_adv2,axis=0)[lat_unin>23.5,:]
    adv2_sum[np.where(lat_unin<-23.5)[0],:] = np.nanpercentile(adv2[np.in1d(mon_cmip6_all,[12,1,2])],per_adv2,axis=0)[lat_unin<-23.5,:]
    adv2_sum[np.where((lat_unin>-23.5)&(lat_unin<23.5))[0],:] = np.nanpercentile(adv2,per_adv2,axis=0)[(lat_unin>-23.5)&(lat_unin<23.5)]
    

    
    ADV_corr2 = np.load( r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'Corr_strd_adv_'+str(yearstart)+'_'+str(yearend) ,allow_pickle=True)
    ADV_corr2 = np.array([np.abs(i) for i in ADV_corr2.ravel()]).reshape((ADV_corr2.shape[0],ADV_corr2.shape[1]))
    ADV_slope2 = np.load( r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'slope_strd_adv_'+str(yearstart)+'_'+str(yearend) ,allow_pickle=True)


    vername = '_235_'+str(2015)+'_'+str(2060)
    land_map1 = np.load(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'label_map_o'+vername,allow_pickle=True)[2]
    vername = '_235_'+str(2055)+'_'+str(2100)
    land_map2 = np.load(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_cmip6/'+modelname+'label_map_o'+vername,allow_pickle=True)[2]
    land_map = land_map2-land_map1

    delta_strd = strd2_sum-strd1_sum
    delta_adv = (adv2_sum-adv1_sum)
    delta_advcorr =  ADV_corr2 -ADV_corr1
    delta_advslope =  ADV_slope2 -ADV_slope1




    M_temp = np.where(land_map2[(land_map2>0)|(land_map1>0)])[0]
    n_all = len(M_temp)
    

    n_out = len(np.where(land_map[land_map>0])[0])
    if n_out >n_all:
        n_out=int(0.6*n_all)
    n_trial = 1000
    print(n_all )
    print(len(np.where(land_map[land_map>0])[0]) )

    indices_random = [] # .astype('bool')
    #Generate 5 random numbers between 10 and 30
    for i in range(n_trial):
            #print(i_label,i)
        randomlist = random.sample(range(n_all), n_out)
        indices_random.append(np.array([int(M_temp[j] )for j in randomlist]))

    delta_strd_po.append(np.array([np.nanmean(delta_strd.ravel()[iii]) for iii in indices_random]))
    delta_adv_po.append(np.array([np.nanmean(delta_adv.ravel()[iii]) for iii in indices_random]))
    delta_ADVcorr_po.append(np.array([np.nanmean(delta_advcorr.ravel()[iii]) for iii in indices_random]))
    delta_ADVslope_po.append(np.array([np.nanmean(delta_advslope.ravel()[iii]) for iii in indices_random]))



    n_out = len(np.where(land_map[land_map<0])[0])
    if n_out >n_all:
        n_out=int(0.6*n_all)
    n_trial = 1000
    print(n_all )
    print(len(np.where(land_map[land_map<0])[0]) )

    indices_random = [] # .astype('bool')
    #Generate 5 random numbers between 10 and 30
    for i in range(n_trial):
            #print(i_label,i)
        randomlist = random.sample(range(n_all), n_out)
        indices_random.append(np.array([int(M_temp[j] )for j in randomlist]))

    delta_strd_ne.append(np.array([np.nanmean(delta_strd.ravel()[iii]) for iii in indices_random]))
    delta_adv_ne.append(np.array([np.nanmean(delta_adv.ravel()[iii]) for iii in indices_random]))
    delta_ADVcorr_ne.append(np.array([np.nanmean(delta_advcorr.ravel()[iii]) for iii in indices_random]))
    delta_ADVslope_ne.append(np.array([np.nanmean(delta_advslope.ravel()[iii]) for iii in indices_random]))




    temp_xr = xr.DataArray(np.array(delta_strd_po))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_strd_po.nc')

    temp_xr = xr.DataArray(np.array(delta_adv_po))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_adv_po.nc')

    temp_xr = xr.DataArray(np.array(delta_ADVcorr_po))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_ADVcorr_po.nc')

    temp_xr = xr.DataArray(np.array(delta_ADVslope_po))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_ADVslope_po.nc')


    temp_xr = xr.DataArray(np.array(delta_strd_ne))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_strd_ne.nc')

    temp_xr = xr.DataArray(np.array(delta_adv_ne))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_adv_ne.nc')

    temp_xr = xr.DataArray(np.array(delta_ADVcorr_ne))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_ADVcorr_ne.nc')

    temp_xr = xr.DataArray(np.array(delta_ADVslope_ne))
    temp_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/Cmip6_ssp585_wget/'+modelname+'sig_'+'_delta_ADVslope_ne.nc')
