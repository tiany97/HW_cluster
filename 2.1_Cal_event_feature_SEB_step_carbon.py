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


from matplotlib import colors as c
import csv
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm

import cc3d
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl
import matplotlib.cm as cm

from scipy import signal
#%%
NCname = '/Net/Groups/data_BGC/era5/e1/0d25_daily/t2mmax/t2mmax.daily.an.era5.1440.720.1950.nc'
NCData = Dataset(NCname)
lon_era5 = NCData.variables['longitude'][:]
lat_era5 = NCData.variables['latitude'][:]
LON_era5, LAT_era5 = np.meshgrid(lon_era5, lat_era5)
# var = NCData.variables['t2mmax'][0]
NCData.close()
NCname = '/Net/Groups/BGI/scratch/yt/data/era5_land_surface_mask.nc'
NCData = Dataset(NCname)
land_mask = np.squeeze(NCData.variables['lsm'][:])
lon_era50 = NCData.variables['longitude'][:]
lat_era50 = NCData.variables['latitude'][:]
NCData.close()
land_mask = np.concatenate((land_mask[:,720:] ,land_mask[:,:720]),axis=1 )
lon_era50 = np.concatenate((lon_era50[720:]-360 ,lon_era50[:720]) )
LAT_era5[LAT_era5>np.nanmax(lat_era50)] = np.nanmax(lat_era50)
LAT_era5[LAT_era5<np.nanmin(lat_era50)] = np.nanmin(lat_era50)
LON_era5[LON_era5>np.nanmax(lon_era50)] = np.nanmax(lon_era50)
LON_era5[LON_era5<np.nanmin(lon_era50)] = np.nanmin(lon_era50)

my_interpolating_function = RegularGridInterpolator((lat_era50[::-1],lon_era50), land_mask[::-1])
land_mask  = my_interpolating_function((LAT_era5.ravel(), LON_era5.ravel())).reshape((len(LAT_era5[:,0]),len(LON_era5[0])))


year_era5_all = []
mon_era5_all = []
day_era5_all = []
for i_year in range(1979,2021):
    NCname = '/Net/Groups/data_BGC/era5/e1/0d25_daily/t2mmax/t2mmax.daily.an.era5.1440.720.'+str(i_year)+'.nc'
    NCData = Dataset(NCname)
    time = NCData.variables['time']
    dates = list(num2date(time[:], time.units, time.calendar))
    lon_era5 = NCData.variables['longitude'][:]
    lat_era5 = NCData.variables['latitude'][:]
    LON_era5, LAT_era5 = np.meshgrid(lon_era5, lat_era5)
    NCData.close()

    year_era5 = np.array([date.year for date in dates])
    mon_era5 = np.array([date.month for date in dates])
    day_era5 = np.array([date.day for date in dates])
    md_era5 = 100*np.array(mon_era5)+np.array(day_era5)

    year_era5 = np.squeeze(np.array(year_era5)[md_era5!=229])
    mon_era5 = np.squeeze(np.array(mon_era5)[md_era5!=229])
    day_era5= np.squeeze(np.array(day_era5)[md_era5!=229])

    year_era5_all.extend(year_era5)
    mon_era5_all.extend(mon_era5)
    day_era5_all.extend(day_era5)

year_era5_all = np.array(year_era5_all )
mon_era5_all = np.array(mon_era5_all )
day_era5_all = np.array(day_era5_all )
print(lon_era5[:10])
print(len(year_era5_all))
#%%
print('hello from YT')
#%%
def remove_seasonal(var):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(42)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        temp = signal.detrend(temp,axis=0)
        for i_year in range(42):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend 

#%%
def remove_seasonal_normalized(var):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(42)])
        temp = (temp - np.array(len(temp)*[np.nanmean(temp,axis=0)]))/np.array(len(temp)*[np.nanstd(temp,axis=0)])
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(42):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend 
#%%
def SEB_ano(var,skt):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(42)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        temp1 =  np.array([skt[365*i_year+i_box] for i_year in range(42)])
        temp1 = np.array(len(temp1)*[np.nanmean(temp1,axis=0)])
        temp = temp/(4*5.68*1e-8*temp1**3)
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(42):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend


#%%

filelist = os.listdir(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/')
leastdate = 5

filelist = os.listdir(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree')
HW_date = []
HW_lat = []
HW_lon = []
HW_location_lat = []
HW_location_lon = []
HW_duration = []
HW_aera = []
HW_date_centroid = []
HW_location_centroid = []
for fl in filelist:
    if fl[-4:] != '.csv':
        continue
    flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/' + fl
    with open(flname, 'r') as file:
        reader = csv.reader(file)
        all = []
        for i_row,row in enumerate(reader):
            if i_row in [6,7,8]:
                all.append(np.array([ round(float(i)) for i in row]))
            else:
                all.append(np.array([ int(i) for i in row]))
    if all[5]<leastdate:
        continue
    if all[6]<40000:
        continue
    if   lat_era5[int(all[8][0])]<-23.5:
        if mon_era5_all[int(all[7])] not in [12,1,2]:
            continue
    if   lat_era5[int(all[8][0])]>=23.5:
        if mon_era5_all[int(all[7])] not in [6,7,8]:
            continue
    HW_date.append(all[0])
    HW_lat.append(all[1])
    HW_lon.append(all[2])
    HW_location_lat.append(all[3])
    HW_location_lon.append(all[4])   
    HW_duration.append(all[5])
    HW_aera.append(all[6])
    HW_date_centroid.append(all[7])
    HW_location_centroid.append(all[8])
#%%
HW_duration = np.squeeze(np.array(HW_duration))
HW_aera = np.squeeze(np.array(HW_aera))
HW_date_centroid = np.squeeze(np.array(HW_date_centroid))
HW_location_centroid = np.squeeze(np.array( HW_location_centroid ))
#%%
event_N = len(HW_duration)
print( 'extreme event n = ' +str(event_N))
#%%

vername = '_235'
def get_HW_ano_all(var,variable, HW_date,HW_lat,HW_lon):
    vername = '_235'
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        HW_ano.append(variable[i_date,i_lat,i_lon])
    with open(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/HW_'+var+'_ano_6days_all'+vername+'_detrend.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(HW_ano)
        print(var+'end')
        #%%
# print(event_N )
#%%
def remove_seasonal_modis(var,n_step,n_year):
    var_detrend = np.zeros_like(var)
    for i_box in range(n_step):
        temp = np.array([var[n_step*i_year+i_box] for i_year in range(n_year)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        temp[~np.isnan(temp)] = signal.detrend(temp[~np.isnan(temp)],axis=0)
        for i_year in range(n_year):
            var_detrend[n_step*i_year+i_box]  = temp[i_year]
    return var_detrend 
#%%
vername = '_235'
def get_HW_ano_mean_modis(variable, HW_date,HW_lat,HW_lon):
    vername = '_235'
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        if (np.array(i_date)<0).any() or (np.array(i_date)>=len(variable)).any():
            HW_ano.append(np.nan)
        else:
            HW_ano.append(np.nanmean((variable[i_date,i_lat,i_lon])))
    return np.array(HW_ano)
def get_HW_ano_mean_modis_all(var,variable, HW_date,HW_lat,HW_lon):
    vername = '_235'
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        if (np.array(i_date)<0).any() or (np.array(i_date)>=len(variable)).any():
            HW_ano.append([np.nan])
        else:
            HW_ano.append((variable[i_date,i_lat,i_lon]))
    with open(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/HW_'+var+'_ano_6days_all'+vername+'_detrend.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(HW_ano)
        print(var+'end')

#%%



#%%
HW_year = np.array([year_era5_all[i] for i in HW_date])
HW_mon = np.array([mon_era5_all[i] for i in HW_date])
#%%
filelist_FLUXCOM = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/')
filelist_FLUXCOM = np.array([i for i in filelist_FLUXCOM if i[:3]=='NEE'])
filelist_FLUXCOM1 = [np.int(i[-7:-3]) for i in filelist_FLUXCOM]
filelist_FLUXCOM = filelist_FLUXCOM[np.argsort(filelist_FLUXCOM1)]

filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+filelist_FLUXCOM[0]
NCData = Dataset( filename)
lon_FLUXCOM = NCData.variables['lon'][:]
lat_FLUXCOM = NCData.variables['lat'][:]
LON_FLUXCOM,LAT_FLUXCOM = np.meshgrid(lon_FLUXCOM,lat_FLUXCOM)

lat_FLUXCOM[0]=90
lat_FLUXCOM[-1]=-90
lon_FLUXCOM[0]=-180
lon_FLUXCOM[-1]=180
#%%
FLUXCOM_year = []
FLUXCOM_mon = []
FLUXCOM_NEE = []
for fl in filelist_FLUXCOM:
    filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+fl
    NCData = Dataset( filename)
    data = NCData.variables['NEE'][:]
    temp = data.data
    temp[data.mask]=np.nan
    temp1 = []

    for i_temp in temp:
        # my_interpolating_function = RegularGridInterpolator(
        # (lat_FLUXCOM[::-1],lon_FLUXCOM), i_temp)
        # temp1.append( my_interpolating_function((LAT_era5[::-1,:].ravel(), LON_era5.ravel())).reshape((len(LAT_era5[:,0]),len(LON_era5[0]))))

        temp_new=np.zeros((720,1440))
        temp_new[::2,::2]=i_temp[:,:]
        temp_new[::2,1::2]=i_temp[:,:]
        temp_new[1::2,::2]=i_temp[:,:]
        temp_new[1::2,1::2]=i_temp[:,:]
        temp1.append(temp_new)
 
        
        FLUXCOM_year.append(np.int(fl[-7:-3]))
    FLUXCOM_mon.extend(np.arange(1,13))
    FLUXCOM_NEE.extend(temp1)

FLUXCOM_year = np.array(FLUXCOM_year)
FLUXCOM_mon = np.array(FLUXCOM_mon)
FLUXCOM_NEE = np.array(FLUXCOM_NEE)
print('read NEE')
    #%%
HW_ym = []
for i_year,i_mon in zip(HW_year ,HW_mon):
    i_ym = (i_year-FLUXCOM_year[0])*12+i_mon
    HW_ym.append(i_ym)
HW_ym = np.array(HW_ym)
NEE_ano = remove_seasonal_modis(FLUXCOM_NEE,12,int(len(FLUXCOM_year)/12))

get_HW_ano_mean_modis_all('NEE',NEE_ano, HW_ym,HW_lat,HW_lon)
print('cal NEE')



#%%
filelist_FLUXCOM = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/')
filelist_FLUXCOM = np.array([i for i in filelist_FLUXCOM if i[:3]=='GPP'])
filelist_FLUXCOM1 = [np.int(i[-7:-3]) for i in filelist_FLUXCOM]
filelist_FLUXCOM = filelist_FLUXCOM[np.argsort(filelist_FLUXCOM1)]

filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+filelist_FLUXCOM[0]
NCData = Dataset( filename)
lon_FLUXCOM = NCData.variables['lon'][:]
lat_FLUXCOM = NCData.variables['lat'][:]
LON_FLUXCOM,LAT_FLUXCOM = np.meshgrid(lon_FLUXCOM,lat_FLUXCOM)

lat_FLUXCOM[0]=90
lat_FLUXCOM[-1]=-90
lon_FLUXCOM[0]=-180
lon_FLUXCOM[-1]=180
FLUXCOM_year = []
FLUXCOM_mon = []
FLUXCOM_GPP = []
for fl in filelist_FLUXCOM:
    filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+fl
    NCData = Dataset( filename)
    data = NCData.variables['GPP'][:]
    temp = data.data
    temp[data.mask]=np.nan
    temp1 = []

    for i_temp in temp:
        # my_interpolating_function = RegularGridInterpolator(
        # (lat_FLUXCOM[::-1],lon_FLUXCOM), i_temp)
        # temp1.append( my_interpolating_function((LAT_era5[::-1,:].ravel(), LON_era5.ravel())).reshape((len(LAT_era5[:,0]),len(LON_era5[0]))))

    
        temp_new=np.zeros((720,1440))
        temp_new[::2,::2]=i_temp[:,:]
        temp_new[::2,1::2]=i_temp[:,:]
        temp_new[1::2,::2]=i_temp[:,:]
        temp_new[1::2,1::2]=i_temp[:,:]
        temp1.append( temp_new)
        FLUXCOM_year.append(np.int(fl[-7:-3]))
    FLUXCOM_mon.extend(np.arange(1,13))
    FLUXCOM_GPP.extend(temp1)

FLUXCOM_year = np.array(FLUXCOM_year)
FLUXCOM_mon = np.array(FLUXCOM_mon)
FLUXCOM_GPP = np.array(FLUXCOM_GPP)
print('read GPP')
    #%%
HW_ym = []
for i_year,i_mon in zip(HW_year ,HW_mon):
    i_ym = (i_year-FLUXCOM_year[0])*12+i_mon
    HW_ym.append(i_ym)
HW_ym = np.array(HW_ym)
GPP_ano = remove_seasonal_modis(FLUXCOM_GPP,12,int(len(FLUXCOM_year)/12))

get_HW_ano_mean_modis_all('GPP',GPP_ano, HW_ym,HW_lat,HW_lon)
print('cal GPP')
#%%






#%%
#%%
filelist_FLUXCOM = os.listdir(r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/')
filelist_FLUXCOM = np.array([i for i in filelist_FLUXCOM if i[:3]=='TER'])
filelist_FLUXCOM1 = [np.int(i[-7:-3]) for i in filelist_FLUXCOM]
filelist_FLUXCOM = filelist_FLUXCOM[np.argsort(filelist_FLUXCOM1)]

filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+filelist_FLUXCOM[0]
NCData = Dataset( filename)
lon_FLUXCOM = NCData.variables['lon'][:]
lat_FLUXCOM = NCData.variables['lat'][:]
LON_FLUXCOM,LAT_FLUXCOM = np.meshgrid(lon_FLUXCOM,lat_FLUXCOM)

lat_FLUXCOM[0]=90
lat_FLUXCOM[-1]=-90
lon_FLUXCOM[0]=-180
lon_FLUXCOM[-1]=180
FLUXCOM_year = []
FLUXCOM_mon = []
FLUXCOM_TER = []
for fl in filelist_FLUXCOM:
    filename = r'/Net/Groups/BGI/scratch/yt/data/FLUXCOM/RS_METEO/ensemble/ERA5/monthly/'+fl
    NCData = Dataset( filename)
    data = NCData.variables['TER'][:]
    temp = data.data
    temp[data.mask]=np.nan
    temp1 = []

    for i_temp in temp:
        # my_interpolating_function = RegularGridInterpolator(
        # (lat_FLUXCOM[::-1],lon_FLUXCOM), i_temp)
        # temp1.append( my_interpolating_function((LAT_era5[::-1,:].ravel(), LON_era5.ravel())).reshape((len(LAT_era5[:,0]),len(LON_era5[0]))))

        temp_new=np.zeros((720,1440))
        temp_new[::2,::2]=i_temp[:,:]
        temp_new[::2,1::2]=i_temp[:,:]
        temp_new[1::2,::2]=i_temp[:,:]
        temp_new[1::2,1::2]=i_temp[:,:]
        temp1.append( temp_new)
        FLUXCOM_year.append(np.int(fl[-7:-3]))
    FLUXCOM_mon.extend(np.arange(1,13))
    FLUXCOM_TER.extend(temp1)

FLUXCOM_year = np.array(FLUXCOM_year)
FLUXCOM_mon = np.array(FLUXCOM_mon)
FLUXCOM_TER = np.array(FLUXCOM_TER)
print('read TER')

HW_ym = []
for i_year,i_mon in zip(HW_year ,HW_mon):
    i_ym = (i_year-FLUXCOM_year[0])*12+i_mon
    HW_ym.append(i_ym)
HW_ym = np.array(HW_ym)
TER_ano = remove_seasonal_modis(FLUXCOM_TER,12,int(len(FLUXCOM_year)/12))
get_HW_ano_mean_modis_all('TER',TER_ano, HW_ym,HW_lat,HW_lon)
print('cal TER')















