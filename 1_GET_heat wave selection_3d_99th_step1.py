#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:07:13 2023

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
from matplotlib import pyplot
import matplotlib.patches as patches
import xarray as xr
from haversine import haversine, Unit
import cc3d
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl
import matplotlib.cm as cm

def moving_window_percentile_15(var,per):
    var_per = np.zeros_like(var)
    for i_box in range(365):
        if i_box<=7:
            temp = np.array([var[365*i_year:365*i_year+15] for i_year in range(42)])
        elif i_box>=365-7:  
            temp = np.array([var[365*i_year+365-15:365*i_year+365] for i_year in range(42)])
        else:                  
            temp = np.array([var[365*i_year+i_box-7:365*i_year+i_box+8] for i_year in range(42)])
        temp1 = []
        for i in temp:
            temp1.extend(i)
        temp2 = np.nanpercentile(np.array(temp1),per,axis=0)
        #temp = signal.detrend(temp,axis=0)
        var_per[i_box::365]  = np.array(42*[temp2])
    return var_per 
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
#%% definition
t2m_max = []
for i_year in range(1979,2021):
    NCname = '/Net/Groups/data_BGC/era5/e1/0d25_daily/t2mmax/t2mmax.daily.an.era5.1440.720.'+str(i_year)+'.nc'
    NCData = Dataset(NCname)
    time = NCData.variables['time']
    dates = list(num2date(time[:], time.units, time.calendar))
    year_era5 = np.array([date.year for date in dates])
    mon_era5 = np.array([date.month for date in dates])
    day_era5 = np.array([date.day for date in dates])
    md_era5 = 100*np.array(mon_era5)+np.array(day_era5)

    var = NCData.variables['t2mmax'][:]
    NCData.close()
    var= np.squeeze(var[md_era5!=229])
    t2m_max.extend(np.array(var))

t2m_max = np.array(t2m_max)
print(t2m_max.shape)
#%%

#%%
print('cal per99_start')
base_pctl90 = moving_window_percentile_15(t2m_max,99)
# base_pctl90_xr = xr.DataArray(base_pctl90 )
# base_pctl90_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/data/era5_daily_1x1/base_pctl99_1979_2020_dailymean_025x025_del29Feb.nc')
print('cal per99_end')
#%%
print('cal label_start')
for i in t2m_max:
    i[(land_mask==0)|(LAT_era5<-60)]=np.nan
for j in base_pctl90:
    j[(land_mask==0)|(LAT_era5<-60)]=np.nan
    #%%
extreme_label = np.zeros_like(t2m_max)
#%%
extreme_label[t2m_max>base_pctl90]=1
print(len(extreme_label[extreme_label==1])/len(extreme_label[~np.isnan(t2m_max)]))
print('cal label_end')
#%%
print('cc3d_start')
connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
extreme_labels_cc3d, N = cc3d.connected_components(extreme_label, connectivity=connectivity,return_N=True,out_dtype=np.uint64)
print(N)
print('cc3d_end')
#%%
print('writing')
extreme_label_xr = xr.DataArray(extreme_label,dims=['time','lat','lon'])
extreme_label_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_label.nc')
#%%
extreme_labels_cc3d_xr = xr.DataArray(extreme_labels_cc3d,dims=['time','lat','lon'])
extreme_labels_cc3d_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_labels_cc3d.nc')

#%%
N_xr = xr.DataArray(N)
N_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_labels_cc3d_N.nc')
print('3d HW cc3d end')
