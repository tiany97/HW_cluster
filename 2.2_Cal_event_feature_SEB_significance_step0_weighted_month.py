# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import random
#%%

def remove_seasonal(var):
    var_detrend = np.zeros_like(var)
    for i_box in range(365):
        temp = np.array([var[365*i_year+i_box] for i_year in range(42)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(42):
            var_detrend[365*i_year+i_box]  = temp[i_year]
    return var_detrend 
#%%
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
def remove_seasonal_modis(var,n_step,n_year):
    var_detrend = np.zeros_like(var)
    for i_box in range(n_step):
        temp = np.array([var[n_step*i_year+i_box] for i_year in range(n_year)])
        temp = temp - np.array(len(temp)*[np.nanmean(temp,axis=0)])
        #temp = signal.detrend(temp,axis=0)
        for i_year in range(n_year):
            var_detrend[n_step*i_year+i_box]  = temp[i_year]
    return var_detrend 
#%%
def get_HW_ano_mean_modis(variable, HW_date,HW_lat,HW_lon):
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        if (np.array(i_date)<0).any() or (np.array(i_date)>=len(variable)).any():
            HW_ano.append(np.nan)
        else:
            HW_ano.append(np.nanmean((variable[i_date,i_lat,i_lon])))
    return np.array(HW_ano)
    
def get_HW_ano_mean_modis_all(var,variable, HW_date,HW_lat,HW_lon):
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        if (np.array(i_date)<0).any() or (np.array(i_date)>=len(variable)).any():
            HW_ano.append([np.nan])
        else:
            HW_ano.append((variable[i_date,i_lat,i_lon]))
    with open(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/HW_'+var+'_ano_6days_all.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(HW_ano)
        print(var+'end')

#%%
def get_HW_ano_all_mean(var_ravel,  N):
    HW_ano = []
    for i in N:
        temp = []
        for j in i:
            j = np.array([int(ii) for ii in j])
            temp.append(np.nanmean(var_ravel[j]))
        HW_ano.append(np.array(temp))
    return np.array(HW_ano)

print('hello from YT')
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
        #temp = signal.detrend(temp,axis=0)
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

HW_year = np.array([year_era5_all[i] for i in HW_date])
HW_mon = np.array([mon_era5_all[i] for i in HW_date])

HW_year_all = np.array([year_era5_all[i] for i in HW_date])
HW_mon_all = np.array([mon_era5_all[i] for i in HW_date])
HW_ym = []
for i_year,i_mon in zip(HW_year_all ,HW_mon_all):
    i_ym = (i_year-1979)*12+i_mon
    HW_ym.append(i_ym)
HW_ym = np.array(HW_ym)

event_N = len(HW_duration)
print( 'extreme event n = ' +str(event_N))

#%%
vername = '_235'
flname = r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/HW_labels'+vername+'.csv'
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

#%%
# monthly

ym =np.array(40*[np.arange(1,13)]).ravel()
label_map = np.zeros((4,720,1440))
nn = 0
for i_k  in np.arange(event_N):
    # if HW_size_label[i_k]==1:
    #     continue
    xx = HW_lon[i_k]
    yy = HW_lat[i_k]
    zz = HW_labels_all[i_k]
    for i,j,k in zip(yy,xx,zz):
        label_map[k,i,j]=label_map[k,i,j]+1
    nn = nn+1


f_map = np.zeros((4,40*12,720,1440))
nn = 0
for i_k  in np.arange(event_N):
    xx = HW_lon[i_k]
    yy = HW_lat[i_k]
    zz = HW_labels_all[i_k]
    tt = HW_ym[i_k]-1


    for z,t,i,j in zip(zz,tt,yy,xx):
        if t>=40*12:
            continue
        f_map[z,t,i,j]=f_map[z,t,i,j]+1
        #print(f_map[z,t,i,j])
    nn = nn+1
f_map1 = copy.deepcopy(f_map)
##
f_map1[f_map1==0]=np.nan
f_map[f_map==0]=np.nan
##
len(f_map1[~np.isnan(f_map1)])
##
# for t in range(0*12):
#     #print(t)
#     for i in range(181):
#         for j in range(360):
#             temp = f_map[:,t,i,j]
#             #print(t,i,j)
#             if temp[np.argsort(temp)[2]]-temp[np.argsort(temp)[1]]==0:
#                 f_map[:,t,i,j]=f_map[:,t,i,j]*np.nan
#             else:
#                 f_map[:,t,i,j][np.argsort(temp)[:2]]=f_map[:,t,i,j][np.argsort(temp)[:2]]*np.nan
# ##


vername = '_235'
# M_all = []
#M_all1 = []
for i_label in range(4):
    M_temp = []
    print(i_label)
    for i_random in range(1000):
        
        all_ano = np.zeros((40*12,720,1440))
        temp = np.nansum(f_map1[i_label ],axis=0)
        for ii, i in enumerate(all_ano):
            i[(land_mask<0.1)|(LAT_era5<-60)]=np.nan
            i[temp<1] = np.nan
            if ym[ii] in [3,4,5,6,7,8,9,10,11]:
                i[LAT_era5<-23.5]=  np.nan
            if ym[ii] in [1,2,3,4,5,9,10,11,12]:
                i[LAT_era5>23.5]=  np.nan
        for i_lat in range(720):
            for i_lon in range(1440):
                n_all = len(np.where(~np.isnan(all_ano[:,i_lat,i_lon]))[0])
                if n_all==0:
                    continue
                n_out = int(temp[i_lat,i_lon])
                if n_out==0:
                    continue
                randomlist = random.sample(range(n_all), n_out)
                randomlist_new = [np.where(~np.isnan(all_ano[:,i_lat,i_lon]))[0][j] for j in randomlist]
                for j in randomlist_new:
                    all_ano[j,i_lat,i_lon] =1
    #plt.imshow(np.nansum(all_ano,axis=0))
        M_temp.append(np.where(all_ano.ravel()==1)[0])
    indices_xr = xr.DataArray(np.array(M_temp))
    indices_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/strict_significance_N_'+str(i_label)+'monthly'+vername+'_weighted.nc')

    # M_all.append(np.array(M_temp))
    #M_all1.append(np.nansum(all_ano,axis=0))