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

vername = '_235'
def get_HW_ano_all(var,variable, HW_date,HW_lat,HW_lon):
    vername = '_235'
    HW_ano = []
    for i_date,i_lat,i_lon in zip(HW_date,HW_lat,HW_lon):
        HW_ano.append(variable[i_date,i_lat,i_lon])
    with open(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/HW_'+var+'_ano_6days_all'+vername+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(HW_ano)
        print(var+'end')
        #%%

    
#%%
def read_data_025(varname, varname1):
    var_all = []
    for i_year in range(1979,2021):
        print(varname+'read start')  
        NCname = '/Net/Groups/data_BGC/era5/e1/0d25_daily/'+varname+'/'+varname+'.daily.'+varname1+'.era5.1440.720.'+str(i_year)+'.nc'
        NCData = Dataset(NCname)
        time = NCData.variables['time']
        dates = list(num2date(time[:], time.units, time.calendar))
        year_era5 = np.array([date.year for date in dates])
        mon_era5 = np.array([date.month for date in dates])
        day_era5 = np.array([date.day for date in dates])
        md_era5 = 100*np.array(mon_era5)+np.array(day_era5)

        var = NCData.variables[varname][:]
        unit_temp = NCData.variables[varname].units
        if unit_temp=='MJ m2':
            var=var*1e6/24/3600
        elif unit_temp=='J m**-2':
            var=var/24/3600
        var[var==-9999.]=np.nan
        NCData.close()
        var= np.squeeze(var[md_era5!=229])
        var_all.extend(np.array(var))
    var_all = np.array(var_all)  
    print(varname+'read end, shape')  
    print(var_all.shape)
    return var_all
skt = read_data_025('skt','an')

#%%
sshf = read_data_025('sshf','fc')
sshf_ano = SEB_ano(sshf,skt)
del sshf
get_HW_ano_all('sshf',sshf_ano, HW_date,HW_lat,HW_lon)

del sshf_ano

slhf = read_data_025('slhf','fc')
slhf_ano = SEB_ano(slhf,skt)
del slhf
get_HW_ano_all('slhf',slhf_ano, HW_date,HW_lat,HW_lon)
del slhf_ano

ssr = read_data_025('ssr','fc')
ssr_ano = SEB_ano(ssr,skt)
del ssr
get_HW_ano_all('ssr',ssr_ano, HW_date,HW_lat,HW_lon)
del ssr_ano

strd = read_data_025('strd','fc')
strd_ano = SEB_ano(strd,skt)
del strd
get_HW_ano_all('strd',strd_ano, HW_date,HW_lat,HW_lon)

del strd_ano
del skt
# get_HW_ano_all('q',skt_ano, HW_date,HW_lat,HW_lon)


adv = []
for i_year in np.arange(1979,2021):
    temp = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/data/era5_025_025/adv/adv_'+str(i_year)+'.nc') ['__xarray_dataarray_variable__'][:][:])
    adv.extend(temp)
adv = np.array(adv)
print(adv.shape)
adv_ano = remove_seasonal(adv)*3600*24/1000
get_HW_ano_all('adv',adv_ano, HW_date,HW_lat,HW_lon)
del adv_ano


skt = read_data_025('skt','an')
skt_ano = remove_seasonal(skt)
del skt
get_HW_ano_all('skt',skt_ano, HW_date,HW_lat,HW_lon)
del skt_ano

t2m = read_data_025('t2m','an')
t2m_ano = remove_seasonal(t2m)
del t2m
get_HW_ano_all('t2m',t2m_ano, HW_date,HW_lat,HW_lon)
del t2m_ano

