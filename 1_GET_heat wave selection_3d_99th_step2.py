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

import sys

run0=int(sys.argv[1])
run1=int(sys.argv[2])
print(run0,run1)
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


#%%
print('start feature')
N = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_labels_cc3d_N.nc') ['__xarray_dataarray_variable__'])
if run1>N:
    run1=N
print('N=',N)
extreme_label = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_label.nc') ['__xarray_dataarray_variable__'][:][:])
extreme_labels_cc3d = np.array(xr.open_dataset('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_025_degree/extreme_labels_cc3d.nc') ['__xarray_dataarray_variable__'][:][:])

#%%
stats = cc3d.statistics(extreme_labels_cc3d)


print('Clump_size')

Clump_size = np.array([stats['voxel_counts'][i_N] for i_N in range(N)])
print('Clump_bounding_boxes')
Clump_bounding_boxes = np.array([stats['bounding_boxes'][i_N] for i_N in range(N)])
print('Clump_centroids')
Clump_centroids = np.array([stats['centroids'][i_N] for i_N in range(N)])
print('Clump_bounding_duration')
Clump_bounding_duration = np.array([Clump_bounding_boxes[i_N][0].stop-Clump_bounding_boxes[i_N][0].start for i_N in range(N)])
print(len(Clump_bounding_duration))

del stats
#%%
yy = [[int(Clump_centroids[i_k][1])] for i_k in range(1,N)]
xx = [[int(Clump_centroids[i_k][2])] for i_k in range(1,N)]
#%%
HW_labels = []

leastdate = 5
nn=0

print(run0,run1)
for i_k in range(run0,run1): #only consider points with extreme_label of 1   
    if land_mask[int(Clump_centroids[i_k][1]),int(Clump_centroids[i_k][2])]<0.1:
        if os.path.isfile(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/event_'+str(i_k)+'.csv'):
            os.remove( r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/event_'+str(i_k)+'.csv')
        print(i_k,'ocean')
        continue #exclude island
        print(i_k,'ocean')

    if os.path.isfile(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/event_'+str(i_k)+'.csv'):
     
        print(i_k,'exist')
        continue
    print(i_k,nn)
    if Clump_size[i_k]<leastdate:
         #at least {leastdate} connected points (1 spatially X {leastdate} temporally)
        print(i_k,'less than 6',Clump_bounding_duration[i_k])
        continue
    if Clump_bounding_duration[i_k]<leastdate:
         #at least {leastdate} consecutive days
        print(i_k,'less than 6',Clump_bounding_duration[i_k])
        continue

    if   lat_era5[int(Clump_centroids[i_k][1])]<-23.5:
        if mon_era5_all[int(Clump_centroids[i_k][0])] not in [12,1,2]:
        
            print(i_k,'not summer')
            continue
    if   lat_era5[int(Clump_centroids[i_k][1])]>=23.5:
        if mon_era5_all[int(Clump_centroids[i_k][0])] not in [6,7,8]:
              #only record summer heatwave 
            print('not summer')
            continue
    label_k = np.where(extreme_labels_cc3d==i_k)

    i_date = label_k[0]
    i_lat = label_k[1]
    i_lon = label_k[2]
    i_duration = np.max(i_date)-np.min(i_date)

    hor_set = []
    i_area = 0
    for i,j in zip(i_lat,i_lon):
        if (i,j) not in hor_set:
            hor_set.append((i,j) )

            y_dis = haversine((0, 0), (0.25, 0), unit='km')
            if lon_era5[j]+0.25>=180:
                x_dis = haversine((lat_era5[i], 180-0.25), (lat_era5[i], 180), unit='km')
            else:
                x_dis = haversine((lat_era5[i], lon_era5[j]), (lat_era5[i], lon_era5[j]+0.25), unit='km')
            i_area = i_area + x_dis*y_dis
    hor_set = np.array(hor_set)


    if i_duration<leastdate:
        print(i_k,'less than 6')
        continue #at least {leastdate} consecutive days
    if i_area<40000:
      
        print(i_k,'less than 40000')
        continue #at least 40000 km^2  

    HW_labels.append(i_k)
    print(i_k,i_duration,i_area,len(hor_set),\
    year_era5_all[int(Clump_centroids[i_k][0])],\
        mon_era5_all[int(Clump_centroids[i_k][0])],\
            lat_era5[int(Clump_centroids[i_k][1])],\
               lon_era5[int(Clump_centroids[i_k][2])] )
    nn=nn+1
    
    row_list = [i_date,i_lat,i_lon,hor_set[:,0],hor_set[:,1],[i_duration],[i_area],[Clump_centroids[i_k][0]],[Clump_centroids[i_k][1],Clump_centroids[i_k][2]]]
    with open(r'/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_025_degree/event_'+str(i_k)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
# HW_labels_xr = xr.DataArray(HW_labels,dims=['HW_number'])
# HW_labels_xr.to_netcdf('/Net/Groups/BGI/scratch/yt/result/Heat_wave_3D_99th_connect6_event_new/HW_labels_all.nc')

    




