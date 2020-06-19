#importing M5-0313, processed through PostSorting/open_field_spatial_data.py

import numpy as np
from matplotlib import pyplot as plt
import random
import math 
import csv
import pandas as pd
import pickle

def get_data(t, h, location_x, location_y, res_t):
    max_t = max(t)
    t_array=np.arange(0,max_t,res_t) #1 s bins, 0-1, 1-2...
    shape=(t_array.shape[0],2)
    positions=np.zeros(shape)
    shape=(t_array.shape[0])
    hd=np.zeros(shape)

    for i in np.arange(t_array.shape[0]):
        positions[i,0]=np.mean(location_x[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
        positions[i,1]=np.mean(location_y[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
        hd[i]=np.mean(h[(t>=t_array[i]) & (t<(t_array[i]+res_t))])
    
    df = pd.DataFrame(positions[:,0]).interpolate()
    positions[:,0]=df.values[:,0]
    df2 = pd.DataFrame(positions[:,1]).interpolate()
    positions[:,1]=df2.values[:,0]
    
    df3 = pd.DataFrame(hd).interpolate()
    hd=df3.values
    hd=hd+180
    
    x=max(positions[:,0])
    y=max(positions[:,1])
    
    return positions, x, y, hd, t_array
    
#import dataframe
all_dat=pd.read_pickle("./trajectory_all_mice.pkl")
dat1=all_dat[112:113]
location_x_p=dat1.position_x
location_x=location_x_p[~np.isnan(location_x_p)]
location_y=dat1.position_y[~np.isnan(location_x_p)]
h=dat1.hd[~np.isnan(location_x_p)]
t_pre=dat1.synced_time[~np.isnan(location_x_p)]#seconds
t=t_pre-t_pre[0]
res_t=0.001
res=1 #cm

#segment into 1 ms bins
positions, x, y, hd, t_array=get_data(t, h, location_x, location_y, res_t)

#save output
f = open('behaviour', 'wb')
pickle.dump([positions, hd, t_array], f)
f.close()

