#characterizes non-uniform conjunctive cell fields
#generates convergent grid cells with and without uniform conjunctive cell fields

import numpy as np
from matplotlib import pyplot as plt
import random
import math 
from neuron import h, gui
from scipy.stats import multivariate_normal
import csv
import glob
import os
import pandas as pd
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import pickle
from skimage import measure
from skimage.transform import rotate
from multiprocessing import Pool
##Morphology
h.xopen("./nolan/hocfile.hoc")
#Behavioral data
f = open('behaviour', 'rb')
dat=pickle.load(f)
f.close()
positions=dat[0]
hd=dat[1]
t_array=dat[2]


def grid_cell(x,y, res, ngrid, grid_period):
    #x=50
    #y=50
    x_size=np.arange(0,x,res)
    y_size=np.arange(0,y,res)
    x_array, y_array = np.meshgrid(x_size, y_size)
    pos = np.dstack((x_array.T, y_array.T))

    #ngrid=4 ##how many grid cells
    #grid_dist=1 ##every x cm there is a grid cell phase

    ##Template: per grid cell: x,y dimension
    height=(math.sqrt(3)/2)*grid_period
   
    rand_x1 = np.arange(0, grid_period, grid_period/4)
    rand_y1 = np.arange(0, height, height/4)
    rand_x2, rand_y2 = np.meshgrid(rand_x1, rand_y1)
    rand_x=np.ndarray.flatten(rand_x2)
    rand_y=np.ndarray.flatten(rand_y2)
    
    shape=(x_size.shape[0],y_size.shape[0], ngrid)
    OG_grid=np.zeros(shape)

    for i in np.arange(ngrid):
        x_phase=15
        y_phase=15
    
        peaks_x=np.arange(x_phase-grid_period, x+grid_period, grid_period)
        peaks2_x=np.arange(x_phase-0.5*grid_period, x+grid_period, grid_period)

        peaks_y=np.arange(y_phase, y+height, 2*height)
        peaks2_y=np.arange(y_phase-height, y+height, 2*height)

        peaks_x_array, peaks_y_array = np.meshgrid(peaks_x, peaks_y)
        peaks2_x_array, peaks2_y_array=np.meshgrid(peaks2_x, peaks2_y)

        peak_x=np.append([np.ndarray.flatten(peaks_x_array)],[np.ndarray.flatten(peaks2_x_array)])
        peak_y=np.append([np.ndarray.flatten(peaks_y_array)],[np.ndarray.flatten(peaks2_y_array)])
        peaks_all=np.transpose(np.append([peak_x],[peak_y], axis=0))

        shape=(x_size.shape[0],y_size.shape[0],peaks_all.shape[0])
        grids=np.zeros(shape)
        cov=[[4*1.8*res*10,0],[0,4*1.8*10*res]] #covariance for width of distribution

        for j in np.arange(peaks_all.shape[0]):
            mean=peaks_all[j,:]
            rand2 = np.random.random(1)
            grids[:,:,j] = multivariate_normal(mean, cov).pdf(pos)*rand2

        OG_grid[:,:,i]=np.sum(grids,axis=-1)
        prob_ad=280*0.001 #0.001s is a ms
        adjust=prob_ad/np.amax(OG_grid[:,:,i]) #to 10 Hz
        #for 10 Hz, 10 spikes/minute, probability of firing a spike per ms is 10/1000=0.01
        OG_grid[:,:,i]=OG_grid[:,:,i]*adjust #probability of firing in 0.001s, with max determined by 10 Hz multiply by 1000 to get HZ

    return x_array.T, y_array.T, OG_grid, ngrid;

def grid_cell_control(x,y, res, ngrid, grid_period):
    #x=50
    #y=50
    x_size=np.arange(0,x,res)
    y_size=np.arange(0,y,res)
    x_array, y_array = np.meshgrid(x_size, y_size)
    pos = np.dstack((x_array.T, y_array.T))

    #ngrid=4 ##how many grid cells
    #grid_dist=1 ##every x cm there is a grid cell phase

    ##Template: per grid cell: x,y dimension
    height=(math.sqrt(3)/2)*grid_period
   
    rand_x1 = np.arange(0, grid_period, grid_period/4)
    rand_y1 = np.arange(0, height, height/4)
    rand_x2, rand_y2 = np.meshgrid(rand_x1, rand_y1)
    rand_x=np.ndarray.flatten(rand_x2)
    rand_y=np.ndarray.flatten(rand_y2)
    
    shape=(x_size.shape[0],y_size.shape[0], ngrid)
    OG_grid=np.zeros(shape)

    for i in np.arange(ngrid):
        x_phase=15
        y_phase=15
    
        peaks_x=np.arange(x_phase-grid_period, x+grid_period, grid_period)
        peaks2_x=np.arange(x_phase-0.5*grid_period, x+grid_period, grid_period)

        peaks_y=np.arange(y_phase, y+height, 2*height)
        peaks2_y=np.arange(y_phase-height, y+height, 2*height)

        peaks_x_array, peaks_y_array = np.meshgrid(peaks_x, peaks_y)
        peaks2_x_array, peaks2_y_array=np.meshgrid(peaks2_x, peaks2_y)

        peak_x=np.append([np.ndarray.flatten(peaks_x_array)],[np.ndarray.flatten(peaks2_x_array)])
        peak_y=np.append([np.ndarray.flatten(peaks_y_array)],[np.ndarray.flatten(peaks2_y_array)])
        peaks_all=np.transpose(np.append([peak_x],[peak_y], axis=0))

        shape=(x_size.shape[0],y_size.shape[0],peaks_all.shape[0])
        grids=np.zeros(shape)
        cov=[[4*1.8*res*10,0],[0,4*1.8*10*res]] #covariance for width of distribution

        for j in np.arange(peaks_all.shape[0]):
            mean=peaks_all[j,:]
            rand2 = np.random.random(1)
            grids[:,:,j] = multivariate_normal(mean, cov).pdf(pos)

        OG_grid[:,:,i]=np.sum(grids,axis=-1)
        prob_ad=280*0.001 #0.001s is a ms
        adjust=prob_ad/np.amax(OG_grid[:,:,i]) #to 10 Hz
        #for 10 Hz, 10 spikes/minute, probability of firing a spike per ms is 10/1000=0.01
        OG_grid[:,:,i]=OG_grid[:,:,i]*adjust #probability of firing in 0.001s, with max determined by 10 Hz multiply by 1000 to get HZ

    return x_array.T, y_array.T, OG_grid, ngrid;


def create_hd(n_hd):
    angle_mean=360/n_hd #angle
    shape=(n_hd)
    angle_means=np.zeros(shape)
    for i in np.arange(n_hd):
        angle_means[i]=i*angle_mean

    variance=60
    x=np.arange(0,361,1)
    shape=(361, n_hd)#angle, firing probability
    hd_prob=np.zeros(shape)
    for i in np.arange(n_hd):
        hd_prob[:,i]=1/(variance * np.sqrt(2 * np.pi)) * np.exp( - (x - angle_means[i])**2 / (2 * variance**2))
        if angle_means[i]<180:
            hd_prob[180+int(angle_means[i]):361,i]=1/(variance * np.sqrt(2 * np.pi)) * np.exp( - (x - (360+angle_means[i]))**2 / (2 * variance**2))[180+int(angle_means[i]):361]    
        if angle_means[i]>180:
            hd_prob[0:int(angle_means[i]-180),i]=1/(variance * np.sqrt(2 * np.pi)) * np.exp( - (x - (-(360-angle_means[i])))**2 / (2 * variance**2))[0:int(angle_means[i]-180)]    

    prob_ad=120*0.001 #0.001s is a ms
    adjust=prob_ad/np.amax(hd_prob) #to 10 Hz
    #for 12 Hz, 12 spikes/minute, probability of firing a spike per ms is 12/1000=0.012
    hd_prob=hd_prob*adjust   

    sel=np.arange(0,n_hd,1) #select head direction for grid cell
    theta = np.linspace(0, 2*np.pi, 361)

    return hd_prob, sel, theta;

def generate_firing(res, res_t, ngrid, n_hd, positions, hd, hd_prob, OG_grid):

    shape=(int(ngrid), int(positions.shape[0]))
    grid_firing=np.zeros(shape)

    for i in np.arange(positions.shape[0]):
        hd_samp = int(hd[i])
        x = int(positions[i,0]/res)-1 
        y = int(positions[i,1]/res)-1
        
        rand2 = np.random.random(1)
        grid_firing[:,i]=rand2<OG_grid[x,y,:]*hd_prob[hd_samp, :]

    return grid_firing

def generate_cells(n):
    single_cell = []
    for i in np.arange(n):
        cell=h.Cell1()
        single_cell.append(cell)
    return single_cell

def generate_attachments(single_cell):
    #where dend_sel=apical, basal or all, firing=place or grid firing, sel=which place or grid cell
    #times are spike times
    #type_here: 0 for all, 1 for basal, 2 for apical

    num_dend_basal=0
    for i in single_cell[0].basal:
        num_dend_basal=num_dend_basal+1
    
    shape=(num_dend_basal)
    length_basal=np.zeros(shape)
    j=0
    for i in single_cell[0].basal:
        length_basal[j]=i.L
        j=j+1
    
    n_dend_attach=round(num_dend_basal/10)
    total_length=sum(length_basal)
    prob_dend=length_basal/total_length
    attachment_dend=np.random.choice(num_dend_basal, n_dend_attach, replace=False, p=prob_dend)
    
    return attachment_dend

def generate_stimulus(single_cell, attachment_dend, t_array, firing, sel, weight):
    times=h.Vector(t_array[firing[:]==1]*1000) #eventvec move from seconds to ms
    single_cell[0].tlist.append(times)
    inputs=h.VecStim() #presyn
    inputs.play(times)
    single_cell[0].vslist.append(inputs)
    for i in np.arange(attachment_dend.shape[0]): #create synapses
        synapse = h.ExpSyn(single_cell[0].dend[int(attachment_dend[i])](0.5))
        synapse.tau=2
        single_cell[0].synlist.append(synapse)
        ncstim = h.NetCon(inputs, synapse)
        ncstim.weight[0]=weight #0.03 weight at which threshold for initiating spike is crossed w 500 inputs, and also 
        ncstim.threshold=-32.4
        ncstim.delay=1
        single_cell[0].nclist.append(ncstim)

    return single_cell


def set_recording_vectors(cell):
    soma_v_vec=h.Vector()
    t_vec=h.Vector()
    soma_v_vec.record(cell.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)
    return soma_v_vec, t_vec


def simulate(t_array):
    h.tstop=max(t_array)*1000
    h.run()


def plot(single_cell):
    soma_v_vec, t_vec = set_recording_vectors(single_cell[0])
    simulate(t_array) #1 min60000 max(t_array)*1000

    return t_vec, soma_v_vec


def parameters(single_cell):
    h.celsius=37

    single_cell[0].soma.ena=45
    single_cell[0].soma.ek=-85

            #Sodium and potassium
    single_cell[0].soma.gbar_kv = 10000#potassium fast
    single_cell[0].soma.gbar_km = 15 #potassium slow
    single_cell[0].soma.gbar_na=72000 #sodium

    single_cell[0].soma.g_pas = 0.000033
    single_cell[0].soma.e_pas = -65 #v_init

    j=0
    for i in single_cell[0].axonal:
        single_cell[0].axon[j].gbar_kv =500
        single_cell[0].axon[j].gbar_km = 7
        single_cell[0].axon[j].gbar_na=1000
        single_cell[0].axon[j].ena=45
        single_cell[0].axon[j].ek=-85
        single_cell[0].axon[j].g_pas= 0.00001
        single_cell[0].axon[j].e_pas=-65
        j=j+1


    j=0
    for i in single_cell[0].basal:
        single_cell[0].dend[j].g_pas=0.00001
        single_cell[0].dend[j].e_pas=-65
        j=j+1
    return single_cell


def create_l5b_cells(weight, n_hd, grid_firing):
    single_cell=generate_cells(1)
    single_cell=parameters(single_cell)
    for i in np.arange(n_hd):
        n_hd=i
        attachment_dend=generate_attachments(single_cell)
        single_cell=generate_stimulus(single_cell, attachment_dend, t_array, grid_firing[i,:], i, weight)
   
    #plot
    t_vec, soma_v_vec=plot(single_cell)
    
    return t_vec, soma_v_vec

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


#for analysis of conjunctive cell hd
def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

def get_hd_histogram(hd_fr):
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(hd_fr, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=23)
    return smooth_hd

def get_hd_hist(hd_firing, hd):
    smooth_hd1=get_hd_histogram(hd_firing* np.pi / 180)
    smooth_hd2=get_hd_histogram(hd*np.pi/180)/1000
    
    smooth_hd=(smooth_hd1/smooth_hd2)
    preferred_direction =np.where(smooth_hd==max(smooth_hd))
    max_firing_rate_hd = np.max(smooth_hd)
    return smooth_hd, preferred_direction, max_firing_rate_hd, smooth_hd2

def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster): #for specific
    cluster_id = 0
    #hd_in_field, spike_times = get_hd_in_field_spikes(rate_map_indices, spatial_firing_cluster)
    hd_in_bin=[]
    hd_at_fire_in_bin=[]
    #define area to count spikes and times in, maybe switch away from square?
    for i in np.arange(rate_map_indices.shape[0]):
        j=rate_map_indices[i,1]
        k=rate_map_indices[i,0]
        y_min=j*bin_size_cm
        y_max=(j+1)*bin_size_cm
        x_min=k*bin_size_cm 
        x_max=(k+1)*bin_size_cm

        #determine head-direction data within this bin
        hd_in_bin.extend(hd[(positions[:,0]>=x_min) & (positions[:,0]<x_max) & (positions[:,1]>=y_min) & (positions[:,1]<y_max),0]) #head_directions recorded when the animal was in bin
        hd_at_fire_in_bin.extend(hd_firing[(target_firing[:,0]>=x_min)&(target_firing[:,0]<x_max) & (target_firing[:,1]>=y_min) & (target_firing[:,1]<y_max)]) #what hd are we firing at at this time
        
    hd_in_bin=np.asarray(hd_in_bin)
    hd_at_fire_in_bin=np.asarray(hd_at_fire_in_bin)
    smooth_hd, preferred_direction, max_firing_rate_hd, smooth_hd2=get_hd_hist(hd_at_fire_in_bin, hd_in_bin)
        
    return smooth_hd, preferred_direction, max_firing_rate_hd, smooth_hd2

def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny

def worker(i):
    #inp=[3,6,9,12,15,18,21,24,27,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
    #weights=[0.000850, 0.00054, 0.000412, 0.000339, 0.000275, 0.000241, 0.000218, 0.000196, 0.000173,0.000161, 0.000130,0.000107,9.26e-5,8.17e-5,7.29e-5,6.62e-5,5.97e-5, 4.23e-5,3.15e-5, 2.61e-5,2.15e-5,1.91e-5,1.68e-5,1.50e-5,1.32e-5,1.13e-5,9.68e-6,8.41e-6,7.61e-6,6.83e-6]
    inp=[30]
    weights=[0.000161]
    np.random.seed(inp[i])
    res=1
    res_t=0.001 #1ms resolution
    grid_period=50
    h.cvode.active(1)
    x=105
    y=105
    trial_num=20
    print('input_number='+str(inp[i]))
    
    for a in np.arange(trial_num):
        print('trial_number='+str(a))
        n_hd=ngrid=inp[i]
        weight=weights[i]
        hd_prob, sel, theta=create_hd(n_hd)
        x_array, y_array, OG_grid, ngrid=grid_cell(x,y,res,ngrid,grid_period)
        grid_firing=generate_firing(res, res_t, ngrid, n_hd, positions, hd, hd_prob, OG_grid)
        t_vec, soma_v_vec=create_l5b_cells(weight, n_hd, grid_firing)

        soma_array=soma_v_vec.to_python()
        peaks, _=find_peaks(soma_array,height=20)
        peak_times = [t_vec[i] for i in peaks] #ms
        peak_times=np.round(np.asarray(peak_times)).astype(int)
        target_firing=positions[peak_times, :]
        hd_firing=hd[peak_times]

        x_firing=target_firing[:,0]
        y_firing=target_firing[:,1]

        values_cell1=[(t_array, positions[:,0], positions[:,1], hd)]
        labels = ['synced_time', 'position_x','position_y', 'hd']
        spatial_data = pd.DataFrame.from_records(values_cell1, columns=labels)
        if a==0:
            spatial_data.to_pickle("./v_spatial_data")
        
        values2=[(peak_times, x_firing, y_firing, hd_firing)] #peak times in ms
        labels=['firing_times','position_x', 'position_y', 'hd'] #where x and y are firing locations
        firing_data_spatial=pd.DataFrame.from_records(values2, columns=labels)
        
        firing_data_spatial.to_pickle("./v_grid"+str(n_hd)+"trials"+str(a))

def control(i):
    #inp=[3,6,9,12,15,18,21,24,27,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
    #weights=[0.000850, 0.00054, 0.000412, 0.000339, 0.000275, 0.000241, 0.000218, 0.000196, 0.000173,0.000161, 0.000130,0.000107,9.26e-5,8.17e-5,7.29e-5,6.62e-5,5.97e-5, 4.23e-5,3.15e-5, 2.61e-5,2.15e-5,1.91e-5,1.68e-5,1.50e-5,1.32e-5,1.13e-5,9.68e-6,8.41e-6,7.61e-6,6.83e-6]
    inp=[30]
    weights=[0.000161]
    np.random.seed(inp[i])
    res=1
    res_t=0.001 #1ms resolution
    grid_period=50
    h.cvode.active(1)
    x=105
    y=105
    trial_num=20
    print('input_number='+str(inp[i]))
    
    for a in np.arange(trial_num):
        print('trial_number='+str(a))
        n_hd=ngrid=inp[i]
        weight=weights[i]
        hd_prob, sel, theta=create_hd(n_hd)
        x_array, y_array, OG_grid, ngrid=grid_cell_control(x,y,res,ngrid,grid_period)
        grid_firing=generate_firing(res, res_t, ngrid, n_hd, positions, hd, hd_prob, OG_grid)
        t_vec, soma_v_vec=create_l5b_cells(weight, n_hd, grid_firing)

        soma_array=soma_v_vec.to_python()
        peaks, _=find_peaks(soma_array,height=20)
        peak_times = [t_vec[i] for i in peaks] #ms
        peak_times=np.round(np.asarray(peak_times)).astype(int)
        target_firing=positions[peak_times, :]
        hd_firing=hd[peak_times]

        x_firing=target_firing[:,0]
        y_firing=target_firing[:,1]
        
        values2=[(peak_times, x_firing, y_firing, hd_firing)] #peak times in ms
        labels=['firing_times','position_x', 'position_y', 'hd'] #where x and y are firing locations
        firing_data_spatial=pd.DataFrame.from_records(values2, columns=labels)
        
        firing_data_spatial.to_pickle("./v_grid_control"+str(n_hd)+"trials"+str(a))
        
#Characterize conjunctive cell grid and hd max firing rates
x=105
y=105
res=1
res_t=0.001
n_hd=ngrid=100 #100 samples
np.random.seed(0)
x_array, y_array, OG_grid, ngrid=grid_cell(x,y,res,ngrid,grid_period=50) #16
hd_prob, sel, theta=create_hd(n_hd)
grid_firing=generate_firing(res, res_t, ngrid, n_hd, positions, hd, hd_prob, OG_grid)

max_firing_rate=np.zeros(100)
max_firing_rate_head=np.zeros(100)

#figure 4a from k=0 and k=50
for k in np.arange(100):
    print(k)
    x=105
    y=105
    peak_times=(np.round(t_array[grid_firing[k,:]==1]*1000)).astype(int)
    target_firing=positions[peak_times, :]

    spike_positions_x=target_firing[:,0]
    spike_positions_y=target_firing[:,1]
    hd_firing=hd[peak_times]

    bin_size_cm = 2.5
    number_of_bins_x = math.ceil(x/ bin_size_cm)
    number_of_bins_y = math.ceil(y / bin_size_cm)
    bin_size_pixels = bin_size_cm

    shape=(number_of_bins_x, number_of_bins_y)
    firing_rate_map=np.zeros(shape)
    spikes_in_bin=np.zeros(shape)
    times_in_bin=np.zeros(shape)
    smooth = 5
    dt_position_ms = 1
    positions_x=positions[:,0]
    positions_y=positions[:,1]

    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (1/1000))
    
    max_firing_rate[k]=np.amax(firing_rate_map)
    
    smooth_hd, preferred_direction, max_firing_rate_hd, smooth_hd2=get_hd_hist(hd_firing, hd)
    max_firing_rate_head[k]=max_firing_rate_hd
    
print(np.mean(max_firing_rate))
print(np.std(max_firing_rate))
print(np.mean(max_firing_rate_head))
print(np.std(max_firing_rate_head))
        
#run in parallel, designed for multiple inputs
#if __name__ == '__main__':
#    numbers = np.arange(1) #how many input numbers are being investigated
#    pool = Pool(processes=1)
#    a=pool.map(worker, numbers)

#Run and save files for analysis
#run for single input
worker(0)
#figure 4b,c based on v_grid30trials19 
#through PostSorting/open_field_grid_cells.py (b top), open_field_head_direction.py (b bottom) and PostSorting/open_field_firing_fields.py (c)

#run control with uniform conjunctive fields
control(0)
