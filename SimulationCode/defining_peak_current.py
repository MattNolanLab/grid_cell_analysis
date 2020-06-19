import numpy as np
from matplotlib import pyplot as plt
import random
import math 
from neuron import h, gui
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import h5py
from os import listdir 
from os.path import isfile
import csv
import glob
import os
import pandas as pd
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import cmocean
import pickle
from skimage import measure
from skimage.transform import rotate
from sympy import Eq, Symbol, solve
#Morphology
h.xopen("./nolan/hocfile.hoc")


def generate_cells(n):
    single_cell = []
    for i in np.arange(n):
        cell=h.Cell1()
        single_cell.append(cell)
    return single_cell

def generate_attachments(single_cell):
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
    
    return attachment_dend, n_dend_attach

def generate_stimulus(single_cell, attachment_dend, times, weight, n_dend_attach):
    times=h.Vector(times) #eventvec move from seconds to ms
    single_cell[0].tlist.append(times)
    inputs=h.VecStim() #presyn
    inputs.play(times)
    single_cell[0].vslist.append(inputs)
    
    for i in np.arange(n_dend_attach): #create synapses
        synapse = h.ExpSyn(single_cell[0].dend[int(attachment_dend[i])](0.5))
        synapse.tau=2
        single_cell[0].synlist.append(synapse)
        ncstim = h.NetCon(inputs, synapse)
        ncstim.weight[0]=weight
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


def simulate ():
    h.tstop=7000
    h.run()


def plot(single_cell):
    soma_v_vec, t_vec = set_recording_vectors(single_cell[0])
    simulate() #1 min60000 max(t_array)*1000

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


def create_test_cell():
    single_cell=generate_cells(1)
    single_cell=parameters(single_cell)
    for i in np.arange(n_hd):
        times=random.sample(range(5000,6000), 12)
        times=np.sort(times, axis=0)
        attachment_dend, n_dend_attach=generate_attachments(single_cell)
        single_cell=generate_stimulus(single_cell, attachment_dend, times, weight, n_dend_attach)
   
    #plot
    t_vec, soma_v_vec=plot(single_cell)
    
    return t_vec, soma_v_vec


def define_peak_current(n_hd_o): 
#maximal input current tested is adjusted manually for each input number such that firing rate reaches plateau
    h.cvode.active(1)
    inp=np.linspace(0, 0.000322, 50)
    tstop=7000
    np.random.seed(n_hd_o)
    random.seed(n_hd_o)

    shape=(len(n_hd_o),len(inp),20)
    l5b_all=np.zeros(shape)

    for j in np.arange(len(n_hd_o)):
        n_hd=n_hd_o[j]
        for i in np.arange(len(inp)):
            weight=inp[i]
            for m in np.arange(20):
                t_vec, soma_v_vec=create_test_cell()

                soma_array=soma_v_vec.to_python()
                peaks, _=find_peaks(soma_array,height=20)
                peak_times = [t_vec[i] for i in peaks] #ms
                l5b_all[j,i,m]=len(np.asarray(peak_times).astype(int))

    mean_rates=np.mean(l5b_all, axis=2)
    variance_rates=np.var(l5b_all, axis=2)
    x=inp
    
    #plot and fit
    a=19
    b=28
    plt.errorbar(x[a:b], mean_rates[0][a:b], np.sqrt(variance_rates[0][a:b]))
    z=np.polyfit(x[a:b], mean_rates[0][a:b], 2)
    p = np.poly1d(z)
    plt.plot(x[a:b], p(x)[a:b], linewidth=3, color='k') #black spline fit
    xfm = Symbol('x')
    eqn = Eq(p[0]+p[1]*xfm+p[2]*xfm**2, 12)
    sol=solve(eqn)
    print('peak_current='+str(sol))
 
#run and print peak_current
n_hd_o=30 #run for 30 input cells
define_peak_current(n_hd_o)
    
