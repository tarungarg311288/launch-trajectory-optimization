
# coding: utf-8

# In[1]:

import multiprocessing as mp
from multiprocessing import Lock
import time
import random
import sys
import os
import errno
import shutil 
import subprocess
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

import cma
from decimal import Decimal
from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit

from poliastro.plotting import plot
from poliastro.plotting import OrbitPlotter
from numpy import linalg as LA


from mpl_toolkits.mplot3d import Axes3D

import astropy.units as u
from astropy import time as ast_time
 
from poliastro import iod

from astropy.coordinates import solar_system_ephemeris
from poliastro.ephem import get_body_ephem

get_ipython().magic('matplotlib inline')

#initial_state_vector = np.concatenate(rr_earth.value, vv_earth.value/24, axis=0)


# In[ ]:

def call_lamberts_problem_solver(boundary_dates):
    launch_date = Time(boundary_dates[0], format='jd',scale='utc')
    arrival_date = Time(boundary_dates[1], format='jd',scale='utc')
    launch_date_iso = launch_date.iso
    arrival_date_iso = arrival_date.iso 
    
    # Pankaj (Read this. done means maine kar liya hai and 5th and 6th point mein kar lunga.)
    '''
    Call the lambert problem module for finding the minimum delta_v for this particular time boundary conditions
    
    1) Tomorrow pehle to lamberts_solver mein input arguments pass karwaane hai launch_date and arrival_date.
    
    2)[Done]fir usko hee test karo ek alag python file bana ke jo lambert_solver ko import karke usko use kare.
    
    3)[Done]uske baad wo launch_date and arrival date automatic input.cst mein daal de iske liye ek method banaana hai.
        yeh method scripting se as well as python se kar sakte hain
    
    4)Yahan se lambert_solver ko call karna hai. 
    
    5) [I will do it] lambert_problem mein [delta_v_x, delta_v_y, delta_v_z, fitness] ko return karne ka scene kar
    
    6) [I will do it in the last] ek baar serially theek chal jaaye then isko multi-processor pe laana hai.
    '''
    # import lambert_problem
    [delta_v_x, delta_v_y, delta_v_z, fitness] = lambert_problem(launch_date_iso, arrival_date_iso)
    
    return ([delta_v_x, delta_v_y, delta_v_z, fitness])
    
def multiproc_master_boundary_dates():
    
   # global X
    lock = Lock()
    queue = mp.Queue()
    
    # Launch time and Arrival time are taken as unknown.
    # The limits for launch time is taken as the complete window of Earth-Mars minimum distance encounter. 
    # Usually around 2.3 years. Arrival date is taken > arrival date + 200 days atleast. So we will put 
    # this condition inside the while loop for selective children formations. 
    # Have to use Julian dates. To make the es work easily.
    # Launch date: [07-07-2017 00:00:00.0, 07-01-2020 00:00:00.0] in JD : [2457941.5, 2458855.5]
    # Arrival Date: [07-07-2017 00:00:00.0 + 200days, 07-01-2020 00:00:00.0 + 400days] in JD [2458141.5, 2459255.5]
    es = cma.CMAEvolutionStrategy([2458200.0, 2458400.5], 50, {'seed':10000, 'CMA_mu':cmu_mu, 'popsize':pop_size,
                                'bounds': [[2457941.5, 2458855.5], [2458141.5, 2459255.5]], 
                                                    'ftarget': 0, 'maxiter':100,'verb_append':1})
        
    while not es.stop():
        X, fit = [], []
        while len(X) < es.popsize:
            x_tmp = es.ask(1)[0]
            if (x_tmp[1] - x_tmp[0] > 200)
                X.append(x_tmp)
        #print(X)
       
        # Do it multi Processed.
        [delta_v_x, delta_v_y, delta_v_z, Earth_Mars_Distance] = call_lamberts_problem_solver(X,launch_date,arrival_date)
        fit = Earth_Mars_Distance
        #check for acceptence
        es.tell(X, fit)       
        es.disp(5)
    
        
    print('X: {0} and Fitness: {1}.\n'.format(X,fit)) 

if __name__ == '__main__':
    split_jobs = multiproc_master_boundary_dates()
    #Final__velocity_vector = 
    print("Tarun")

