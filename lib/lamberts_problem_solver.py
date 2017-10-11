
# coding: utf-8

import subprocess
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
from operator import itemgetter

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

global cmu_mu
global pop_size
cmu_mu = 2
pop_size = 15

# write a script to input this data in input.cst file..
def calculate_sv_at_boundary(t_launch, t_arrival):

    # Earth and mars position and velocity at time of launch
    launch_date_iso = ast_time.Time(t_launch, format='iso', scale='utc')
    arrival_date_iso = ast_time.Time(t_arrival, format='iso', scale='utc')
    rr_earth, vv_earth = get_body_ephem("earth", launch_date_iso)
    rr_mars, vv_mars = get_body_ephem("mars", arrival_date_iso)

    return [rr_earth.value, rr_mars.value, vv_earth.to(u.km/u.second).value]


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def folder_already_exists_check(path):
    if os.path.exists(path):
        remove_temp_dir(path)

def worskspace_exists_check(workspace_path):
    try:
        os.makedirs(workspace_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def remove_temp_dir(path):
    shutil.rmtree(path, ignore_errors=True)

def fitness_function(sw, process_number, final_vec, initial_v, destination_vector, queue):

    #print('Hello: {} {}'.format(LA.norm(x[0:3]),LA.norm(destination_vector[0:3])))

    start = time.time()
    sys.stdout.flush()

    # Fitness calculation comparing difference between the final position of the satellite and final position
    # of Mars and testing if its within the permissible limits
    #print('Velocity: {}|{}|{}'.format(x, initial_v,destination_vector))
    #print('Final Vector Satellite:{}'.format(final_vec))
    #print('Initial Velocity:{}'.format(initial_v))
    #print('Destination Position:{}'.format(destination_vector))

    term1 = LA.norm(final_vec[0:3] - destination_vector[0:3])
    term2 = LA.norm(initial_v)

    if term1 > 10**6:
        fitness = term1
    else:
        fitness = term1/10 + 15**term2
    #print('{} {} {}'.format(term1,term2,fitness))

    end = time.time()
    sys.stdout.flush()
    queue.put((process_number, fitness))

def fitness_calculator_splitter(sw, initial_vel, final_vector_satellite, final_vector_destination):
    # Call the fitness function to calculate the fitness of each solution
    #rearrange

    #sys.exit()
    queue = mp.Queue()
    if len(final_vector_destination) == 1:
        processes = [mp.Process(target=fitness_function, args=(sw, x,final_vector_satellite[x], initial_vel[x], final_vector_destination, queue)) for x in range(pop_size)]
    else:
        processes = [mp.Process(target=fitness_function, args=(sw, x, final_vector_satellite[x], initial_vel[x],final_vector_destination[x], queue))  for x in range(pop_size)]
    for p in processes:
        p.start()

    # Unhash the below if you run on Linux (Windows and Linux treat multiprocessing
    # differently as Windows lacks os.fork())
    for p in processes:
        p.join()

    fitness_results = [queue.get() for p in processes]
    #return results in arranged order
    candidates_fitness = np.zeros((pop_size,1))

    tmp_fit = sorted(fitness_results,key=itemgetter(0))
    for i in range(pop_size):
        candidates_fitness[i] = tmp_fit[i][1]

    return(candidates_fitness)

def calc_initial_vector(x,sv_old, earth_velocity):
    sv_init = np.zeros(6)
    sv_init = [sv_old[0], sv_old[1], sv_old[2], x[0] + earth_velocity[0] , x[1] + earth_velocity[1], x[2] + earth_velocity[2]]

    return(sv_init)


def integrator(process_number, x, initial_position, earth_velocity, queue, boundary_time = None):

    start = time.time()
    #print("Process {} has started at {}".format(process_number, start))
    sys.stdout.flush()

    # check for path existence
    #folder_already_exists_check(path)
    #create_folder(path)
    path = '../workspace/%d' % process_number
    folder_already_exists_check(path)
    if not os.path.exists(path):
        create_folder(path)

    # To copy the integrator
    subprocess.call('cp -r ../Orbit_Sim/* ../workspace/%d' % process_number , shell = True)

    # Enter the launch and end date in input.cst of master orbit_sim folder
    if boundary_time is not None:
        subprocess.call('../scripts/enter_times_integrator.sh "%s" "%s" %d' % (boundary_time[0], boundary_time[1], process_number), shell = True)

    # after adding delta v (impulse)
    sv_instant = calc_initial_vector(x,initial_position,earth_velocity) # intitial_state_vector corresponds to the earth's position at launch
        #print(sv_instant)

    # call to save the state vector in val.txt in the respective folder
    fo = open('../workspace/%d/val.txt' % process_number, "wb")
    np.savetxt(fo , sv_instant, fmt='%5.6f', delimiter=' ', newline=' ')
    fo.close()

    # call to edit input.cst
    subprocess.call('../scripts/enter_vector_intergrator.sh %d' % process_number , shell = True)

    # call to run each instance of integrator
    try:
        subprocess.call('cd ../workspace/%d && ./orbitsim64_rhel5' % process_number, shell = True)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # call to read final state vector
    subprocess.call('../scripts/read_final_sv.sh %d' % process_number, shell = True)
    sv_final = np.genfromtxt('../workspace/%d/output/FinalVector.txt' % process_number)

    # call to destroy val.txt for next step
    os.remove('../workspace/%d/val.txt' % process_number)
    os.remove('../workspace/%d/output/FinalVector.txt' % process_number)
    #subprocess.call('../scripts/delete_tmp_file.sh %d' % process_number , shell = True)

    #remove_temp_dir(path)

    # Returning the final vector to the calling method
    end = time.time()
    #print("Process {} has ended at {}".format(process_number, end))
    sys.stdout.flush()

    if queue is not None:
        queue.put((process_number, sv_final))
    else:
        return(sv_final)


def process_splitter(X, start_position, earth_velocity, dates_boundary = None):
    # check for workspace folder existence
    workspace_folder = '../workspace'
    #worskspace_exists_check(workspace_folder)
    if not os.path.exists(workspace_folder):
        create_folder(workspace_folder)

    queue = mp.Queue()
    if len(start_position) == 1:
        processes = [mp.Process(target=integrator, args=(x, X[x], start_position, earth_velocity, queue)) for x in range(pop_size)]
    else:
        processes = [mp.Process(target=integrator, args=(x, X[x], start_position[x], earth_velocity[x], queue, dates_boundary[x])) for x in range(pop_size)]
    for p in processes:
        p.start()

    # Unhash the below if you run on Linux (Windows and Linux treat multiprocessing
    # differently as Windows lacks os.fork())
    for p in processes:
        p.join()

    results = [queue.get() for p in processes]
    #Re-arrange the final candidates according to the process number
    candidates = np.zeros((pop_size,6))
    tmp = sorted(results,key=itemgetter(0))

    for i in range(pop_size):
        candidates[i] = tmp[i][1]

    return(candidates)


def multiproc_master_lamberts_problem(launch_position,destination_position, earth_velocity, ftar = None):

   # global X

    lock = Lock()
    queue = mp.Queue()

    if ftar is None:
        es = cma.CMAEvolutionStrategy([0,0,0],30, {'seed':10000, 'CMA_mu':cmu_mu, 'popsize':pop_size, 'bounds': [[-20, -20, -20], [20, 20, 20]], 'ftarget': 0, 'maxiter':150,'verb_append':1})
    else:
        es = cma.CMAEvolutionStrategy([-1, 2, -1],30, {'seed':10000, 'CMA_mu':cmu_mu, 'popsize':pop_size, 'bounds': [[-20, -20, -20], [20, 20, 20]], 'ftarget': ftar, 'maxiter':200,'verb_append':1})

    while not es.stop():
        X = []
        X = es.ask()
        #print(X)

        final_vector = process_splitter(X,launch_position, earth_velocity)
        fit = fitness_calculator_splitter(final_vector, destination_position)

        #check for acceptence
        es.tell(X, fit)
        es.disp(5)

    return [X,fit]


def lamberts_problem_solver(t_launch, t_arrival, ftarget = None):

    # Enter the launch and end date in input.cst of master orbit_sim folder
    subprocess.call('../scripts/enter_times_master_intergrator.sh "%s" "%s"' % (t_launch, t_arrival), shell = True)

    # Testing func_main()
    #X = np.array([-27.22701736111111, 11.944599537037037, 5.176816666666667])
    #a = func_main(1,X)
    [position_launch, position_arrival, velocity_earth_launch] = calculate_sv_at_boundary(t_launch, t_arrival)
    print('Earth State Vector: {},{}'.format(position_launch, velocity_earth_launch))

    if ftarget is None:
        [X,fitness] = multiproc_master_lamberts_problem(position_launch, position_arrival, velocity_earth_launch)
    else:
        [X,fitness] = multiproc_master_lamberts_problem(position_launch, position_arrival, velocity_earth_launch, ftarget)
    #print("X = {} \n Fitness = {}.".format(X, fitness))

    # clean worskpace
    path = '../workspace'
    remove_temp_dir(path)

    #
    min_index = fitness.tolist().index(min(fitness))
    return [X[np.array(min_index)], fitness[np.array(min_index)]]

def test1():
    print('Hello')
    return 1
