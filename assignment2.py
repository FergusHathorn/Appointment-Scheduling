# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:54:39 2022

@author: Fergus

Simulate in python an appointment schedule of length 3 hours, 5-minute intervals, and 12 patients, 
with lognormally distributed service times with mean 13 and standard deviation 5. 
(Make sure your distribution has the right parameters.) 
There are no no-shows (all patients show up), patients are on time.

"""
import numpy as np
import math
import time
from itertools import permutations
import random
import scipy.stats
import statistics
from statistics import mode
from statistics import mean

#%%
"""
a.
Use simulation to estimate the expected average patient waiting time and the tardiness 
of the doctor (the time the doctor is busy after the end of the appointment block), first 
for the individual schedule (where patients are equally spaced).

"""

# calculating mu and sigma: https://en.wikipedia.org/wiki/Log-normal_distribution
ln_mu = math.log((13**2)/np.sqrt(13**2 + 5**2))
ln_sigma = np.sqrt(math.log(1 + (5**2)/(13**2)))

# function to return an appointment length
def apt(mu = ln_mu, sig = ln_sigma):
    return np.random.lognormal(mean=mu,sigma=sig)

params = {
    'patients':12,
    'interval':5,
    'sim_length':3*60,
    'appt_block_length':15
    }

schedule = {i: (i-1)*params['appt_block_length'] for i in range(1,params['patients']+1)} # evenly spaced schedule

def simulate(schedule,simulations, params):
    tardiness = []
    waiting = []
    for i in range(simulations):
        waiting_times = []
        finish_times = []
        
        sim_time = 0
        finish_time = 0
        for patient in schedule:
            waiting_times.append(max(0,finish_time-schedule[patient]))
            finish_time = sim_time+apt()
            finish_times.append(finish_time)
            sim_time = max(finish_time, schedule[patient])
            
        tardiness.append(max(sim_time-params['sim_length'],0))
        waiting.append(waiting_times)
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    
    return mean_waiting_time, mean_tardiness

'''
def simulate(schedule,simulations, params):
    tardiness = []
    waiting = []
    for i in range(simulations):
        waiting_times = []
        finish_times = []
        
        sim_time = 0
        finish_time = 0
        for patient in schedule:
            waiting_times.append(max(0,finish_time-schedule[patient]))
            finish_time = sim_time+params['interval']*round(apt()/params['interval'])
            finish_times.append(finish_time)
            
            sim_time = max(finish_time, schedule[patient])
            
        tardiness.append(max(sim_time-params['sim_length'],0))
        waiting.append(waiting_times)
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    
    return mean_waiting_time, mean_tardiness
'''

print(simulate(schedule,1000, params))

#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""

individual_schedule = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]

def simopt(schedule,simulations,params,appointment_lengths):
    tardiness = []
    waiting = []
    for i in range(simulations):
        waiting_times = []
        finish_times = []
        
        sim_time = 0
        finish_time = 0
        for patient in schedule:
            waiting_times.append(max(0,finish_time-schedule[patient]))
            finish_time = sim_time+params['interval']*round(appointment_lengths[patient]/params['interval'])
            finish_times.append(finish_time)
            
            sim_time = max(finish_time, schedule[patient])
            
        tardiness.append(max(sim_time-params['sim_length'],0))
        waiting.append(waiting_times)
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    
    return mean_waiting_time, mean_tardiness

def get_objective_simopt(schedule,appointment_lengths,params=params,simulations=1):
    '''
    input: 
        the schedule of form (1,0,0,1,... etc) in intervals of 5 minutes. 1 represents the start
        of a proposed appointment, 0 means that an appointment is not scheduled to start in that period.
    
    output:
        objective value = 2x(mean tardiness) + mean waiting time
    '''
    start_times = [i for i,v in enumerate(schedule) if v] # index of scheduled appointment start time
    schedule_dict = {i: start_times[i]*params['interval'] for i in range(params['patients'])} # dictionary of scheduled start time per patient

    # simulating the schedule
    wait,tardiness = simopt(schedule_dict,simulations,params,appointment_lengths)
    
    return 2*wait + tardiness
    
"""
Understanding of sim-opt:
    create a neighborhood of solutions
    choose an initial state (maybe the individual schedule)
    randomly choose another neighbor
    simulate the same consulting times on each schedule
    'winner stays on'
    count how many times each schedule is simulated and also adjust the average score for each schedule

"""

# creating the neighborhood
slots=list(range(len(individual_schedule)))
blank_schedule = np.zeros(len(individual_schedule))
neighborhood=[individual_schedule]
neighborhood_size = 10000
while len(neighborhood) <= neighborhood_size:
    anchor = random.choice(slots)
    appt_slots = random.sample(slots,12)
    if len([i for i in appt_slots if i < 18]) >= 6 and 0 in appt_slots: 
        proposed_schedule = blank_schedule.copy()
        proposed_schedule[appt_slots]=1
        three_consec = False
        for i in range(len(proposed_schedule)-2):
            if sum(proposed_schedule[i:i+3]) == 3:
                three_consec = True
                break
        if not three_consec:
            if tuple(proposed_schedule) not in neighborhood:
                neighborhood.append(list(proposed_schedule))
        
# simulating
neighbors = neighborhood_size
budget = 10000
#primary = random.choice(range(neighbors))
primary = 0
scores = {i:[0,0] for i in range(neighbors)}
while budget > 0:
    random_neighbor = random.choice(range(neighbors))
    while random_neighbor == primary:
        random_neighbor = random.choice(range(neighbors))
    appointment_lengths = [apt() for i in range(params['patients'])]
    neighbor_obj = get_objective_simopt(neighborhood[random_neighbor],appointment_lengths)
    primary_obj = get_objective_simopt(neighborhood[primary],appointment_lengths)
    scores[primary][0] += 1
    scores[primary][1] += primary_obj
    scores[random_neighbor][0] += 1
    scores[random_neighbor][1] += neighbor_obj
    if primary_obj < neighbor_obj:
        primary = random_neighbor
    budget -= 1

best_solution = np.argmax([scores[i][0] for i in scores])
print(best_solution)
print(neighborhood[best_solution])

'''
# this isn't actually simopt, but interesting difference
mean_output = [scores[i][1]/scores[i][0] for i in scores if scores[i][0] != 0]
idx_min = np.argmin(mean_output)
idx = [i for i in scores if scores[i][0] != 0][idx_min]
print(idx)

'''
    
#%%
"""
c. 
Calculate for the optimal schedule the objective and its confidence interval. Simulate often 
enough to get a small confidence interval. Good solutions are ranked and at maximum 1 point is 
given according to the ranking.

"""

def get_objective(schedule,params=params,simulations=100):
    '''
    input: 
        the schedule of form (1,0,0,1,... etc) in intervals of 5 minutes. 1 represents the start
        of a proposed appointment, 0 means that an appointment is not scheduled to start in that period.
    
    output:
        objective value = 2x(mean tardiness) + mean waiting time
    '''
    start_times = [i for i,v in enumerate(schedule) if v] # index of scheduled appointment start time
    schedule_dict = {i: start_times[i]*params['interval'] for i in range(params['patients'])} # dictionary of scheduled start time per patient

    # simulating the schedule
    wait,tardiness = simulate(schedule_dict,simulations,params)
    
    return 2*wait + tardiness

def CI(optimal_schedule):
  width = 100000
  means_of_batches = []
  mean = 0
  while width > mean/100:
    appointment_lengths = [apt() for i in range(params['patients'])] 
    obj=get_objective(optimal_schedule,simulations=100)
    means_of_batches.append(obj)
    n = len(means_of_batches)
    if n>3: # do at least 3 batches
      t_value = scipy.stats.t.ppf(1-0.05/2, n-1)
      width = 2*t_value*np.std(means_of_batches)/math.sqrt(n)
    mean = np.mean(means_of_batches)
  CI = [mean-width/2, mean+width/2]
  return {'mean':mean,'CI':CI, 'width':width,'batches':n}

results_dict = CI(neighborhood[best_solution])
print(results_dict)

#%%
"""
d. 
Calculate each 10th percentile of the waiting time of each patient, and put them in a plot with 
the patient number on the x-axis.

"""
