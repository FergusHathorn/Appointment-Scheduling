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
    for i in range(10000):
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

print(simulate(schedule,10000, params))

#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""

individual_schedule = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]

def objective(schedule,simulations=100000,params=params):
    '''
    input: 
        the schedule of form (1,0,0,1,... etc) in intervals of 5 minutes. 1 represents the start
        of a proposed appointment, 0 means that an appointment is not scheduled to start in that period.
    
    output:
        objective value = 2x(mean tardiness) + mean waiting time
    '''
    start_times = [i for i,v in enumerate(schedule) if v] # index of scheduled appointment start time
    schedule_dict = {i: start_times[i]*params['interval'] for i in range(0,params['patients'])} # dictionary of scheduled start time per patient
    
    # simulating the schedule
    wait,tardiness = simulate(schedule_dict,simulations,params)
    
    return 2*wait + tardiness
    
print(objective(individual_schedule)) 

#%%
"""
c. 
Calculate for the optimal schedule the objective and its confidence interval. Simulate often 
enough to get a small confidence interval. Good solutions are ranked and at maximum 1 point is 
given according to the ranking.

"""
#%%
"""
d. 
Calculate each 10th percentile of the waiting time of each patient, and put them in a plot with 
the patient number on the x-axis.

"""