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

tardiness = []
waiting = []
for i in range(100000):
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
        


#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""
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