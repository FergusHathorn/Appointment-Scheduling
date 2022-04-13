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
import random
import scipy.stats
import statistics
from statistics import mode
from statistics import mean
import matplotlib.pyplot as plt

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

#individual_schedule = [1,0,0,2,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]
individual_schedule = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]

schedule = {i: (i-1)*params['appt_block_length'] for i in range(1,params['patients']+1)} # evenly spaced schedule

def simulate(schedule,simulations, params):
    tardiness = []
    waiting = []
    for i in range(simulations):
        waiting_times = []
        finish_times = []
        
        sim_time = 0 # start at t=0
        finish_time = 0 # start at 0
        for patient in schedule:
            waiting_times.append(max(0,finish_time-schedule[patient])) # if no wait, then append 0, else append the time between previous finish and scheduled start
            finish_time = sim_time+apt() # calculate new finish time, which is current sim time + appointment length
            finish_times.append(finish_time)
            sim_time = max(finish_time, schedule[patient]) # move simulation to whatever is last, the finish time (just calculated) or the next patient scheduled
            
        tardiness.append(max(sim_time-params['sim_length'],0)) # if finished after 180 min, append the time at which the last patient finished their appt
        waiting.append(waiting_times)
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    
    return mean_waiting_time, mean_tardiness, np.array(waiting)

def get_objective(schedule,params=params,simulations=100):
    '''
    input: 
        the schedule of form (1,0,0,1,... etc) in intervals of 5 minutes. 1 represents the start
        of a proposed appointment, 0 means that an appointment is not scheduled to start in that period.
    
    output:
        objective value = 2x(mean tardiness) + mean waiting time
    '''
    start_times = []
    for i in range(len(schedule)):
        if schedule[i]>0:
            for j in range(int(schedule[i])):
                start_times.append(i)
    
    #start_times = [i for i,v in enumerate(schedule) if v] # index of scheduled appointment start time
    schedule_dict = {i: start_times[i]*params['interval'] for i in range(params['patients'])} # dictionary of scheduled start time per patient

    # simulating the schedule
    wait,tardiness,waiting_array = simulate(schedule_dict,simulations,params)
    
    return 2*wait + tardiness,wait,tardiness,waiting_array

def CI(sched):
    obj_batches = []
    wait_batches = []
    tardiness_batches = []
    waiting_array = []
    
    mean_obj = 1
    mean_wait = 1
    mean_tardiness = 1
    
    width = [100000,100000,100000]
    
    while max(width[0]/mean_obj,width[1]/mean_wait,width[2]/mean_tardiness) > 0.05:
      appointment_lengths = [apt() for i in range(params['patients'])] 
      obj,wait,tardiness,wait_array=get_objective(sched,simulations=100)
      waiting_array.append(wait_array)
      obj_batches.append(obj)
      wait_batches.append(wait)
      tardiness_batches.append(tardiness)
      n = len(obj_batches)
      if n>3: # do at least 3 batches
        t_value = scipy.stats.t.ppf(1-0.05/2, n-1)
        width = [2*t_value*np.std(obj_batches)/math.sqrt(n),2*t_value*np.std(wait_batches)/math.sqrt(n),2*t_value*np.std(tardiness_batches)/math.sqrt(n)]
      mean_obj = np.mean(obj_batches)
      mean_wait = np.mean(wait_batches)
      mean_tardiness = np.mean(tardiness_batches)
  
    CI_obj = [mean_obj-width[0]/2, mean_obj+width[0]/2]
    CI_wait = [mean_wait-width[1]/2, mean_wait+width[1]/2]
    CI_tardiness = [mean_tardiness-width[2]/2, mean_tardiness+width[2]/2]
    return {'mean':[mean_obj, mean_wait, mean_tardiness],'CI':[CI_obj,CI_wait,CI_tardiness], 'width':width,'batches':n}, waiting_array

'''
finish_time = sim_time+params['interval']*round(apt()/params['interval'])
'''

print(CI(individual_schedule)[0])

#print(simulate(schedule,10000, params))

#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""
start_time = time.time()

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
            finish_time = sim_time+appointment_lengths[patient]
            #finish_time = sim_time+params['interval']*round(appointment_lengths[patient]/params['interval'])
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
    #start_times = [i for i,v in enumerate(schedule) if v] # index of scheduled appointment start time
    start_times = []
    for i in range(len(schedule)):
        if schedule[i]>0:
            for j in range(int(schedule[i])):
                start_times.append(i)
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
neighborhood_size = 100
while len(neighborhood) <= neighborhood_size:
    appt_slots = random.sample(slots,12)
    #appt_slots = random.choices(slots,k=12) # for multiple appts on one time
    if 0 in appt_slots and len([i for i in appt_slots if i >= 34]) <= 0: # len([i for i in appt_slots if i < 18]) >= 6 and
        proposed_schedule = blank_schedule.copy()
        for prop in appt_slots:
            proposed_schedule[prop]+=1
        #proposed_schedule[appt_slots]=1
        if tuple(proposed_schedule) not in neighborhood:
            #neighborhood.append(list(proposed_schedule))
            consec_ones = False
            consec_zeros = False
            for i in range(len(proposed_schedule)-2):
                if sum(proposed_schedule[i:i+3]) == 3:
                    consec_ones = True
                    break
            for i in range(len(proposed_schedule)-3):
                if sum(1-j for j in proposed_schedule[i:i+4]) == 4:
                    consec_zeros = True
                    break
            if not consec_ones and not consec_zeros:
                #if tuple(proposed_schedule) not in neighborhood:
                neighborhood.append(list(proposed_schedule))     

neighborhood_time = time.time()
print('Neighborhood generated')
print('Sim-opt in progress...')
        
# simulating
neighbors = neighborhood_size
budget = 500000
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
    if scores[primary][1]/scores[primary][0] < scores[random_neighbor][1]/scores[random_neighbor][0]:
    #if primary_obj < neighbor_obj:
        primary = random_neighbor
    budget -= 1

most_frequent_solution = np.argmax([scores[i][0] for i in scores])

sim_opt_time = time.time()

print('Time for generating neighborhood: {:.2f}'.format(neighborhood_time-start_time))
print('Time for sim-opt: {:.2f}'.format(sim_opt_time-neighborhood_time))
print('Total time: {:.2f}'.format(sim_opt_time-start_time))
    
#%%
"""
c. 
Calculate for the optimal schedule the objective and its confidence interval. Simulate often 
enough to get a small confidence interval. Good solutions are ranked and at maximum 1 point is 
given according to the ranking.

"""

results_dict,full_waiting_array = CI(neighborhood[most_frequent_solution])

print('Mean objective value: {:.2f}'.format(results_dict['mean'][0]))
print('CI: {}'.format(results_dict['CI'][0]))
print('CI is {:.2f}% of the mean'.format(np.divide(100*results_dict['width'][0],results_dict['mean'][0])))
print('Simulation batches: {:.2f}'.format(results_dict['batches']))

#%%
"""
d. 
Calculate each 10th percentile of the waiting time of each patient, and put them in a plot with 
the patient number on the x-axis.

"""
wait_times_by_sim = []
for sim in full_waiting_array:
    for sub_sim in sim:
        wait_times_by_sim.append(sub_sim)
patient_waiting_times = [[i[k] for i in wait_times_by_sim] for k in range(12)]

pcts = []
for j in [10,20,30,40,50,60,70,80,90]:
    pcts.append([np.percentile(i,j) for i in patient_waiting_times])
    
for i in pcts:
    plt.plot(i)
    plt.title('Percentile waiting times per patient', fontsize=15)
    plt.xlabel('Patient', fontsize=15)
    plt.ylabel('Waiting time (minutes)', fontsize=15)
'''
plt.bar(list(range(1,13)),pct_10)
plt.title('10th percentile waiting times per patient', fontsize=15)
plt.xlabel('Patient', fontsize=15)
plt.ylabel('Waiting time (minutes)', fontsize=15)
'''

