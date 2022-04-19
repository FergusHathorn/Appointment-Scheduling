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

random.seed(10)
#%%
# """
# a.
# Use simulation to estimate the expected average patient waiting time and the tardiness 
# of the doctor (the time the doctor is busy after the end of the appointment block), first 
# for the individual schedule (where patients are equally spaced).

# """

# # calculating mu and sigma: https://en.wikipedia.org/wiki/Log-normal_distribution
mu_of_normal = math.log(13**2/np.sqrt(13**2 + 5**2))
sigma_of_normal = np.sqrt(math.log(1 + 5**2./13**2))

# function to return an appointment length
def apt(mu = mu_of_normal, sig = sigma_of_normal):
    return np.random.lognormal(mean=mu,sigma=sig)


params = {
    'patients':12,
    'interval':5,
    'sim_length':3*60,
    'appt_block_length':15
    }

def schedule_to_dict(schedule):
    '''
    input: 
        the schedule of form (1,0,0,1,... etc) in intervals of 5 minutes.
    output:
        dictionary with patients and start times: {1: 0, 2: 15, 3: 30.....}
    '''
    start_times = []
    for i in range(len(schedule)):
        if schedule[i]>0:
            for j in range(int(schedule[i])):
                start_times.append(i)
    schedule_dict = {i: start_times[i]*params['interval'] for i in range(params['patients'])} # dictionary of scheduled start time per patient
    return schedule_dict

individual_schedule = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]

def simulate(schedulelist,appointment_lengths = None,simulations=1,params=params):
    schedule = schedule_to_dict(schedulelist)
    tardiness = []
    waiting = []
    
    for i in range(simulations):
        if appointment_lengths == None:
            appointment_lengths = [apt() for i in range(params['patients'])]
            
        waiting_times = []
        sim_time = 0
        finish_time = 0
        for patient in schedule:
            sim_time = max(finish_time, schedule[patient])
            waiting_times.append(max(0,sim_time-schedule[patient]))
            finish_time = sim_time+appointment_lengths[patient]
            
        tardiness.append(max(finish_time-params['sim_length'],0))
        waiting.append(waiting_times)
        appointment_lengths = None
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    objective_value = 2*mean_tardiness + mean_waiting_time
    
    return objective_value, mean_waiting_time, mean_tardiness, waiting

def CI(sched, maxwidth=0.2):
    obj_batches = []  # list of the means of the objective value from each batch of 100 simulations
    wait_batches = []
    tardiness_batches = []
    
    mean_obj = 1
    mean_wait = 1
    mean_tardiness = 1
    
    width = [100000,100000,100000]
    
    while max(width[0]/mean_obj,width[1]/mean_wait,width[2]/mean_tardiness) > maxwidth:
      obj,wait,tardiness,_=simulate(sched,simulations=100)
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
    
    results_dict={'mean':[mean_obj, mean_wait, mean_tardiness],'CI':[CI_obj,CI_wait,CI_tardiness], 'width':width,'batches':n}
    print('Mean objective value: {:.2f}'.format(results_dict['mean'][0]))
    print('CI: {}'.format(results_dict['CI'][0]))
    print('CI is {:.2f}% of the mean'.format(np.divide(100*width[0],mean_obj)))
    print('Mean waiting time: {:.2f}'.format(mean_wait))
    print('CI: {}'.format(CI_wait))
    print('CI is {:.2f}% of the mean'.format(np.divide(100*width[1],mean_wait)))
    print('Mean tardiness: {:.2f}'.format(mean_tardiness))
    print('CI: {}'.format(CI_tardiness))
    print('CI is {:.2f}% of the mean'.format(np.divide(100*width[2],mean_tardiness)))
    print('Simulation batches:', n)
    #return results_dict,wait_batches

'''
finish_time = sim_time+params['interval']*round(apt()/params['interval'])
'''
CI(individual_schedule)


#print(simulate(schedule,10000, params))

#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""

def get_neighbour(schedule):
    ''' generate a neighbour from the neighbourhood of the current schedule
    input: schedule of the form [1,0,0,1,0,...]
    '''
    neighbour = schedule.copy()
    # pick a random interval where there is a patient:

    not_acceptable = True
    while not_acceptable:
        neighbour = schedule.copy()
        index = random.choice([index for index,n_patients in enumerate(schedule) if n_patients > 0 and index not in tuple([34,35])]) 
        
        # pick a new interval for selected patient:
        if index != 0 and index != 33:
            index2 = random.choice([-1,1]) + index
        elif index == 0:
            index2 = index+1
        elif index == 33:
            index2 = index-1
            
        neighbour[index] -= 1
        neighbour[index2] += 1
        
        consec_ones = False  # are there 3 patients in a 15 minute interval?
        consec_zeros = False  # are there 4 consecutive slots empty?
        for i in range(8,len(neighbour)-3):
            if sum(neighbour[i:i+3]) > 2:
                consec_ones = True
                break
        for i in range(len(neighbour)-10):
            if sum(neighbour[i:i+4]) == 0:
                consec_zeros = True
                break
        if not consec_ones and not consec_zeros:
            not_acceptable = False

    return neighbour
    
#%%
start_time=time.time()
scores = {tuple(individual_schedule): {'count': 0, 'mean': 0, 'sum': 0, 'returns':0}} # save n(x), mean objective value and sum of objective values
current = individual_schedule
budget = 1000000
while budget > 0:
        
    neighbour = get_neighbour(current)
    if tuple(neighbour) not in scores:
        scores[tuple(neighbour)] = {'count': 0, 'mean': 0, 'sum': 0, 'returns':0}    
        sims = 2000
    else: 
        sims = 1
        
    primary_objs,neighbour_objs = 0,0
    for sim in range(sims):
        appointment_lengths = [apt() for i in range(params['patients'])]
        primary_obj,_,_,_ = simulate(current,appointment_lengths=appointment_lengths,simulations=1)
        neighbour_obj,_,_,_ = simulate(neighbour,appointment_lengths=appointment_lengths,simulations=1)
        primary_objs+=primary_obj
        neighbour_objs+=neighbour_obj
    mean_primary_obj = primary_objs/sims
    mean_neighbour_obj = neighbour_objs/sims
            
    scores[tuple(current)]['count'] += sims
    scores[tuple(neighbour)]['count'] += sims
    scores[tuple(current)]['sum'] += primary_objs
    scores[tuple(neighbour)]['sum'] += neighbour_objs
    scores[tuple(current)]['mean'] = scores[tuple(current)]['sum']/scores[tuple(current)]['count']
    scores[tuple(neighbour)]['mean'] = scores[tuple(neighbour)]['sum']/scores[tuple(neighbour)]['count']
    if scores[tuple(neighbour)]['mean'] < scores[tuple(current)]['mean']:
        current = neighbour.copy()
        scores[tuple(neighbour)]['returns'] += 1
    budget -= 2*sims

print("Total time: {:.2f}".format(time.time()-start_time))
    

#%%
schedule_with_counts = {schedule:scores[schedule]['count'] for schedule in scores}
optimal_schedule = max(schedule_with_counts,key=schedule_with_counts.get)
#optimal = np.argmax(counts)
#print(optimal,n_jumps)

schedule_with_objective = {schedule:scores[schedule]['mean'] for schedule in scores if scores[schedule]['count'] >= 8}
optimal_schedule2 = min(schedule_with_objective,key=schedule_with_objective.get)
scores[optimal_schedule2]
scores[optimal_schedule]
print('-------------------')
CI(optimal_schedule, maxwidth=0.05)
print('-------------------')
CI(optimal_schedule2, maxwidth=0.05)
print('-------------------')
CI(individual_schedule)
# #%%
# proposed = [1,0,1]+10*[0,0,1]+[0,0,0]
# proposed2 = [1,0,1,0] + [1,0,0]*9 + [1,0,0,0,0]
# CI(proposed2, maxwidth=0.01)
# #np.savetxt("7,41.csv", np.asarray(optimal_schedule2), delimiter=",")
# #%%
# for i in range(36):
#     print(proposed2[i]==optimal_schedule[i])
#%%
"""
d. 
Calculate each 10th percentile of the waiting time of each patient, and put them in a plot with 
the patient number on the x-axis.

"""
_,_,_,waiting_times = simulate(optimal_schedule,simulations=100000)

patient_waiting_times = [[i[k] for i in waiting_times] for k in range(12)]

pcts = []
for j in [10,20,30,40,50,60,70,80,90]:
    pcts.append([np.percentile(i,j) for i in patient_waiting_times])
    
for i in pcts:
    plt.plot(i)
    plt.title('Percentile waiting times per patient', fontsize=15)
    plt.xlabel('Patient', fontsize=15)
    plt.ylabel('Waiting time (minutes)', fontsize=15)