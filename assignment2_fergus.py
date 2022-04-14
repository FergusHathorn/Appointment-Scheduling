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

# #individual_schedule = [1,0,0,2,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]
individual_schedule = [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]
#schedule = schedule_to_dict(individual_schedule)

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
            #waiting_times.append(max(0,finish_time-schedule[patient]))
            waiting_times.append(max(0,sim_time-schedule[patient]))
            #finish_time = sim_time+appointment_lengths[patient]
            finish_time = sim_time+appointment_lengths[patient]
            #sim_time = max(finish_time, schedule[patient])
            
        tardiness.append(max(finish_time-params['sim_length'],0))
        waiting.append(waiting_times)
        appointment_lengths = None
        
    mean_waiting_time = np.mean(waiting)
    mean_tardiness = np.mean(tardiness)
    objective_value = 2*mean_tardiness + mean_waiting_time
    
    return objective_value, mean_waiting_time, mean_tardiness

def CI(sched):
    obj_batches = []  # list of the means of the objective value from each batch of 100 simulations
    wait_batches = []
    tardiness_batches = []
    
    mean_obj = 1
    mean_wait = 1
    mean_tardiness = 1
    
    width = [100000,100000,100000]
    
    while max(width[0]/mean_obj,width[1]/mean_wait,width[2]/mean_tardiness) > 0.20:
      obj,wait,tardiness=simulate(sched,simulations=100)
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
    print('Simulation batches: {:.2f}'.format(n))
    #return results_dict

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


"""
Understanding of sim-opt:
    create a neighborhood of solutions
    choose an initial state (maybe the individual schedule)
    randomly choose another neighbor
    simulate the same consulting times on each schedule
    'winner stays on'
    count how many times each schedule is simulated and also adjust the average score for each schedule

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
        index = random.choice([index for index,n_patients in enumerate(schedule) if n_patients > 0 and index not in tuple([0,34,35])]) 
        # pick a new interval for selected patient:
        
        #if index != 0 and index != 35:
            #index2 = random.choice([-1,1])+index
        if index != 1 and index != 33:
            index2 = random.choice([-1,1])+index
        elif index == 1:
            index2 = index+1
        elif index == 33:
            index2 = index-1
        
        '''
        elif index == 0:
            index2 = index+1
        else:
            index2 = index-1
        '''
        
        #index2 = random.choice([i for i in range(1,34) if i!=index]) # possibly add condition for 2s
            
        neighbour[index] -= 1
        neighbour[index2] += 1
        
        consec_ones = False
        consec_zeros = False
        for i in range(len(neighbour)-3):
            if sum(neighbour[i:i+3]) > 2:
                consec_ones = True
                break
        for i in range(len(neighbour)-4):
            if sum(neighbour[i:i+4]) == 0:
                consec_zeros = True
                break
        if not consec_ones and not consec_zeros:
            not_acceptable = False

    return neighbour
    
#%%
start_time=time.time()
scores = {tuple(individual_schedule): {'count': 0, 'mean': 0, 'sum': 0}} # save n(x), mean objective value and sum of objective values
current = individual_schedule
n_jumps = 0
jump = []
sims = 10
budget = 1000000/(2*sims)
while budget > 0:
        
    neighbour = get_neighbour(current)
    primary_objs,neighbour_objs = 0,0
    for sim in range(sims):
        appointment_lengths = [apt() for i in range(params['patients'])]
        _,_,primary_obj = simopt(current,appointment_lengths=appointment_lengths,simulations=1)
        _,_,neighbour_obj = simopt(neighbour,appointment_lengths=appointment_lengths,simulations=1)
        primary_objs+=primary_obj
        neighbour_objs+=neighbour_obj
    mean_primary_obj = primary_objs/sims
    mean_neighbour_objs = neighbour_objs/sims
    
    if tuple(neighbour) not in scores:
        scores[tuple(neighbour)] = {'count': 0, 'mean': 0, 'sum': 0}
        
    scores[tuple(current)]['count'] += 1
    scores[tuple(neighbour)]['count'] += 1
    scores[tuple(current)]['sum'] += primary_obj
    scores[tuple(neighbour)]['sum'] += neighbour_obj
    scores[tuple(current)]['mean'] = scores[tuple(current)]['sum']/scores[tuple(current)]['count']
    scores[tuple(neighbour)]['mean'] = scores[tuple(neighbour)]['sum']/scores[tuple(neighbour)]['count']
    if scores[tuple(neighbour)]['mean'] < scores[tuple(current)]['mean']:
        jump.append(1)
        current = neighbour.copy()
    else:
        jump.append(0)
    budget -= 1

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
CI(optimal_schedule)
CI(optimal_schedule2)

#%%
# simulating
neighbors = neighborhood_size
budget = 500000
#primary = random.choice(range(neighbors))
primary = 0
scores = {i:[0,0] for i in range(neighbors)}   # saving sum of objective values and n(x)
while budget > 0:
    appointment_lengths = [apt() for i in range(params['patients'])]
    neighbor_obj,_,_ = simulate(neighborhood[random_neighbor],appointment_lengths)
    primary_obj,_,_ = simulate(neighborhood[primary],appointment_lengths)
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



'''
print('Mean objective value: {:.2f}'.format(results_dict['mean'][0]))
print('CI: {}'.format(results_dict['CI'][0]))
print('CI is {:.2f}% of the mean'.format(np.divide(100*results_dict['width'][0],results_dict['mean'][0])))
print('Simulation batches: {:.2f}'.format(results_dict['batches']))
'''
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
#%%
"""
b.
Implement the sim-opt algorithm discussed during the lectures and use it to find the best 
possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting 
time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate 
neighborhood and explain your choice also in the  report.

"""
start_time = time.time()

neighborhood_time = time.time()
print('Neighborhood generated')
print('Sim-opt in progress...')
        
# simulating
slots=list(range(len(individual_schedule)))
blank_schedule = np.zeros(len(individual_schedule))
neighborhood=[individual_schedule]
budget = 500000
#primary = random.choice(range(neighbors))
primary = 0
#scores = {i:[0,0] for i in range(neighbors)}
scores = {0:[0,0]}
while budget > 0:
    not_acceptable=True
    while not_acceptable:
        minus_one = random.choice([i for i,v in enumerate(neighborhood[primary]) if v])
        plus_one = random.choice(list(range(36)))
        neighbor = neighborhood[primary].copy()
        neighbor[minus_one] -= 1
        neighbor[plus_one] += 1
        if tuple(neighbor) not in neighborhood:
            if neighbor[0] == 1 and sum(neighbor[-2:]) != 0:
                consec_ones = False
                consec_zeros = False
                for i in range(len(neighbor)-2):
                    if sum(neighbor[i:i+3]) == 3:
                        consec_ones = True
                        break
                for i in range(len(neighbor)-3):
                    if sum(1-j for j in neighbor[i:i+4]) == 4:
                        consec_zeros = True
                        break
                if not consec_ones and not consec_zeros:
                    #if tuple(proposed_schedule) not in neighborhood:
                    neighborhood.append(neighbor)  
                    random_neighbor_idx = len(neighborhood)-1
                    existing_neighbor=False
                    not_acceptable = False
        else:
            random_neighbor_idx = [i for i,v in enumerate(neighborhood) if v == tuple(neighbor)]
            existing_neighbor=True
                
            
    appointment_lengths = [apt() for i in range(params['patients'])]
    neighbor_obj,_,_ = simulate(neighborhood[random_neighbor_idx],appointment_lengths)
    primary_obj,_,_ = simulate(neighborhood[primary],appointment_lengths)
    scores[primary][0] += 1
    scores[primary][1] += primary_obj
    if existing_neighbor:
        scores[random_neighbor_idx][0]+=1
        scores[random_neighbor_idx][1]+=neighbor_obj
    else:
        scores.update({random_neighbor_idx:[1,neighbor_obj]})
    if scores[primary][1]/scores[primary][0] < scores[random_neighbor_idx][1]/scores[random_neighbor_idx][0]:
    #if primary_obj < neighbor_obj:
        primary = random_neighbor_idx
    budget -= 1

most_frequent_solution = np.argmax([scores[i][0] for i in scores])

sim_opt_time = time.time()

print('Time for generating neighborhood: {:.2f}'.format(neighborhood_time-start_time))
print('Time for sim-opt: {:.2f}'.format(sim_opt_time-neighborhood_time))
print('Total time: {:.2f}'.format(sim_opt_time-start_time))