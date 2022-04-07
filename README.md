Simulate in python an appointment schedule of length 3 hours, 5-minute intervals, and 12 patients, with lognormally distributed service times with mean 13 and standard deviation 5. (Make sure your distribution has the right parameters.) No no-shows (all patients show up), patients are on time.

a. Use simulation to estimate the expected average patient waiting time and the tardiness of the doctor (the time the doctor is busy after the end of the appointment block), first for the individual schedule (where patients are equally spaced).

b. Implement the sim-opt algorithm discussed during the lectures and use it to find the best possible schedule, counting the doctor's tardiness 2x heavier than the average patient waiting time. You have a simulation budget of 10000 simulations. Construct yourself an appropriate neighborhood and explain your choice also in the  report.

c. Calculate for the optimal schedule the objective and its confidence interval. Simulate often enough to get a small confidence interval. Good solutions are ranked and at maximum 1 point is given according to the ranking. 

d. Calculate each 10th percentile of the waiting time of each patient, and put them in a plot with the patient number on the x-axis.
