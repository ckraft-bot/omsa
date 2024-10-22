"""
Homework 9 header 
Question 13.2
In this problem, you can simulate a simplified airport security system at a busy airport. 
Passengers arrive according to a Poisson distribution with λ1 = 5 per minute (i.e., mean interarrival rate 1 = 0.2 minutes) 
to the ID/boarding-pass check queue, where there are several id_checkers who each have exponential service time with mean rate 2 = 0.75 minutes. 
[Hint: model them as one block that has more than one resource.] 
After that, the passengers are assigned to the shortest of the several personal-check queues, 
where they go through the personal scanner (time is uniformly distributed between 0.5 minutes and 1 minute).

Use the Arena software (PC users) or Python with sy (PC or Mac users) to build a simulation of the system, 
and then vary the number of ID/boarding-pass checkers and personal-check queues to determine how many are needed to keep average wait times below 15 minutes. 
[If you’re using sy, or if you have access to a non-student version of Arena, you can use λ1 = 50 to simulate a busier airport.]
"""

import simpy as sy
import random
from statistics import mean, stdev
import numpy as np

# Set random seed for reproducibility
random.seed(1)

# Global variables for tracking statistics
total_wait_time = 0
total_passengers = 0
wait_times = []
id_service_times = []
scan_times = []
num_replications = 50  # Number of replications for the simulation

class Airport:
    """ID/boarding pass check queue and security scanning"""
    def __init__(self, env, num_id_checkers, num_scanners):
        self.env = env
        self.id_checker = sy.Resource(env, num_id_checkers)  # ID checkers
        self.scanner = sy.Resource(env, num_scanners)  # Scanners
    
    def IDcheck(self, passenger):
        """Airport staff check IDs and direct passengers to scanners."""
        service_time = random.expovariate(1 / 0.75)  # Exponential distribution with mean 0.75 minutes
        id_service_times.append(service_time)
        yield self.env.timeout(service_time)
    
    def scan(self, passenger):
        """Passengers are sent to shortest queue after IDs are checked."""
        scan_time = random.uniform(0.5, 1)  # Uniform distribution between 0.5 and 1 minute
        scan_times.append(scan_time)
        yield self.env.timeout(scan_time)

def Passenger(env, number, airport):
    """Passengers arrive, have IDs checked by staff, then go through scanners."""
    global total_wait_time, total_passengers
    
    Arrivaltime = env.now
    with airport.id_checker.request() as request:
        yield request
        yield env.process(airport.IDcheck(number))
    
    with airport.scanner.request() as request:
        yield request
        yield env.process(airport.scan(number))
    
    pass_time = env.now - Arrivaltime
    total_wait_time += pass_time
    total_passengers += 1
    wait_times.append(pass_time)

def setup(env, num_passengers, airport):
    """Setup passengers arriving at the airport"""
    for i in range(num_passengers):
        arrival_int = random.expovariate(5)  # Poisson process with λ = 5 per minute
        yield env.timeout(arrival_int)
        env.process(Passenger(env, i, airport))

def run_simulation(num_id_checkers, num_scanners, num_passengers, run_time):
    """Run the airport security simulation"""
    global total_wait_time, total_passengers, wait_times, id_service_times, scan_times
    total_wait_time = 0
    total_passengers = 0
    wait_times = []
    id_service_times = []
    scan_times = []
    
    env = sy.Environment()
    airport = Airport(env, num_id_checkers, num_scanners)
    
    env.process(setup(env, num_passengers, airport))
    env.run(until=run_time)
    
    average_wait_time = total_wait_time / total_passengers if total_passengers > 0 else 0
    return average_wait_time, wait_times, id_service_times, scan_times

# Parameters
num_id_checkers = 24
num_scanners = 15
num_passengers = 500
run_time = 600  # Simulation run time in minutes

# Running the simulation multiple times
all_wait_times = []
all_id_service_times = []
all_scan_times = []
for _ in range(num_replications):
    avg_time, rep_wait_times, rep_id_service_times, rep_scan_times = run_simulation(num_id_checkers, num_scanners, num_passengers, run_time)
    all_wait_times.extend(rep_wait_times)
    all_id_service_times.extend(rep_id_service_times)
    all_scan_times.extend(rep_scan_times)

# Calculate statistics for wait times
avg_wait_time = mean(all_wait_times) if all_wait_times else 0
max_wait_time = max(all_wait_times) if all_wait_times else 0
min_wait_time = min(all_wait_times) if all_wait_times else 0
std_dev_wait_time = stdev(all_wait_times) if len(all_wait_times) > 1 else 0

# Calculate statistics for ID service times
avg_id_service_time = mean(all_id_service_times) if all_id_service_times else 0
max_id_service_time = max(all_id_service_times) if all_id_service_times else 0
min_id_service_time = min(all_id_service_times) if all_id_service_times else 0
std_dev_id_service_time = stdev(all_id_service_times) if len(all_id_service_times) > 1 else 0

# Calculate statistics for scan times
avg_scan_time = mean(all_scan_times) if all_scan_times else 0
max_scan_time = max(all_scan_times) if all_scan_times else 0
min_scan_time = min(all_scan_times) if all_scan_times else 0
std_dev_scan_time = stdev(all_scan_times) if len(all_scan_times) > 1 else 0

# Calculate statistics for full routine times
full_routine_times = [wait + id + scan for wait, id, scan in zip(all_wait_times, all_id_service_times, all_scan_times)]
avg_routine_time = mean(full_routine_times) if full_routine_times else 0
max_routine_time = max(full_routine_times) if full_routine_times else 0
min_routine_time = min(full_routine_times) if full_routine_times else 0
std_dev_routine_time = stdev(full_routine_times) if len(full_routine_times) > 1 else 0

# Print the results
print(f'Average wait time: {avg_wait_time:.2f} minutes')
print(f'Maximum wait time: {max_wait_time:.2f} minutes')
print(f'Minimum wait time: {min_wait_time:.2f} minutes')
print(f'Standard deviation of wait time: {std_dev_wait_time:.2f} minutes')

print(f'Average ID check time: {avg_id_service_time:.2f} minutes')
print(f'Maximum ID check time: {max_id_service_time:.2f} minutes')
print(f'Minimum ID check time: {min_id_service_time:.2f} minutes')
print(f'Standard deviation of ID check time: {std_dev_id_service_time:.2f} minutes')

print(f'Average scan time: {avg_scan_time:.2f} minutes')
print(f'Maximum scan time: {max_scan_time:.2f} minutes')
print(f'Minimum scan time: {min_scan_time:.2f} minutes')
print(f'Standard deviation of scan time: {std_dev_scan_time:.2f} minutes')

print(f'Average full routine time: {avg_routine_time:.2f} minutes')
print(f'Maximum full routine time: {max_routine_time:.2f} minutes')
print(f'Minimum full routine time: {min_routine_time:.2f} minutes')
print(f'Standard deviation of full routine time: {std_dev_routine_time:.2f} minutes')
