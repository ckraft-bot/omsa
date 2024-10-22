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

#---------------------------------------------- Simple: single queue ----------------------------------------------#
"""
The flow of this simple simulation follows a single queue system for both the ID check stations and scanners.

1. **Arrival**:  
   - Passengers arrive at the airport based on a Poisson process (random time intervals) and are immediately placed into a **single queue** for the ID check stations.

2. **ID Check Queue**:
   - Passengers wait in line to request access to an available **ID checker** resource (one of the 2 ID check stations).
   - Once a passenger's request is granted (meaning an ID checker becomes available), they go through the ID check process. The time taken for this process is drawn from an exponential distribution (with a mean of 0.75 minutes).

3. **Scanner Queue**:
   - After completing the ID check, passengers are placed in another **single queue** for the scanners (there are 2 scanners).
   - They then request access to an available scanner. Once granted, the scanning process takes place, with a time drawn from a uniform distribution between 0.5 and 1 minute.

4. **Exit**:
   - After completing both the ID check and the scan, the passenger exits the system, and their total time in the system (from arrival to the end of scanning) is recorded.

### Key Points:
- **Single Queue for ID Check**: All passengers share the same queue for the ID check stations, which means they wait for the first available ID checker.
- **Single Queue for Scanners**: Similarly, after the ID check, all passengers wait in a single queue for the first available scanner.
  
This single queue approach ensures fairness in the system, as passengers are served based on their order of arrival rather than being assigned to a specific ID checker or scanner. This also optimizes resource usage by minimizing idle time for ID checkers and scanners.

So, the flow is as follows:
- **One queue for ID check** → **One queue for scanners** → **Exit**.

This flow effectively simulates a realistic airport security process where passengers are routed from one checkpoint (ID check) to another (scanner) without being tied to any specific station but rather the next available one.
"""

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
    
    def id_check(self, passenger):
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
        yield env.process(airport.id_check(number))
    
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
num_id_checkers = 2
num_scanners = 2
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
print('------------------------------------Simple------------------------------------')
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


#---------------------------------------------- Complex: dual queues ----------------------------------------------#
"""
The flow of this complex simulation follows a multi queue system for both the ID check stations and scanners.

1. **Arrival**:  
   - Passengers arrive at the airport based on a Poisson process, which generates random arrival times. Upon arrival, each passenger enters the ID check queue. In this simulation, there are **separate queues for each ID checker**.

2. **ID Check Queue**:
   - Passengers are assigned to a random ID check queue (one of the 2 ID check stations). Each ID checker has an independent queue.
   - The passenger waits in their assigned ID checker's queue for the next available staff member.
   - Once they reach the front of the queue, they undergo the ID check, which takes a time drawn from an **exponential distribution** with a mean of 0.75 minutes.

3. **Scanner Queue**:
   - After completing the ID check, the passenger enters a separate scanner queue. Similar to the ID check process, each scanner has an independent queue (there are 2 scanners).
   - Passengers are assigned to a random scanner queue and wait for the next available scanner.
   - Once a scanner becomes available, the scanning process begins. This time is drawn from a **uniform distribution** between 0.5 and 1 minute.

4. **Exit**:
   - After completing the ID check and scanning process, the passengers exit the system, and their total time spent in the system (from arrival to the end of scanning) is recorded.

### Key Points:
- **Two Independent Queues**: There are two separate sets of queues—one for ID check stations and one for scanners. Each passenger is assigned to a specific queue for both the ID check and the scanner.
- **Queue Assignment**: Passengers do not stay in a single queue for all ID checkers or scanners. Instead, they are randomly assigned to a specific queue (either for ID checking or scanning), simulating a more decentralized queuing system.
- **Random Service Time**: 
   - ID check times are drawn from an exponential distribution, which simulates varying lengths of time for ID checks.
   - Scanning times are drawn from a uniform distribution, modeling the relatively consistent but varied duration of security scanning.
  
This two-queue model simulates a more complex and realistic airport security process where passengers are not waiting in a single line but are instead distributed across two independent queues for both ID check and scanning stations.
"""

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
    """ID check queue and security scanning with separate queues"""
    def __init__(self, env, num_id_checkers, num_scanners):
        self.env = env
        self.id_checker_queue = [sy.Resource(env, 1) for _ in range(num_id_checkers)]  # Separate queues for ID checkers
        self.scanner_queue = [sy.Resource(env, 1) for _ in range(num_scanners)]  # Separate queues for scanners

    def id_check(self, passenger, queue_id):
        """Passenger uses a specific ID checker queue"""
        service_time = random.expovariate(1 / 0.75)  # Exponential distribution with mean 0.75 minutes
        id_service_times.append(service_time)
        yield self.env.timeout(service_time)

    def scan(self, passenger, queue_id):
        """Passenger uses a specific scanner queue"""
        scan_time = random.uniform(0.5, 1)  # Uniform distribution between 0.5 and 1 minute
        scan_times.append(scan_time)
        yield self.env.timeout(scan_time)

def Passenger(env, number, airport):
    """Passengers arrive, queue for ID check, then queue for scanners."""
    global total_wait_time, total_passengers
    
    Arrivaltime = env.now
    
    # Pick a random queue for the ID check
    id_checker_queue_id = random.randint(0, len(airport.id_checker_queue) - 1)
    with airport.id_checker_queue[id_checker_queue_id].request() as request:
        yield request
        yield env.process(airport.id_check(number, id_checker_queue_id))

    # After ID check, pick a random queue for the scanner
    scanner_queue_id = random.randint(0, len(airport.scanner_queue) - 1)
    with airport.scanner_queue[scanner_queue_id].request() as request:
        yield request
        yield env.process(airport.scan(number, scanner_queue_id))

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
    """Run the airport security simulation with two separate queues"""
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
num_id_checkers = 2
num_scanners = 2
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
print('------------------------------------Complex------------------------------------')
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


