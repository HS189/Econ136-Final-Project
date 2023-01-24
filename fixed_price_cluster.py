from scipy.stats import pareto, lognorm, truncexpon, truncnorm
from collections import deque
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

from compute_cluster import Job, job_bid_value
import power_supply

cur_time = 0
# jobs = np.load('jobs_file.npy', allow_pickle=True)
with open('google_job_cache.pkl', 'rb') as f:
	jobs = pickle.load(f)
auction_interval = 60*6
auction_interval_min = auction_interval / 60

queued_jobs = {}

auction_times = []
green_prices = []
green_energy_usage = []
brown_avg_price = []
brown_energy_usage = []
green_supply = []
num_queued_jobs = []

# first fit pricing - whoever comes in first gets run
# 0.1 for solar only, 0.05 for solar + wind
fixed_price = 0.42 # per 6min auction freq in google



# 446, 452 for solar + wind
available_minutely_energy = 72000 / 6. # per 6min auction freq for google


for i in range(len(jobs)):
	if i % 1000 == 0:
		print('job #: {}'.format(i))
	cur_job = jobs[i]
	if cur_job.submission < cur_time + auction_interval:
		queued_jobs[cur_job.id] = cur_job
	else:
		cur_time += auction_interval

		auction_times.append(cur_time)

		num_queued_jobs.append(len(queued_jobs))

		# run auction to allocate green energy
		available_green_energy = power_supply.monthlong_solar_power(cur_time)
		green_supply.append(available_green_energy)

		queued_job_ids = sorted(list(queued_jobs.keys())) # process lowest ids first (submitted earliest)

		cur_minute_energy = available_minutely_energy
		cur_minute_green_usage = 0
		cur_minute_brown_usage = 0

		for j_id in queued_job_ids:
			if cur_minute_energy <= 0:
				break

			pending_job = queued_jobs[j_id]
			job_val = job_bid_value(pending_job, cur_time)
			if job_val >= fixed_price and cur_minute_energy > 0:
				# run jobs that are willing to pay fixed price
				cur_minute_energy -= 1
				if available_green_energy > 0:
					available_green_energy -= 1
					cur_minute_green_usage += 1
				else:
					cur_minute_brown_usage += 1

				pending_job.elapsed_time += auction_interval
				pending_job.total_payments += fixed_price

				if pending_job.duration - pending_job.elapsed_time <= 0:
				# job is done
					pending_job.end_time = cur_time + auction_interval # job finishes at END of period
					del queued_jobs[j_id]

		# # update disutility
		# for j_id in queued_job_ids:
		# 	pending_job = queued_jobs[j_id]
		# 	disutility = job_bid_value(pending_job, cur_time) - pending_job.instrinsic_minute_value
		# 	pending_job.total_disutility += disutility

		# for win_id in winning_ids:
		# 	if queued_jobs[win_id].duration - queued_jobs[win_id].elapsed_time <= 0:
		# 		# job is done
		# 		queued_jobs[win_id].end_time = cur_time + auction_interval # job finishes at END of period
		# 		del queued_jobs[win_id]

		green_energy_usage.append(cur_minute_green_usage)
		brown_energy_usage.append(cur_minute_brown_usage)

# total_revenue = 0
# total_value = 0
# total_runtime = 0
# finished_num_jobs = 0

# for i in range(len(jobs)):
# 	cur_job = jobs[i]
# 	if cur_job.duration - cur_job.elapsed_time <= 0:
# 		# finished job
# 		total_revenue += cur_job.total_payments
# 		total_runtime += cur_job.elapsed_time # move inside if statement for finished jobs
# 		finished_num_jobs += 1

# 		# what job would've paid to be run in the timeframe
# 		job_total_value = (cur_job.total_disutility + cur_job.instrinsic_minute_value * (cur_job.est_duration / 60.))
# 		total_value += job_total_value

# print(finished_num_jobs)
# print(total_revenue, total_value)

total_revenue = 0
total_value = 0
total_runtime = 0
total_disutility = 0

low_pri_jobs = 0
low_pri_checkpoints = 0
low_pri_delays = []
low_pri_payments = []

mid_pri_jobs = 0
mid_pri_checkpoints = 0
mid_pri_delays = []
mid_pri_payments = []

high_pri_jobs = 0
high_pri_checkpoints = 0
high_pri_delays = []
high_pri_payments = []


for i in range(len(jobs)):
	cur_job = jobs[i]
	cur_job.total_disutility = 0
	minute_delay = cur_job.end_time -  (cur_job.submission + cur_job.duration)
	minute_delay = int(minute_delay / 60)

	if cur_job.duration - cur_job.elapsed_time <= 0:
		# finished job
		if cur_job.priority == 'low':
			for mi in range(1, minute_delay):
					# true disutility
				cur_job.total_disutility += 0.01 * np.log(mi + np.e)

			low_pri_jobs += 1
			low_pri_checkpoints += cur_job.num_restarts
			low_pri_delays.append(cur_job.end_time -  (cur_job.submission + cur_job.duration))
			low_pri_payments.append(cur_job.total_payments)
		elif cur_job.priority == 'mid':
			for mi in range(1, minute_delay):
				cur_job.total_disutility += 0.1 + 0.01 * mi
			mid_pri_jobs += 1
			mid_pri_checkpoints += cur_job.num_restarts
			mid_pri_delays.append(cur_job.end_time -  (cur_job.submission + cur_job.duration))
			mid_pri_payments.append(cur_job.total_payments)
		else:
			for mi in range(1, minute_delay):
				cur_job.total_disutility += 1 + 0.05*mi*mi
			high_pri_jobs += 1
			high_pri_checkpoints += cur_job.num_restarts
			high_pri_delays.append(cur_job.end_time -  (cur_job.submission + cur_job.duration))
			high_pri_payments.append(cur_job.total_payments)

	total_revenue += cur_job.total_payments
	total_runtime += cur_job.elapsed_time # move inside if statement for finished jobs

	# what job would've paid to be run in the timeframe
	job_total_value = (cur_job.total_disutility + cur_job.instrinsic_minute_value * (cur_job.est_duration / 60.))
	total_value += job_total_value
	total_disutility += cur_job.total_disutility

print(total_revenue, total_value)
print('total disutility', total_disutility)
print('num checkpoints: {} {} {}'.format(low_pri_checkpoints, mid_pri_checkpoints, high_pri_checkpoints))
print('# low pri: {}, # mid pri: {}, # high pri: {}'.format(low_pri_jobs, mid_pri_jobs, high_pri_jobs))
print('low pri delay: {}, mid pri delay: {}, high pri delay: {}'.format(np.mean(low_pri_delays), np.mean(mid_pri_delays), np.mean(high_pri_delays)))
print('low pri cost: {}, mid pri cost: {}, high pri cost: {}'.format(np.mean(low_pri_payments), np.mean(mid_pri_payments), np.mean(high_pri_payments)))
print('green energy total usage: {}'.format(np.sum(green_energy_usage)))
print('total available green energy: {}'.format(np.sum(green_supply)))

print('brown energy total usage: {}'.format(np.sum(brown_energy_usage)))


total_runtime /= 60 # in minutes
print('gross execution time:', total_runtime)

print('green energy total usage: {}'.format(np.sum(green_energy_usage)))
print('brown energy total usage: {}'.format(np.sum(brown_energy_usage)))

plt.figure(figsize=(8,6))
plt.plot(auction_times, green_energy_usage)
plt.title('Green Usage')
plt.show()
plt.plot(auction_times, brown_energy_usage)
plt.title('brown Usage')
plt.show()
plt.plot(auction_times, num_queued_jobs)
plt.title('Num Queued Jobs')
plt.show()




