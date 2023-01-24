from scipy.stats import pareto, lognorm, truncexpon, truncnorm
from collections import deque
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import argparse
import pickle

from auction import *
import power_supply

class Job:
	def __init__(self, submission, priority, duration, est_duration, elapsed_time, j_id):
		self.submission = submission
		self.priority = priority
		self.instrinsic_minute_value = 0
		if self.priority == 'low':
			self.instrinsic_minute_value = 0.01
		elif self.priority == 'mid':
			self.instrinsic_minute_value == 0.1
		elif self.priority == 'high':
			self.instrinsic_minute_value = 1

		self.duration = duration
		self.est_duration = est_duration
		self.elapsed_time = elapsed_time
		self.num_restarts = 0

		self.end_time = 0

		self.total_payments = 0
		self.total_disutility = 0

		self.id = j_id

np.random.seed(100) # fix seed for reproducibility

def generate_jobs(start_time, num_jobs):
	arrival = pareto.rvs(4, loc=1, size=num_jobs) * 3 * 0.5

	rate = 3 # try 6 too?
	priority = truncexpon.rvs(rate, scale=1.0/rate, size=num_jobs)
	duration = lognorm.rvs(1.634, scale=447, size=num_jobs)
	print('est duration', np.mean(duration))

	submissions = np.cumsum(arrival)
	submissions += start_time
	print(np.histogram(priority, bins=[0, 0.33, 0.66, 1]))

	lower, upper = 0.75, 1.25
	mu, sigma = 1, 0.5
	noise_factors = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=num_jobs)

	estimated_duration = duration * noise_factors

	jobs = []
	for i in range(num_jobs):
		pri = 'low'
		if priority[i] > (1./3) and priority[i] < (2./3):
			pri = 'mid'
		elif priority[i] >= (2./3):
			pri = 'high'
		jobs.append(Job(submissions[i], pri, duration[i], estimated_duration[i], 0, i))
	return jobs

def real_google_jobs():
	if os.path.exists('google_job_cache.pkl'):
		with open('google_job_cache.pkl', 'rb') as f:
			jobs = pickle.load(f)
			return jobs

	path = 'google_cluster/all_tasks.csv'
	df = pd.read_csv(path)
	jobs = []

	start_time = df.start_time.iloc[0]

	duration = np.array(df.runtime)
	lower, upper = 0.75, 1.25
	mu, sigma = 1, 0.5
	noise_factors = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=len(df))

	estimated_duration = duration * noise_factors
	num_mid = 0
	for i in range(len(df)):
		if i % 100000 == 0:
			print(f'{i}/{len(df)} iters')
		runtime = df.runtime.iloc[i]
		google_priority = df.start_priority.iloc[i]
		if google_priority < 4:
			pri = 'low'
		elif google_priority > 7:
			pri = 'high'
		else:
			pri = 'mid'

		jobs.append(Job(df.start_time.iloc[i] - start_time, pri, duration[i], estimated_duration[i], 0, i))

	with open('google_job_cache.pkl', 'wb') as f:
		pickle.dump(jobs, f)

	return jobs


def job_bid_value(job, cur_time, runtime_interval=1):
	# get a job's true value to be run for one increment (i.e 1 minute) of time
	runtime_interval = int(runtime_interval)
	true_value = 0

	remaining_minutes = (job.est_duration - job.elapsed_time) / 60.
	# jobs with 30 sec (anything < 1 min) left should still be treated like they have 1 min left
	remaining_minutes = max(remaining_minutes, 1.)

	instrinsic_minute_value = job.instrinsic_minute_value * runtime_interval

	# calculate disutility for jobs exceeding their desired completion time
	desired_end_time = job.submission + job.est_duration
	pred_end_time = cur_time + (job.est_duration - job.elapsed_time)
	num_minutes_exceeded = (pred_end_time - desired_end_time) / 60.
	if num_minutes_exceeded < 0:
		num_minutes_exceeded = 0
	num_minutes_exceeded = int(num_minutes_exceeded)

	disutility = 0
	for nm in range(num_minutes_exceeded, num_minutes_exceeded + runtime_interval):
		if job.priority == 'low':
			disutility += 0.01 * np.log(nm + np.e)
		elif job.priority == 'mid':
			disutility += 0.1 + 0.01 * nm
		elif job.priority == 'high':
			disutility += 1 + 0.05*nm*nm

	true_value = instrinsic_minute_value + disutility
	return true_value


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--solar_only', action='store_true')
	parser.add_argument('--combined', action='store_true')
	parser.add_argument('--use-google', action='store_true')
	args = parser.parse_args()

	if not (args.solar_only or args.combined):
		parser.error('No action requested, add --solar_only or --combined')

	if args.use_google:
		jobs = real_google_jobs()
	else:
		num_jobs = 175000
		jobs = generate_jobs(0, num_jobs)
		# np.save('jobs_file', jobs)
	print('loaded jobs')
	cur_time = 0
	auction_interval = 60*1 # run auction every 60 sec
	auction_interval_min = auction_interval / 60.

	green_reserve_price = 0.01 * auction_interval_min

	brown_reserve_price = 0.06 * auction_interval_min

	checkpointing_available = True

	use_real_solar = False
	use_combined_energy = False
	if args.solar_only:
		use_real_solar = True
	elif args.combined:
	# use_real_wind = False
		use_combined_energy = True

	queued_jobs = {}

	auction_times = []
	green_prices = []
	green_energy_usage = []
	brown_avg_price = []
	brown_energy_usage = []
	green_supply = []
	num_queued_jobs = []

	energy_api = power_supply.solar_gen
	if use_combined_energy:
		energy_api = power_supply.combined_renewable_power
	elif use_real_solar:
		if args.use_google:
			energy_api = power_supply.monthlong_solar_power
		else:
			energy_api = power_supply.real_solar_power
	# elif use_real_wind:
	# 	energy_api = power_supply.real_wind_power
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
			if args.use_google:
				available_green_energy = 30*energy_api(cur_time)
			else:
				available_green_energy = energy_api(cur_time)
			green_supply.append(available_green_energy * auction_interval_min)

			queued_job_ids = list(queued_jobs.keys())

			# update total disutility field for all jobs in queue
			for j_id in queued_job_ids:
				this_job = queued_jobs[j_id]
				disutility = job_bid_value(this_job, cur_time, runtime_interval=auction_interval_min) - this_job.instrinsic_minute_value*auction_interval_min
				# this_job.total_disutility += disutility

			green_job_ids = queued_job_ids
			brown_job_ids = []

			if available_green_energy > 0:
				green_auction = UniformAuction(available_green_energy, green_reserve_price)

				bid_vals = []
				job_ids = []
				for j_id in green_job_ids:
					pending_job = queued_jobs[j_id]
					job_val = job_bid_value(pending_job, cur_time, runtime_interval=auction_interval_min)
					this_bid = JobBid(job_val, pending_job.id)
					bid_vals.append(job_val)
					job_ids.append(j_id)
					green_auction.add_bid(this_bid)

				green_price = green_auction.solve_auction()

				bid_vals = np.array(bid_vals)
				job_ids = np.array(job_ids)


				green_winning_indexes = np.flatnonzero(bid_vals>=green_price)
				green_losing_indexes = np.flatnonzero(bid_vals<green_price)

				if len(green_winning_indexes) > available_green_energy:
					# tiebreaker for when # winners > amt energy
					actual_winners = np.random.choice(green_winning_indexes, size=available_green_energy, replace=False)

					new_losers = set(green_winning_indexes) - set(actual_winners)

					total_losers = set(green_losing_indexes) | new_losers
					green_losing_indexes = np.array(sorted(list(total_losers)))
					green_winning_indexes = np.array(sorted(actual_winners))


				green_winning_ids = job_ids[green_winning_indexes].tolist()
				winning_bid_vals = bid_vals[green_winning_indexes].tolist()

				# ensure green energy price is always lower than brown
				# effective_green_price = min(brown_reserve_price, green_price)
				effective_green_price = green_price

				for g in green_winning_ids:
					pending_job = queued_jobs[g]
					pending_job.total_payments += effective_green_price

				green_prices.append(effective_green_price)

				# print(available_green_energy, green_price)
			else:
				green_prices.append(0)
				green_winning_ids = []

				job_ids = []
				bid_vals = []
				for j_id in green_job_ids:
					job_ids.append(j_id)
					pending_job = queued_jobs[j_id]
					job_val = job_bid_value(pending_job, cur_time, runtime_interval=auction_interval_min)
					bid_vals.append(job_val)
				green_losing_indexes = list(range(0, len(job_ids)))

			green_energy_usage.append(len(green_winning_ids) * auction_interval_min)

			# run brown energy auction
			brown_auction = RSOPAuction(brown_reserve_price)

			# add losing green bids to brown auction
			for j in green_losing_indexes:
				j_id = job_ids[j]
				brown_job_ids.append(j_id)


			brown_bid_vals = []

			for j_id in brown_job_ids:
				pending_job = queued_jobs[j_id]

				job_val = job_bid_value(pending_job, cur_time, runtime_interval=auction_interval_min)
				this_bid = JobBid(job_val, pending_job.id)
				brown_auction.add_bid(this_bid)

			(best_price_1, market_2), (best_price_2, market_1) = brown_auction.solve_auction()
			brown_avg_price.append((best_price_1 + best_price_2) / 2.)
			brown_winning_ids = []
			for b in market_2:
				if b.price >= best_price_1:
					brown_winning_ids.append(b.id)
					queued_jobs[b.id].total_payments += best_price_1
				else:
					# jobs that lost both auctions
					if not checkpointing_available:
						queued_jobs[b.id].elapsed_time = 0
						queued_jobs[b.id].total_payments = 0
					elif queued_jobs[b.id].elapsed_time > 0:
						# count how many times jobs have to be checkpointed
						queued_jobs[b.id].num_restarts += 1

			for b in market_1:
				if b.price >= best_price_2:
					brown_winning_ids.append(b.id)
					queued_jobs[b.id].total_payments += best_price_2
				else:
					if not checkpointing_available:
						queued_jobs[b.id].elapsed_time = 0
						queued_jobs[b.id].total_payments = 0
					elif queued_jobs[b.id].elapsed_time > 0:
						# count how many times jobs have to be checkpointed
						queued_jobs[b.id].num_restarts += 1


			brown_energy_usage.append(len(brown_winning_ids) * auction_interval_min)
			all_winning_jobs = brown_winning_ids + green_winning_ids

			for win_id in all_winning_jobs:
				queued_jobs[win_id].elapsed_time += auction_interval
				if queued_jobs[win_id].duration - queued_jobs[win_id].elapsed_time <= 0:
					# job is done
					queued_jobs[win_id].end_time = cur_time + auction_interval # job finishes at END of period
					del queued_jobs[win_id]



	print(len(jobs) / auction_times[-1], len(jobs), auction_times[-1])
	total_revenue = 0
	total_value = 0
	total_disutility = 0

	total_runtime = 0

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
		if cur_job.duration - cur_job.elapsed_time <= 0:
			# finished job
			minute_delay = cur_job.end_time -  (cur_job.submission + cur_job.duration)
			minute_delay = int(minute_delay / 60)
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

	print('revenue, value', total_revenue, total_value)
	print('num checkpoints: {} {} {}'.format(low_pri_checkpoints, mid_pri_checkpoints, high_pri_checkpoints))
	print('# low pri: {}, # mid pri: {}, # high pri: {}'.format(low_pri_jobs, mid_pri_jobs, high_pri_jobs))
	print('low pri delay: {}, mid pri delay: {}, high pri delay: {}'.format(np.mean(low_pri_delays), np.mean(mid_pri_delays), np.mean(high_pri_delays)))
	print('low pri cost: {}, mid pri cost: {}, high pri cost: {}'.format(np.mean(low_pri_payments), np.mean(mid_pri_payments), np.mean(high_pri_payments)))
	print('green energy total usage: {}'.format(np.sum(green_energy_usage)))
	print('total available green energy: {}'.format(np.sum(green_supply)))

	print('brown energy total usage: {}'.format(np.sum(brown_energy_usage)))


	total_runtime /= 60 # in minutes
	print('gross execution time:', total_runtime)
	print('overall disutility:', total_disutility)
	print('loss func', total_disutility *(low_pri_checkpoints + mid_pri_checkpoints + high_pri_checkpoints) / len(jobs))
	# print('avg price per min: {}'.format(total_revenue / total_runtime))




	np.save('green_prices.txt', green_prices)
	np.save('auction_times.txt', auction_times)
	np.save('green_supply.txt', green_supply)
	np.save('num_queued_jobs.txt', num_queued_jobs)
	np.save('brown_avg_price.txt', brown_avg_price)

	np.save('green_usage.txt', green_energy_usage)
	np.save('brown_usage.txt', brown_energy_usage)


	# plt.figure(figsize=(8,6))
	# # plt.ylim(0, 0.1)
	# plt.plot(auction_times, green_prices)
	# plt.title('Green Prices')
	# plt.show()
	# plt.plot(auction_times, green_supply)
	# plt.title('Green Power Supply')
	# plt.show()
	# plt.plot(auction_times, num_queued_jobs)
	# plt.title('Queued Jobs')
	# plt.show()

	# plt.plot(auction_times, green_energy_usage)
	# plt.title('Green energy usage')
	# plt.show()

	# plt.plot(auction_times, brown_energy_usage)
	# plt.title('Brown energy usage')
	# plt.show()


	# print(np.histogram(t, bins=[0, 0.33, 0.66, 1]))
	print(power_supply.solar_gen(30000))
# print(np.histogram(t, bins=[0, 0.25, 0.50, 0.75, 1]))






