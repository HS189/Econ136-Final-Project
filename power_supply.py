import numpy as np

real_solar_data = np.loadtxt('data/solar_fraction_raw_energy.txt')
real_wind_data = np.loadtxt('data/wind_fraction_raw_energy.txt')
combined_data = np.loadtxt('data/combined_raw_energy.txt')

def solar_gen(time):
	day_time = time % 86400
	# assume 6am to 8pm sunlight
	if day_time < 6*3600 or day_time > 20*3600:
		return 0

	peak = 1000 # watts

	hour_day = day_time / 3600.
	power = -1*(hour_day - 6)*(hour_day-20)
	power *= (peak / 49.)

	# noise = 2
	# while noise > 1 or noise < 0:
	# 	noise = np.abs(np.random.normal())
	# 	noise = 1 - noise
	noise = 1

	return round(noise*power)

def real_solar_power(time):
	# uses real data

	time_min = round(time / 60.) # convert to minutes
	return round(real_solar_data[time_min])

def real_wind_power(time):
	# uses real data

	time_min = round(time / 60.) # convert to minutes
	return round(real_wind_data[time_min])

def combined_renewable_power(time):
	# uses real data

	time_min = round(time / 60.) # convert to minutes
	return round(combined_data[time_min])


