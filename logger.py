import csv

def write_trial(trial_info):
	f = open(f'exp_{trial_info["exp_tag"]}.csv','a+', newline='')
	wr = csv.writer(f)
	wr.writerow(trial_info)