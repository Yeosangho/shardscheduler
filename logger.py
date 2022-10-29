import csv

def write_trial(trial_info):
	f = open(f'exp_{trial_info["exp_tag"]}.csv','a+', newline='')
	wr = csv.writer(f)
	wr.writerow([trial_info["bucket_size"], trial_info["dp"], trial_info["sdp"], trial_info["fsdp"], trial_info["time"]])