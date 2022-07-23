import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rank', dest='rank', default=0, type=int)

args = parser.parse_args()

proc_exec = True
while proc_exec :
    try:
        print("start proc")
        subprocess.check_output(['/home/soboru963/anaconda3/envs/pytorch19/bin/python', 'main.py', '--rank', str(args.rank)])      
        print('end proc')
    except subprocess.CalledProcessError as e:
        #print(e.output)
        for line in e.output.splitlines():
            print(line)
        proc_exec = True