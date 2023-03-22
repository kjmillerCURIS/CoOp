import os
import sys
import glob

def get_job_names(job_nums):
    job_nums = job_nums.split(',')
    job_names = []
    for job_num in job_nums:
        stuff = sorted(glob.glob('*.o' + job_num))
        assert(len(stuff) < 2)
        if len(stuff) > 0:
            job_names.append(os.path.splitext(os.path.basename(stuff[0]))[0])

        print(job_names)

if __name__ == '__main__':
    get_job_names(*(sys.argv[1:]))
