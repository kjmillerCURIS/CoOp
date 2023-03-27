import os
import sys
from crontab import CronTab
from datetime import datetime, timedelta

DEFAULT_WAITING_PERIOD = 2 / 60
DEFAULT_WORKING_DIR = '/usr3/graduate/nivek/data/CoOp/job_scripts'
DEFAULT_BATCH_SIZE = 10

#NOTE: this "jitter" doesn't actually do anything because crontab can only handle minute-level resolution. let's hope qsub can handle concurrency...
JITTER = 1 / 3600 #to make sure no two jobs get submitted at EXACTLY the same time

BOILERPLATE = 'PATH=/usr/local/bin:/usr/bin:/bin SGE_ROOT=/usr/local/sge/sge_root'

class FriendlySubmitter:
    def __init__(self, interval, batch_size=DEFAULT_BATCH_SIZE, waiting_period=DEFAULT_WAITING_PERIOD, working_dir=DEFAULT_WORKING_DIR):
        self.cron = CronTab(user='nivek')
        self.interval = interval
        self.batch_size = batch_size
        self.waiting_period = waiting_period
        self.working_dir = working_dir
        self.commands = []

    def add(self, command):
        self.commands.append(command)

    def run(self):
        nowtime = datetime.now()
        for i, command in enumerate(self.commands):
            batch_index = i // self.batch_size
            full_command = 'cd %s && %s %s'%(self.working_dir, BOILERPLATE, command)
            delay = self.waiting_period + batch_index * self.interval
            subtime = nowtime + timedelta(hours=delay) + timedelta(hours=(i%self.batch_size)*JITTER)
            print('\nWill submit the follow job at %s:\n%s'%(str(subtime), command))
            job = self.cron.new(command=full_command)
            job.setall(subtime)

        self.cron.write()
