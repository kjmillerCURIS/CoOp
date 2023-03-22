import os
import sys

#in case I don't trust qsub
def write_to_log_file(msg):
    print(msg)
    f = open('meow.txt', 'a')
    f.write(msg + '\n')
    f.close()
