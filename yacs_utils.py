import os
import sys

#takes string of format 'k1,v1,k2,v2,...' and parses it into string==>string dict
#this is because yacs for some reason doesn't support dicts
def parsedict(s):
    ss = s.split(',')
    assert(len(ss) % 2 == 0)
    d = {}
    for i in range(len(ss) // 2):
        d[ss[2*i]] = ss[2*i+1]

    return d

def parsedictlist(s):
    d = parsedict(s)
    d = {k : d[k].split('_') for k in sorted(d.keys())}
    return d
