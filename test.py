#this is for test
import pandas as pd
import sys
import numpy as np


def transform(ifile, ofile):
    '''
    transform the data format to fit pandas
    '''
    head = 'uid,x,y,time,place_id\n'
    ifile = open(ifile, 'r')
    ofile = open(ofile, 'w')
    ofile.write(head)
    while True:
        line = ifile.readline()
        if not line:
            break
        if len(line) > 10:
            ofile.write(line)

    ifile.close()
    ofile.close()

if __name__ == '__main__':
    transform(sys.argv[1], sys.argv[2])
