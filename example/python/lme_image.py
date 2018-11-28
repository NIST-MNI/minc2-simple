#! /usr/bin/env python

import sys
import os
import pyezminc
import csv

from minc2_simple import minc2_input_iterator,minc2_output_iterator

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

def load_csv(csv_file):
    '''Load csv file into a dictionary'''
    data={}
    # load CSV file 
    with open(input_csv,'r') as f:
        for r in csv.DictReader(f):
            for k in r.iterkeys():
                try:
                    data[k].append(r[k])
                except KeyError:
                    data[k]=[r[k]]
    return data


# setup automatic conversion for numpy to Rpy
numpy2ri.activate()

# import R objects
# define R objects globally, so that we don't have to transfer them between instances
stats = importr('stats')
base  = importr('base')
nlme  = importr('nlme')

# read the input data
input_csv='longitudinal.csv'
#mask_file='mask.mnc'

# load CSV file
data=load_csv(input_csv)
# columns:
# signal,mask,subject,group,visit
# 

# setup R objects for performing linear modelling
subject = ro.FactorVector(data['subject'])
visit   = ro.FactorVector(data['visit'])
group   = ro.FactorVector(data['group'])

# allocate R formula, saves a little time for interpreter
random_effects = ro.Formula('~1|subject')
# outputs
zero=np.zeros(shape=[6],dtype=np.float64,order='C')

# this function will be executed in parallel
def run_lme(signal,mask):
    # this object have to be defined within the function to avoid funny results due to concurrent execution
    fixed_effects = ro.Formula('signal ~ group+visit')

    fixed_effects.environment["mask"]   = rm = ro.BoolVector(mask>0.5)
    fixed_effects.environment["signal"] = ro.FloatVector(signal).rx(rm)
    # assign variables 
    fixed_effects.environment["subject"]  = subject.rx(rm)
    fixed_effects.environment["visit"]    = visit.rx(rm)
    fixed_effects.environment["group"]    = group.rx(rm)
    
    # update jacobian variable

    # allocate space for output
    result=np.zeros(shape=[6],dtype=np.float64,order='C')

    try:
        # run linear mixed-effect model
        l = base.summary(nlme.lme(fixed_effects,random=random_effects,method="ML"))
        # extract coeffecients
        result[0:3]  = l.rx2('coefficients').rx2('fixed')[:]
        # extract t-values
        result[3:6] = l.rx2('tTable').rx(True,3)[:]

    except RRuntimeError:
        # probably model didn't converge
        pass

    return result

if __name__ == "__main__":
    
    max_jobs_queue=100
    
    inp_signal=minc2_input_iterator(files=data['signal'])
    inp_mask=minc2_input_iterator(files=data['mask'])
    
    # setup output iterator
    out=minc2_output_iterator(files=["output_Intercept.mnc","output_group.mnc","output_visit.mnc",
                                     "output_Intercept_t.mnc","output_group_t.mnc","output_visit_t.mnc" ])
        for signal,mask in zip(inp_singnal,inp_mask):
            
            res=run_nlme(signal,mask)
            out.next(res)
    except StopIteration:
        print("Stopped early?")
        pass
    # delete input iterator, free memory, close files, usually done automatically
    inp.close()
    # free up memory, close file not really needed here, usually done automatically
    out.close()
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
