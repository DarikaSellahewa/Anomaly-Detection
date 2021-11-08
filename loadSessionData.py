# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:43:49 2021

@author: darik
"""
import dill

filename = 'globalsave.pkl'
dill.load_session(filename)

#load session data

filename = 'globalsave_svm.pkl'
dill.load_session(filename)