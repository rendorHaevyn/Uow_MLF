# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 1/')

import graphlab as gl

sf = gl.SFrame('people-example.csv')

import pandas as pd

pf = pd.read_csv('people-example.csv')
 
sf.sample(.2)

sf.show()

sf.column_names

sf['Country'].show
sf['Country'].apply(lambda x: 'United States' if x == 'USA' else x)