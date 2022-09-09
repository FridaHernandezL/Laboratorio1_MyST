
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #

Created on Wed Aug 24 14:35:56 2022

@author: FridaHernandezL
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import glob
import os
from scipy.optimize import minimize
import locale
locale.setlocale( locale.LC_ALL, '' )
import functions as fn
import main as mn

#PASIVE
data=fn.import_files('\\files')
tickers=fn.ticker(data['Ticker'])
dates=mn.dates

closing_prices=mn.closing_prices
prices=mn.prices
prices1=mn.prices1

pin=mn.pin
cash=mn.cash
pasiva=mn.pasiva
pasive_rs=mn.pasive_rs



#ACTIVA
closes=mn.closes
yearone=mn.yearone

###port ef###
Peficiente=mn.Peficiente
w_emv=mn.w_emv
rsmax=mn.RS_minvar
frontera=mn.frontera
yeartwo=mn.yeartwo
df_op=mn.df_op
frontera=mn.frontera
active_rs=mn.active_rs
casha=mn.casha

yeartwoplot=mn.yeartwoplot