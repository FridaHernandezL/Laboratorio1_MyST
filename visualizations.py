"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Laboratorio 1. Inversión Pasiva y Activa                                                   -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: FridaHernandezL                                                                             -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/FridaHernandezL/Laboratorio1_MyST                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
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
import data as dta



#prices time series
def prices_timeseries(data):
    plt.figure(figsize=(20,9)) 
    for i in data.columns.values:
        plt.plot( data[i],  label=i)
    plt.title('Prices')
    plt.xlabel('Dates',fontsize=18)
    plt.ylabel('Prices MXN',fontsize=18)
    plt.legend(data.columns.values, loc='best')
    plt.show()
pasiva_ts=prices_timeseries(dta.prices1)

#subplots del dataframe final
def pasive_plot(dataframe):
    dataframe.plot(kind = 'bar',
                 width=0.8,
                 subplots=True,
                 figsize=(10,20),
                   color=["#FF9B85","#60D394","#AAF683"]);
pasiva_suplot=pasive_plot(dta.pasivaplot)


#activa
def frontera_ef(frontera):
    plt.figure(figsize=(6,4))
    plt.scatter(frontera['Vol'],frontera['Media'],c=frontera['RS'],cmap='RdYlBu')
    plt.grid()
    plt.xlabel('Volatilidad $\sigma$')
    plt.ylabel('Rendimiento Esperado $E[r]$')
    plt.colorbar()
fronteraef=frontera_ef(dta.frontera)


def rend_active(data):
    ret=data.iloc[250:].pct_change().dropna()
    plt.figure(figsize=(12.2,4.5)) 
    for i in ret.columns.values:
        plt.hist( ret[i],  label=i, bins = 50)
    plt.title('Histograma de los retornos')
    plt.xlabel('Fecha',fontsize=18)
    plt.ylabel('Precio en USD',fontsize=18)
    #plt.legend(ret.columns.values,loc='best')
    plt.show()
rend_plot=rend_active(dta.closes)