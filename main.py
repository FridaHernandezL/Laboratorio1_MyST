
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Laboratorio 1. Inversión Pasiva y Activa                                                   -- #
# -- script: main.py : python script with the main functionality                                         -- #
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
import functions as fn

#PASIVE

dates = ['2020-01-31','2020-02-28','2020-03-31','2020-04-30','2020-05-29','2020-06-30',
                    '2020-07-31','2020-08-31','2020-09-30','2020-10-30','2020-11-30','2020-12-31',
                    '2021-01-29','2021-02-26','2021-03-31','2021-04-30','2021-05-31','2021-06-30',
                    '2021-07-30','2021-08-31','2021-09-30','2021-10-26','2021-11-30','2021-12-31',
                    '2022-01-26','2022-02-28','2022-03-31','2022-04-29','2022-05-31','2022-06-30',
                    '2022-07-29']

absolute_path = os.path.abspath(os.path.dirname('closing_prices.csv'))
closing_prices=pd.read_csv(absolute_path + '\created_files\closing_prices.csv')
closing_prices=closing_prices.drop(closing_prices.columns[0],axis=1)

prices=fn.transpose_prices(closing_prices)
prices=prices.drop('index',axis=1).T
prices1=prices.T

data=fn.import_files('\\files')
pin=fn.postura_inicial(data)
k=1000000
cash=((100-(pin['Peso (%)'].values.sum()))/100)*k

pasiva=fn.pasiva_results(prices,pin)
pasivaplot=pasiva.set_index('timestamp')

pasive_rs=fn.RS(prices1,pin.iloc[:,1].values)


#ACTIVE
absolute_path = os.path.abspath(os.path.dirname('closing_prices.csv'))
closes=pd.read_csv(absolute_path + '\created_files\closing_prices.csv')
closes=closes.set_index('Date').drop(closes.columns[0],axis=1)
closes=closes.drop(closes.columns[23],axis=1)

yearone=closes.iloc[0:251]

############portafolio eficiente max rs################################
summary=pd.DataFrame(columns=yearone.columns)
summary.loc['Media']=yearone.mean()
summary.loc['Volatilidad']=yearone.std()*(254**(1/2))

#correlaciones
corr=yearone.corr()

#riskfree rate
rf=0.085

#matiz varianza-covarianza
S= np.diag(summary.loc['Volatilidad'].values)
Sigma=S.dot(corr).dot(S)
# rendimientos esperados activos individuales
re=summary.loc['Media'].values

N=len(re)                                               # Número de activos
w0=np.ones(N)/N                                         # Dato inicial
bnds=((0,1),)*N                                         # Cotas de las variables
cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)    # Restricciones

# Portafolio de mínima varianza
minvar=minimize(fn.var, w0, args=(Sigma,),bounds=bnds, constraints=cons)
w_minvar=minvar.x
E_minvar=re.T.dot(w_minvar)
s_minvar=fn.var(w_minvar,Sigma)**0.5
RS_minvar=(E_minvar-rf)/s_minvar


emv=minimize(fn.menos_RS, w0, args=(re,rf,Sigma),bounds=bnds, constraints=cons)

w_emv=emv.x                                     ##PESOS PORT EF
E_emv=re.T.dot(w_emv)
s_emv=fn.var(w_emv,Sigma)**0.5
RS_emv=(E_emv-rf)/s_emv					

w_minvar = minvar.x
E_minvar =re.T.dot(w_minvar)
s_minvar = fn.var(w_minvar, Sigma)**0.5
RS_minvar= (E_minvar - rf) / s_minvar           ##MAX RS

cov_emv_minvar=w_emv.T.dot(Sigma).dot(w_minvar)
corr_emv_minvar=cov_emv_minvar/(s_emv*s_minvar)
w_p=np.linspace(0,1)
frontera = pd.DataFrame(data={'Media' : w_p*E_emv + (1-w_p)*E_minvar,
                             'Vol': ((w_p*s_emv)*2 +((1-w_p)*s_minvar)**2 + 2 * w_p * (1-w_p)*cov_emv_minvar)*0.5})
frontera['RS']=(frontera['Media'] - rf)/frontera['Vol']



Peficiente=fn.port_ef(w_emv,yearone,summary)

dates=['2021-01-29','2021-02-26','2021-03-31','2021-04-30','2021-05-31','2021-06-30','2021-07-30','2021-08-31','2021-09-30','2021-10-29',
      '2021-11-30','2021-12-31','2022-01-31','2022-02-28','2022-03-31','2022-04-29','2022-05-31','2022-06-30','2022-07-29']
yeartwo=closes.iloc[250:]
yeartwo=yeartwo.reset_index()
yeartwo=yeartwo.loc[yeartwo['Date'].isin(dates)]
yeartwo=yeartwo.set_index('Date')
tickers=['AC.MX', 'ALFAA.MX','ALSEA.MX','AMXL.MX','ASURB.MX','BBAJIOO.MX','BIMBOA.MX','BOLSAA.MX','CEMEXCPO.MX','CUERVO.MX',
 'ELEKTRA.MX', 'FEMSAUBD.MX','GAPB.MX','GCARSOA1.MX','GFINBURO.MX','GFNORTEO.MX','GMEXICOB.MX','GRUMAB.MX','KIMBERA.MX',
 'KOFUBL.MX','LABB.MX','LIVEPOLC-1.MX','MEGACPO.MX','OMAB.MX','ORBIA.MX','PE&OLES.MX','PINFRA.MX','TLEVISACPO.MX',
 'WALMEX.MX']

##df operaciones
df_op=pd.DataFrame(columns=['timestamp','titulos_totales','titulos_op','cash_acum','comision_x_op','comision_acum','rend','rend_acum'])
df_op['timestamp']=dates
c=pd.DataFrame(columns=[tickers],index=dates)
com= 0.00125 
k=1000000*(1-0.03)
df_op.iloc[0,1]=round((w_emv*k)/yeartwo.iloc[0,:]).sum()
df_op.iloc[0,2]=df_op.iloc[0,1]
df_op.iloc[0,3]=k
c.iloc[0,:]=(round((w_emv*k)/yeartwo.iloc[0,:])).values
for j in range(len(dates)-1):  
    change=1-(yeartwo.iloc[j+1,:]/yeartwo.iloc[j,:])
    for i in range(len(w_emv)-1):
        if change[i]>=0.05:
            c.iloc[j+1,i]=c.iloc[j,i]*(1-0.025)
        else:
            c.iloc[j+1,i]=c.iloc[j,i]*(1+0.025)
    df_op.iloc[j+1,1]=round(c.iloc[j+1,:].sum(),0)
    df_op.iloc[j+1,2]=df_op.iloc[j,1]-df_op.iloc[j+1,1]
    df_op.iloc[j+1,3]=df_op.iloc[j,3]+round((yeartwo.iloc[j+1,:]*(c.iloc[j,:]-c.iloc[j+1,:]).values).sum(),3)
    df_op.iloc[j+1,4]=round(df_op.iloc[j+1,3]*com,3)
    df_op['comision_x_op']=df_op['comision_x_op'].fillna(0)
    df_op['comision_acum']=df_op['comision_x_op'].cumsum()
    df_op['rend']=df_op['cash_acum'].pct_change().fillna(0)
    df_op['rend_acum']=df_op['rend'].cumsum()  


#rs
active_rs=fn.RS(yeartwo,w_emv)


df_medidas=pd.DataFrame(columns=['Medida','Descripcion','Inv_Pasiva','Inv_Activa'])
desc=['Rendimiento Promedio Mensual','Rendimiento Promedio Acumulado','Radio de Sharpe']
med=['rend_m','rend_c','sharpe']
df_medidas.iloc[:,0]=med
df_medidas.iloc[:,1]=desc
df_medidas.iloc[0,2]=pasiva.rend.mean()*100
df_medidas.iloc[1,2]=pasiva.rend_acum.mean()*100
df_medidas.iloc[2,2]=pasive_rs
df_medidas.iloc[0,3]=df_op.rend.mean()*100
df_medidas.iloc[1,3]=df_op.rend_acum.mean()*100
df_medidas.iloc[2,3]=active_rs