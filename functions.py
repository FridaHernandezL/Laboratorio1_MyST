"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Laboratorio 1. Inversión Pasiva y Activa                                                   -- #
# -- script: functions.py : python script with general functions                                         -- #
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


absolute_path = os.path.abspath(os.path.dirname('closing_prices.csv'))
closing_prices=pd.read_csv(absolute_path + '\created_files\closing_prices.csv')
closing_prices=closing_prices.drop(closing_prices.columns[0],axis=1)

#Imports files from files folder (31)
def import_files(folder):
    files = glob.glob(os.path.abspath(os.path.dirname('NAFTRAC_20200131.csv'))+ folder + "/*.csv")
    data_frame = pd.DataFrame()
    content = []
    for filename in files:
        df = pd.read_csv(filename, skiprows=(0,1)).dropna().sort_values('Ticker').reset_index(drop=True)
        content.append(df)
    data_frame = pd.concat(content)
    data_frame=data_frame[['Ticker','Peso (%)','Precio','Acciones']]
    return data_frame
data=import_files('\\files')

#Selects most common symbols and changes its name to yf tickers
def ticker(datacolumn):
    #Seleccionar tickers que más se repiten
    t=datacolumn.value_counts().sort_index().loc[lambda x: x>=x.max()].index.tolist()
    #Eliminar signos extraños (no concuerdan en yfinance)
    new_tickers=[]
    strange=['*']
    for word in t:
        for letter in word:
            if letter in strange:
                word = word.replace(letter,"")   
        new_tickers.append(word)
    new_tickers[21]='LIVEPOLC-1' #editar single tickers que no coincidan
    #agregar '.MX' al final para que concuerde con yfinance
    addt=['.MX']*len(new_tickers)
    newtickers=[new_tickers[i]+addt[i] for i in range(len(new_tickers))]
    newtickers[23]='MXN=X' #editar single tickers que no coincidan
    return newtickers
tickers=ticker(data['Ticker'])

#Imports close prices from yfinance
def get_adj_closes(tickers, start_date=None, end_date=None):
    # Fecha inicio por defecto (start_date='2010-01-01') y fecha fin por defecto (end_date=today)
    # Descargamos DataFrame con todos los datos
    closes = yf.download(tickers, start_date, end_date)
    # Solo necesitamos los precios ajustados en el cierre
    closes = closes['Close']
    # Se ordenan los índices de manera ascendente
    closes.sort_index(inplace=True)
    return closes

dates = ['2020-01-31','2020-02-28','2020-03-31','2020-04-30','2020-05-29','2020-06-30',
                    '2020-07-31','2020-08-31','2020-09-30','2020-10-30','2020-11-30','2020-12-31',
                    '2021-01-29','2021-02-26','2021-03-31','2021-04-30','2021-05-31','2021-06-30',
                    '2021-07-30','2021-08-31','2021-09-30','2021-10-26','2021-11-30','2021-12-31',
                    '2022-01-26','2022-02-28','2022-03-31','2022-04-29','2022-05-31','2022-06-30',
                    '2022-07-29']

#Transpose prices from yf
def transpose_prices(data):
    price=pd.DataFrame(data).dropna().reset_index()
    #price['Date']=price['Date'].dt.strftime('%Y/%m/%d')
    price=[price[price['Date']==dates[i]] for i in range(len(dates))]
    content = []
    for i in range(len(price)):   
        df = price[i]
        content.append(df)
    prices = pd.concat(content).reset_index(drop=True).set_index('Date')
    return prices

#return timeseries plot
def prices_timeseries(data):
    plt.figure(figsize=(20,9)) 
    for i in data.columns.values:
        plt.plot( data[i],  label=i)
    plt.title('Prices')
    plt.xlabel('Dates',fontsize=18)
    plt.ylabel('Prices MXN',fontsize=18)
    plt.legend(data.columns.values, loc='best')
    plt.show()

prices=transpose_prices(closing_prices)
prices=prices.drop('index',axis=1).T
#returns dataframe (symbols|peso|precio|postura|titulos)31/01/2020
def postura_inicial(data):
    k=1000000
    com= 0.00125
    j20=data.iloc[0:36,0:2]
    t=data['Ticker'].value_counts().sort_index().loc[lambda x: x>=x.max()].index.tolist()
    j20=j20[j20.Ticker.isin(t)].reset_index(drop=True)
    for i in range(len(tickers)):
        j20.iloc[i,0]=tickers[i]
    j20['Precio']=prices.iloc[:,0].values
    j20['Postura']=j20['Peso (%)']/100*k
    j20['Titulos']=round(j20['Postura']/j20['Precio'])
    j20['Postura']=(j20['Peso (%)']/100*k)-(j20['Titulos']*j20['Precio']*com)    #considerando comisiones
    return j20
pin=postura_inicial(data)

#Returns dataframe (timestamp|capital|rend|rend_acum)monthly
def pasiva_results(yfprices,posturainicial):
    k=1000000
    pasiva=pd.DataFrame(columns=['timestamp','capital','rend','rend_acum'])
    post=yfprices.mul(posturainicial.iloc[:,4].values, axis = 0) 
    for i in range(len(dates)):
        pasiva['timestamp']=dates
        pasiva.iloc[i,1]=post.iloc[:,i].sum()
    pasiva['rend']=pasiva['capital'].pct_change().fillna((pasiva.iloc[0,1]-k)/pasiva.iloc[0,1])
    pasiva['rend_acum']=pasiva['rend'].cumsum()  
    return pasiva


#subplots for pasive investment
def pasive_plot(dataframe):
    dataframe.plot(kind = 'bar',
                 width=0.8,
                 subplots=True,
                 figsize=(10,20),
                   color=["#FF9B85","#60D394","#AAF683"]);



#RS
def var(w,Sigma):
    return w.T.dot(Sigma).dot(w)

def menos_RS(w,re,rf,Sigma):
    E_port=re.T.dot(w)
    s_port=var(w,Sigma)**0.5
    RS=(E_port-rf)/s_port
    return -RS

#df portafolio eficiente
def port_ef(pesos,data,summary):
    k=1000000
    Peficiente=pd.DataFrame(columns=summary.columns)
    Peficiente.loc['Pesos%']=pesos*100
    Peficiente.loc['Postura']=Peficiente.iloc[0,:].values/100*k
    Peficiente.loc['Precio']=data.iloc[-1,:]
    Peficiente.loc['Titulos']=round(Peficiente.loc['Postura']/Peficiente.loc['Precio'])
    Peficiente=Peficiente.T.reset_index()
    return Peficiente


#df operaciones
def df_operaciones(dates,ticker,data,w_emv):
    df_op=pd.DataFrame(columns=['timestamp','titulos_totales','titulos_op','cash_acum','comision_x_op','comision_acum'])
    df_op['timestamp']=dates
    c=pd.DataFrame(columns=[tickers],index=dates)
    com= 0.00125
    k=1000000*(1-0.03)
    df_op.iloc[0,1]=round((w_emv*k)/data.iloc[0,:]).sum()
    df_op.iloc[0,2]=df_op.iloc[0,1]
    df_op.iloc[0,3]=k
    c.iloc[0,:]=(round((w_emv*k)/data.iloc[0,:])).values
    for j in range(len(dates)-1):  
        change=1-(data.iloc[j+1,:]/data.iloc[j,:])
        for i in range(len(w_emv)-1):
            if change[i]>=0.05:
                c.iloc[j+1,i]=c.iloc[j,i]*(1-0.025)
            else:
                c.iloc[j+1,i]=c.iloc[j,i]*(1+0.025)
        df_op.iloc[j+1,1]=round(c.iloc[j+1,:].sum(),0)
        df_op.iloc[j+1,2]=df_op.iloc[j,1]-df_op.iloc[j+1,1]
        df_op.iloc[j+1,3]=df_op.iloc[j,3]+round((data.iloc[j+1,:]*(c.iloc[j,:]-c.iloc[j+1,:]).values).sum(),3)
        df_op.iloc[j+1,4]=round(df_op.iloc[j+1,3]*com,3)
        df_op['comision_x_op']=df_op['comision_x_op'].fillna(0)
        df_op['comision_acum']=df_op['comision_x_op'].cumsum() 
    return df_op


def RS(data,pesos):
    summary=pd.DataFrame(columns=data.columns)
    summary.loc['Media']=data.mean()
    summary.loc['Volatilidad']=data.std()*(254**(1/2))
    corr=data.corr()
    S= np.diag(summary.loc['Volatilidad'].values)
    Sigma=S.dot(corr).dot(S)
    re=summary.loc['Media'].values
    rf=0.085
    N=len(re)                                               
    w0=np.ones(N)/N                                        
    bnds=((0,1),)*N                                         
    cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)
    emv=minimize(menos_RS, pesos, args=(re,rf,Sigma),bounds=bnds, constraints=cons)
    w_emv1=emv.x
    E_emv=re.T.dot(w_emv1)
    s_emv=var(w_emv1,Sigma)**0.5
    RS_emv=(E_emv-rf)/s_emv
    return RS_emv