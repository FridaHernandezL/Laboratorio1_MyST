"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Laboratorio 1. Inversión Pasiva y Activa                                                   -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: FridaHernandezL                                                                             -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/FridaHernandezL/Laboratorio1_MyST                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#Imports files from files folder (31)
def import_files(folder):
    path = folder
    files = glob.glob(path + "/*.csv")
    data_frame = pd.DataFrame()
    content = []
    for filename in files:
        df = pd.read_csv(filename, skiprows=(0,1)).dropna().sort_values('Ticker').reset_index(drop=True)
        content.append(df)
    data_frame = pd.concat(content)
    data_frame=data_frame[['Ticker','Peso (%)','Precio','Acciones']]
    return data_frame


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


#Transpose prices from yf
def transpose_prices(data):
    price=pd.DataFrame(data).dropna().reset_index()
    price['Date']=price['Date'].dt.strftime('%Y/%m/%d')
    price=[price[price['Date']==dates[i]] for i in range(len(dates))]
    content = []
    for i in range(len(price)):   
        df = price[i]
        content.append(df)
    prices = pd.concat(content).reset_index(drop=True).set_index('Date')
    return prices


#returns dataframe (symbols|peso|precio|postura|titulos)31/01/2020
def postura_inicial(data):
    j20=data.iloc[0:36,0:2]
    t=data['Ticker'].value_counts().sort_index().loc[lambda x: x>=x.max()].index.tolist()
    j20=j20[j20.Ticker.isin(t)].reset_index(drop=True)
    for i in range(len(tickers)):
        j20.iloc[i,0]=tickers[i]
    j20['Precio']=prices.iloc[:,0].values
    j20['Postura']=j20['Peso (%)']/100*k
    j20['Titulos']=round(j20['Postura']/j20['Precio'])
    j20['Postura']=(j20['Peso (%)']/100*k)-(j20['Titulos']*j20['Precio']*com)    #considerando comisiones
    cash=(100-(pin['Peso (%)'].values.sum()))*k
    return j20


#Returns dataframe (timestamp|capital|rend|rend_acum)monthly
def pasiva_results(yfprices,posturainicial):
    pasiva=pd.DataFrame(columns=['timestamp','capital','rend','rend_acum'])
    post=yfprices.mul(posturainicial.iloc[:,4].values, axis = 0) 
    for i in range(len(dates)):
        pasiva['timestamp']=dates
        pasiva.iloc[i,1]=post.iloc[:,i].sum()
    pasiva['rend']=pasiva['capital'].pct_change().fillna((pasiva.iloc[0,1]-k)/pasiva.iloc[0,1])
    pasiva['rend_acum']=pasiva['rend'].cumsum()  
    return pasiva