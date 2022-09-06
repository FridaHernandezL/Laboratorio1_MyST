"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Laboratorio 1. Inversión Pasiva y Activa                                                   -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: FridaHernandezL                                                                             -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/FridaHernandezL/Laboratorio1_MyST                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

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
prices_timeseries(prices1)

#subplots del dataframe final
def final_plot(dataframe):
    dataframe.plot(kind = 'bar',
                 width=0.8,
                 subplots=True,
                 figsize=(10,20),
                   color=["#FF9B85","#60D394","#AAF683"]);

