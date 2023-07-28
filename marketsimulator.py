

                	  	  			  		 			     			  	 

import datetime as dt  		  	   		  	  			  		 			     			  	 

import os  		





import numpy as np  		  	   		  	  			  		 			     			  	 

  		  	   		  	  			  		 			     			  	 

import pandas as pd  		  	   		  	  			  		 			     			  	 

from util import get_data, plot_data  	



def author():

    return 'nkrishnan42'



def get_balance(order, shares, price, impact, commission):

    if order=='BUY':

        price= (1+impact)*price

        

    else:

        shares=shares*-1

        price= (1-impact)*price

    cash= price*shares*-1-commission

    return (shares, cash)





def assess_portfolio(port_val, sf=252.0):



    daily_rets= get_daily_returns(port_val)

    





    cum_rets= (port_val[-1]/port_val[0])-1

    avg_daily_rets= np.mean(daily_rets)

    std_daily_rets=  np.std(daily_rets)    

     

    return [cum_rets, avg_daily_rets, std_daily_rets]



                   







    



def get_sharpe_ratio(allocs, prices):

    return -1* (assess_portfolio(prices, allocs)[4])



def get_daily_returns(port_val):

    dr= (port_val[1:].values/port_val[:-1])-1

    return dr[1:]





def compute_portvals(df_orders, start_val=1000000, commission=9.95,impact=0.005):  		  	   		  	  			  		 			     			  	 

 		  	   		  	  			  		 			     			  	 	  	   		  	  			  		 			     			  	 	  	   		  	  			  		 			     			  	 

    start_date= df_orders.index.min()

    end_date= df_orders.index.max()	 

    syms= (df_orders['Symbol'].unique().tolist())   		  	  			  		 			     			  	 

  		  	   		  	  			  		 			     			  	 

    df_prices= get_data(symbols=syms,dates=pd.date_range(start_date, end_date))

    df_prices['Cash']= 1.0

    df_temp= df_prices.copy(deep=True)

    df_trades= df_temp.replace(df_temp, 0)

    

    cash_col= np.zeros((df_prices.shape[0], 1))

    df_trades['Cash']= cash_col



    

    #Loop thru orders file and update df_trades with delta num stocks, delta cash



    for i in range(df_orders.shape[0]):

      

        date= df_orders.index[i]

        

      

        symbol= df_orders.iloc[i]['Symbol']

        order= df_orders.iloc[i]['Order']

        num_shares= df_orders.iloc[i]['Shares']



        price= df_prices.loc[date][symbol]

        



        shares, cash= get_balance(order, num_shares, price, impact, commission)



        cash+= df_trades.loc[date]['Cash']

        shares+= df_trades.loc[date][symbol]

        

        df_trades.loc[date][symbol]= shares



        

        df_trades.loc[date]['Cash']= cash

        





    df_holdings= df_trades.copy(deep=True)

   

    

    date= df_holdings.index[0]

    df_holdings.loc[date]['Cash']+=start_val





    #loop through trades and update holdings



    df_holdings=df_holdings.cumsum()







    df_values= df_holdings.copy(deep=True)

  



    df_values=df_values*df_prices







    df_portVals= df_values.sum(axis=1)

    

  #  df_portVals=df_portVals.to_frame('Value')



    return df_portVals 