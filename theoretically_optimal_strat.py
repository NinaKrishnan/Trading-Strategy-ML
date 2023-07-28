

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from marketsimcode import compute_portvals, assess_portfolio

from util import get_data



#############################################################################################################





def author():

    return 'nkrishnan42'



#############################################################################################################





def make_trade(order, symbol, date, df_prices, num_shares, cash, shares, df_trades):

    

    price= df_prices.loc[date][symbol]

    if order=='buy':

        shares+=num_shares

        cash-=price*num_shares

        df_trades.loc[date][symbol]=num_shares

        df_trades.loc[date]['Cash']= (price*num_shares)*-1



    else:

        shares-=num_shares

        cash+=price*num_shares

        df_trades.loc[date][symbol]=num_shares*-1

        df_trades.loc[date]['Cash']= (price*num_shares)



  



    return (shares, cash)





#############################################################################################################



        

def test_policy(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), start_val=100000):

   



    date_range= pd.date_range(sd,ed)



    shares=0

    cash=start_val

    df_prices= get_data([symbol], dates= date_range)

    df_stock_movement= df_prices.diff(periods=-1)

    df_temp= df_prices.copy(deep=True)

    df_trades=df_temp.replace(df_temp, 0)

    df_trades['Cash']=0.0





    for i in range(df_stock_movement.shape[0]-1):

        

     

        date= df_stock_movement.index[i]

        

        

        price_diff= -1*(df_stock_movement.loc[date][symbol])

        sign= np.sign(price_diff)







        if sign==0:

            continue



        elif sign > 0:

            if shares+2000 <= 1000:

                shares, cash=make_trade('buy', symbol, date, df_prices, 2000, cash, shares, df_trades)

            elif shares+1000<= 1000:

                shares, cash= make_trade('buy', symbol, date, df_prices, 1000, cash, shares, df_trades)



        elif sign< 0 and shares-1000 >= -1000: 

            if shares-2000>=-1000:

                shares, cash= make_trade('sell', symbol, date, df_prices, 2000, cash, shares, df_trades)

            elif shares-1000 >= -1000:

                shares, cash= make_trade('sell', symbol, date, df_prices, 1000, cash, shares, df_trades)





        #if stock price goes up next day, buy today

        #if stock price goes down next day, sell today

        #else do nothing



    start_date= df_prices.index[0]

    end_date= df_prices.index[-1]

    buy_price= df_prices.loc[start_date]['JPM']

    end_price= df_prices.loc[end_date]['JPM']



    initial= start_val-(buy_price*1000)

    final= start_val-(1000*end_price)







    trades= df_trades[['JPM']]

    graph_port_vals(trades, sd, ed)



    portvals= get_port_value(trades)

    benchmark= get_benchmark_value(trades)



    get_portfolio_stats(portvals, benchmark)



    return trades

    

#############################################################################################################





def get_benchmark_value(df_trades, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):

    data= {'Symbol':['JPM','JPM'], 'Order':['BUY','BUY'],'Shares':[1000,0]}

    df_benchmark= pd.DataFrame(data=data, index={df_trades.index.min(), df_trades.index.max()})

    

    

    benchmark_val= compute_portvals(df_benchmark, 100000,0,0)

    return benchmark_val



#############################################################################################################

    



def get_order_book(df_trades):

    df_orders= df_trades.copy(deep=True)

  

    

    

    df_orders['Symbol']='JPM'

    df_orders=df_orders.rename(columns={'JPM':'Shares'})





    df_orders['Order'] = np.where(df_trades['JPM']>0, 'BUY', 'SELL')

    df_orders=df_orders[df_orders['Shares']!=0]

    df_orders[['Shares']]=abs(df_orders[['Shares']])

    df_orders=df_orders[['Symbol', 'Order', 'Shares']]



    return df_orders



#############################################################################################################



def get_port_value(df_trades, start_val= 10000, commission=0, impact=0):

    df_orders= get_order_book(df_trades)

    portval= compute_portvals(df_orders, 100000,0,0)

    return portval

    



 #############################################################################################################

   





def graph_port_vals(df_trades, sd, ed):

    plt.clf()

    pv= get_port_value(df_trades)

    pv=pv.to_frame('Value')



    bm= get_benchmark_value(df_trades)

    bm= bm.to_frame('Value')

  

    normed_bm= bm[['Value']]/bm.iloc[0]['Value']

    normed_pv= pv[['Value']]/pv.iloc[0]['Value']







    fig=plt.figure(figsize=(12, 7))



    plt.xlim(sd,ed)

    plt.plot(normed_pv,label='Port val',color='red')

    plt.title("Optimal Portfolio vs. Benchmark")

    plt.ylabel("Normalized Returns")

    plt.xlabel("Date")

    plt.plot(normed_bm,label='Benchmark',color='tab:purple')

    plt.grid(True)

    plt.xlim(right=pv.index[-1])



    plt.legend()

    plt.savefig('TOS_graph.png')

    



#############################################################################################################



def get_portfolio_stats(port_vals, benchmark):

    port_cum_rets, port_mean, port_std= assess_portfolio(port_vals)

    bench_cum_rets, bench_mean, bench_std= assess_portfolio(benchmark)

 



    stats=['Port_cum_rets: ', str(port_cum_rets), "\nPort_mean: ", str(port_mean), "\nPort_std: ", str(port_std)

    , '\n\nBench_cum_rets: ', str(bench_cum_rets), "\nBench_mean: ", str(bench_mean), "\nBench_std: ", str(bench_std)]



    with open('p6_results.txt', 'w') as f:

        f.writelines(stats)



