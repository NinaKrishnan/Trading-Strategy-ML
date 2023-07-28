

from turtle import up

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

import matplotlib.patches as patch

from marketsimcode import compute_portvals, assess_portfolio

from util import get_data





#############################################################################################################





def author():

    return 'nkrishnan42'



#############################################################################################################





def get_prices(symbol, sd, ed, window, col='Adj Close'):

    delta= dt.timedelta(window)

    

    sd-= delta

   

    date_range= pd.date_range(sd, ed)



    df_prices= get_data([symbol], date_range, addSPY=True, colname=col)

    df_prices= df_prices[symbol]



    return df_prices



#############################################################################################################



def get_sma(df_prices, window):

    sma= df_prices.rolling(window, 1).mean()

    return sma



#############################################################################################################



def truncate_df(df, sd):

    end= get_start_index(df, sd)

    return df.drop(df.index[range(0, end)])



#############################################################################################################    



def get_start_index(df, sd):

    return df.index.get_loc(sd, method='nearest')



#############################################################################################################



def get_bb_indicators(df_prices, sd, ed, window):

    

    rolling_mean= get_sma(df_prices, window)



    rolling_std= df_prices.rolling(window, 1).std()

    

    rolling_mean= truncate_df(rolling_mean, sd)

    rolling_std= truncate_df(rolling_std, sd)



    upper_band= rolling_mean + (2*rolling_std)

    lower_band= rolling_mean - (2*rolling_std)





    bandwidth= (upper_band-lower_band)/rolling_mean

    bandwidth=truncate_df(bandwidth, sd)



    trade_signal= (df_prices-rolling_mean)/(2*rolling_std)

    trade_signal= truncate_df(trade_signal, sd)



    prices= truncate_df(df_prices, sd)



    return (prices, (trade_signal.to_frame('Value')), upper_band, lower_band, rolling_mean, rolling_std, bandwidth)



#############################################################################################################



def normalize(df, lower, upper, min_val, max_val):



    return ((upper-lower) * ((df-min_val)/(max_val-min_val))+lower)



#############################################################################################################



def bb_percentage(syms, sd, ed, window):

    df_prices= get_prices(syms, sd, ed, window)

    prices, bb_ind, upper, lower, mean, std, bandwidth= get_bb_indicators(df_prices, sd, ed, window)

    upper= upper.to_frame('Value')

    lower= lower.to_frame('Value')

    mean= mean.to_frame('Value')

    

    a= -1.3

    b= 1.3



    min_b= min(bb_ind['Value'])

    max_b= max(bb_ind['Value'])



    norm= (b-a) * ((bb_ind-min_b)/(max_b-min_b))+a





    

    plt.clf()



    fig,ax= plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)



    plt.xlim(sd,ed)



    top= ax[0]

    bottom= ax[1]

    '''



    top.plot(mean, label='Simple Moving Average, span = 20 days')

    top.plot(upper, label='SMA + 2σ (Overbought')

    top.plot(lower, label='SMA - 2σ (Oversold')

    

    top.set_title('SMA with Bollinger Bands: JPM')

    top.set_xlabel('Stock Price')

    top.set_ylabel('Date')

    top.legend()

    '''



  

    x1= upper.index

    y1= np.reshape(upper[['Value']].values, upper[['Value']].values.shape[0])

    y2= np.reshape(lower[['Value']].values, lower[['Value']].values.shape[0])

    







    top.fill_between(x1, y1, y2, alpha=0.2, label='Band Width')

    top.plot(mean, label='Simple Moving Average, span = 20 days')

    top.plot(upper, label='SMA + 2σ (Overbought')

    top.plot(lower,label='SMA - 2σ (Oversold')

    top.set_title('SMA with Bollinger Bands: JPM')

    top.set_xlabel('Stock Price')

    top.set_ylabel('Date')

    top.legend()



    d= abs(upper-lower)/mean

    dy= np.reshape(d[['Value']].values, d[['Value']].values.shape[0])



    bottom.bar(d.index, dy, color='r', label='Distance between bands/SMA')

    bottom.set_xlabel('Date')

    bottom.set_ylabel('Band Width')

    bottom.set_title('Band Width (Distance between Bands): JPM')

    bottom.legend()



    '''



    bottom.plot(norm, label= 'BB %')

    bottom.axhline(1, color='r',linestyle='--', alpha=1, label='100% (Overbought, price > upper band)')

    bottom.axhline(-1, color='r',linestyle='--', alpha=1, label= '-100% (Oversold, price < lower band)')

  

    bottom.set_title('Bollinger Band Percentages: JPM')

    bottom.set_xlabel('Bollinger Band Percent')

    bottom.set_ylabel('Date')

    bottom.legend()

    '''



    plt.savefig('bandwidth.png')







    

    #plt.savefig("bb_percent.png")

    plt.close()



#############################################################################################################



def golden_cross(symbol, sd, ed):

    df_prices= get_prices(symbol, sd, ed, 200)

    

    small_SMA= truncate_df(get_sma(df_prices, 50), sd)

    large_SMA= truncate_df(get_sma(df_prices, 200), sd)

    small_SMA= small_SMA.to_frame('Value')

    large_SMA= large_SMA.to_frame('Value')



    df_prices=truncate_df(df_prices, sd)



    fig=plt.figure(figsize=(12, 7))



    plt.xlim(sd,ed)







    plt.clf()



    plt.plot(df_prices, label='JPM stock prices')

    plt.plot(small_SMA, label= '50-day moving avg.')

    plt.plot(large_SMA, label='200-day moving avg.')

    plt.legend()



    plt.title('Golden Cross/Death Cross: JPM')

    plt.xlabel('Stock Price')

    plt.ylabel('Date')



    plt.savefig('golden_cross.png')

    plt.close()



#############################################################################################################



def macd(syms, sd, ed):

    df_prices= get_prices(syms, sd, ed,26)

    ema_12= get_ema(df_prices, 12)

    ema_26= get_ema(df_prices, 26)



    macd= (ema_12-ema_26)

    signal= get_ema(macd, 9)

    plt.clf()



    fig,ax= plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

  

    top=ax[0]

    bottom=ax[1]



   # fig=plt.figure(figsize=(12, 7))



    plt.xlim(sd,ed)



    '''

    top.plot(df_prices, label='Stock Prices')

    top.plot(ema_12, label='12 day EMA')

    top.plot(ema_26, label='26 day EMA')

    top.set_title('Price with 12-day and 26-day EMA')

    top.set_ylabel('Stock Price')

    top.set_xlabel('Date')

    top.legend()

    '''



    top.plot(macd, label='MACD (span = 12 days & 26 days)')

    top.plot(signal,label='Signal (span= 9 days)')

    top.set_title("Moving Average Convergence Divergence")

    top.set_ylabel('MACD Value')

    top.set_xlabel('Date')

    top.legend()

    top.grid()



    diffs= macd-signal

    diffs= diffs.to_frame('Value')



    hist= np.reshape(diffs[['Value']].values, diffs[['Value']].values.shape[0])

    colors= np.where(hist > 0, 'g', 'r')

    label= np.where(hist>0, 'Uptrend', 'Downtrend')



    bottom.bar(diffs.index, hist, color= colors, align='center', width=1)

    bottom.set_ylim((-2,2))

    bottom.grid()

    bottom.set_title('Distance between MACD and Signal')

    bottom.set_ylabel('Distance')

    bottom.set_xlabel('Date')

    red= patch.Patch(color='green', label='Upward Trend (MACD > Signal)')

    green= patch.Patch(color='red', label='Downward Trend (MACD < Signal')

    bottom.legend(handles=[red, green])



   

    



    plt.savefig('macd.png')

    plt.close()



#############################################################################################################    



def plot_ema(syms,sd,ed,window, ema, label, ax):

    df_prices= get_prices(syms,sd,ed,window)

    ema= get_ema(df_prices, window)



    plt.clf()

    fig=plt.figure(figsize=(12, 7))



    plt.xlim(sd,ed)



    plt.plot(df_prices, label='label')

    plt.plot(ema, label='EMA (span = 20 days')

    plt.ylabel('Prices')

    plt.xlabel('Date')

    plt.title('Exponential Moving Average: JPM')



    plt.savefig('ema.png')

    plt.close()



#############################################################################################################



def get_ema(df_prices, window):

    smoother= get_multiplier(window)

    ema= df_prices.ewm(span=window, adjust=True).mean()

    return ema



#############################################################################################################



def get_multiplier(window):

    return 2/(window+1)



#############################################################################################################



def stochastic_oscillator(sym, sd, ed):

    df_prices= get_prices(sym, sd, ed, 14)



    k_window= 14

    sma_window= 3



    high= df_prices.rolling(k_window, 1).max()

    low= df_prices.rolling(k_window, 1).min()



    percent_K= ((df_prices.values-low)*100/(high-low))

    percent_D= get_sma(percent_K, sma_window)



    percent_K= truncate_df(percent_K.to_frame('Value'), sd)

    percent_D= truncate_df(percent_D.to_frame('Value'), sd)



    minK= min(percent_K['Value'])

    maxK= max(percent_K['Value'])



    minD= min(percent_D['Value'])

    maxD= max(percent_D['Value'])



    normed_K= normalize(percent_K, 0,102, minK, maxK)

    normed_D= normalize(percent_D, 0,102, minD, maxD)



    df_prices=df_prices.to_frame('Value')



    plt.clf()

    fig=plt.figure(figsize=(12, 7))



    plt.xlim(sd,ed)



    plt.plot(normed_K, label='%K')

    plt.plot(normed_D, label='%D')

    plt.axhline(80, color='r',linestyle='--', alpha=1, label='80% (Overbought)')

    plt.axhline(20, color='g',linestyle='--', alpha=1, label='20% (Oversold)')



    plt.title('Stochastic Oscillator: JPM')

    plt.xlabel('Date')

    plt.ylabel('Stochastic Indicator Percentage')

    plt.ylim((-5,107))

    plt.legend()



    plt.savefig('stochastic_o.png')



#############################################################################################################



def rsi(syms, sd, ed, window):

    df_prices= get_prices(syms, sd, ed, window)

    df_prices=df_prices.to_frame('Value')



    df_price_diffs= df_prices.diff()



    df_gains= df_price_diffs.clip(lower=0)

    df_losses= (df_price_diffs.clip(upper=0)) *-1



    avg_gains= get_ema(df_gains, window-1)

    avg_losses= get_ema(df_losses, window-1)





    relative_strength= (avg_gains/avg_losses)+1

    RSI= 100-(100/relative_strength)

  

    fig,ax= plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    top= ax[0]

    bottom= ax[1]



    plt.xlim(sd,ed)





   # top.plot(df_prices, label='Stock Prices')

    top.plot(df_gains, label= 'Relative High Prices')

    top.plot(df_losses, label= 'Relative Low Prices')

    top.plot(avg_gains, label= 'Relative High Prices EMA')

    top.plot(avg_losses, label='Relative Low Prices EMA')

    top.set_xlabel('Date')

    top.set_ylabel('Relative Stock Price')

    top.set_title('Relative High/Low Prices, Relative High/Low EMA: JPM')

    top.legend()



    bottom.plot(RSI, label= 'RSI')

    bottom.axhline(70, color='r', linestyle= '--', alpha=1, label='70% (Overbought)')

    bottom.axhline(30, color='r', linestyle='--', alpha=1, label='30% (Oversold)')

    bottom.set_ylabel('Relative Strength Index')

    bottom.set_xlabel('Date')

    bottom.set_title('Relative Strength Index: JPM')

    bottom.legend()





    plt.savefig('rsi.png')



#############################################################################################################





def plot_indicators(syms, sd, ed):

    bb_percentage(syms, sd, ed, 40)

    golden_cross(syms, sd, ed)

    macd(syms, sd, ed)

    rsi(syms, sd, ed, 14)

    stochastic_oscillator(syms, sd, ed)



#############################################################################################################



