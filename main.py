import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import functools
import seaborn as sns
from random import randint
import scipy.stats
import math

STOCKS_LIST = [('BMY', (1/7)), ('CVS', (1/7)), ('DAL', (1/7)), ('FB', (1/7)), ('UG', (1/7)), ('SWKS', (1/7)), ('MUV', (1/7))]

START_DATE = "2013-01-01"
END_DATE = "2021-12-31"

#------------------------------- Question 1 ---------------------------------------------

def daily_stock_pct():
    returns = pd.DataFrame({})
    for t in STOCKS_LIST:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",start=START_DATE, end=END_DATE)
        data['return_%s' % (name)] = data['Close'].pct_change(1)*100
        returns = returns.join(data[['return_%s' % (name)]],
                               how="outer").dropna()

    print(returns)
    return returns

def daily_stock():
    returns = pd.DataFrame({})
    for t in STOCKS_LIST:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",start=START_DATE, end=END_DATE)
        data['return_%s' % (name)] = data['Close'].pct_change(1)
        returns = returns.join(data[['return_%s' % (name)]],
                               how="outer").dropna()

    # print(returns)
    return returns

def hist_stock():
    df = daily_stock_pct()
    for i in df:
        df_new = df[i]
        plt.hist(df_new,bins=50, color='#0080FF', edgecolor='black', linewidth=1.8, alpha=0.65)
        plt.title(i)
        plt.xlabel("Changes (Precentage)")
        plt.ylabel("Amount of Days")
        plt.show()

def calculations():
    df = daily_stock()
    for i in df:
        print(df[i])
        mean = sum(df[i])/len(df[i])
        print(f'Mean: {mean}')
        sd = 0
        for val in df[i]:
            sd += pow(val - mean,2)
        sd_value = math.sqrt(sd/(len(df[i])-1))
        print(f'Standard Deviation: {sd_value}')

        # correlation
        corr = pd.Series(df[i]).autocorr()
        print(f'Corr: {corr}')

    #cov matrix
    df = daily_stock_pct()
    stocks_values = []
    for current_stock in STOCKS_LIST:
        current_stock_values = df[f'return_{current_stock[0]}'].values
        stocks_values.append(current_stock_values)

    covariance_matrix = np.cov(current_stock_values)
    covariance_matrix *= 100

    # plot the heatmap
    sns.heatmap(covariance_matrix,
                xticklabels=[x[7:] for x in df.columns],
                yticklabels=[x[7:] for x in df.columns],
                cbar=True,
                annot=True,
                square=True,
                annot_kws={'size': 8, 'color': '#00A2FF', 'ha': 'center', 'va': 'top', 'weight': 'bold'})
    plt.title('Covariance Matrix\n\n', fontweight='bold', fontsize=8)
    plt.show()


calculations()