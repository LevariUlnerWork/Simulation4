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

#------------------------------- Other Func ---------------------------------------------
def daily_stock_pct(STOCKS_LIST):
    returns = pd.DataFrame({})
    for t in STOCKS_LIST:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",start=START_DATE, end=END_DATE)
        data['return_%s' % (name)] = data['Close'].pct_change(1)*100
        returns = returns.join(data[['return_%s' % (name)]],
                               how="outer").dropna()
    returns['Date'] = [date for date in returns.index]
    returns['ID'] = [i for i in range(returns.shape[0])]
    returns = returns.set_index('ID')
    # print(returns)
    return returns

def daily_stock(STOCKS_LIST):
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

def randWin(df_list, wanted_num_win=200):
    total_num_win = len(df_list)
    rand_num_list = []

    # Random Numbers
    while len(rand_num_list) < wanted_num_win:
        rand_num = randint(0,total_num_win)
        if(rand_num not in rand_num_list):
            rand_num_list.append(rand_num)

    # 200 windows concat:
    df_res = pd.concat([df_list[bin_index] for bin_index in rand_num_list],ignore_index=True)

    return df_res

def sim_rand_win(df_list,num_sim = 100,num_win = 200):
    df_list_res = []
    for i in range (num_sim):
        df_list_res.append(randWin(df_list,num_win))
    return df_list_res


#------------------------------- Question 1 ---------------------------------------------

def hist_stock(STOCKS_LIST):
    df = daily_stock_pct(STOCKS_LIST)
    for i in df:
        df_new = df[i]
        plt.hist(df_new,bins=50, color='#0080FF', edgecolor='black', linewidth=1.8, alpha=0.65)
        plt.title(i)
        plt.xlabel("Changes (Precentage)")
        plt.ylabel("Amount of Days")
        plt.show()

def calculations(STOCKS_LIST):
    df = daily_stock(STOCKS_LIST)
    for i in df:
        print(i[7:]," statistics:\n")
        mean = sum(df[i])/len(df[i])
        print(f'Mean: {mean}')
        sd = 0
        for val in df[i]:
            sd += pow(val - mean,2)
        sd_value = math.sqrt(sd/(len(df[i])-1))
        print(f'Standard Deviation: {sd_value}')

        # correlation
        corr = pd.Series(df[i]).autocorr()
        print(f'Corr: {corr}','\n')

    #cov matrix
    stocks_values = []

    for current_stock in STOCKS_LIST:
        current_stock_values = df[f'return_{current_stock[0]}'].values
        stocks_values.append(current_stock_values)

    covariance_matrix = np.cov(stocks_values)
    covariance_matrix *= 100

    # plot the heatmap
    sns.heatmap(covariance_matrix,
                xticklabels=[x[7:] for x in df.columns],
                yticklabels=[x[7:] for x in df.columns],
                cbar=True,
                annot=True,
                square=True,
                annot_kws={'size': 9, 'color': 'red', 'ha': 'center', 'va': 'top', 'weight': 'bold'})
    plt.title('Covariance Matrix\n\n', fontweight='bold', fontsize=14)
    plt.show()

def Q1(STOCKS_LIST=STOCKS_LIST):
    print("------------------------------- Question 1 ---------------------------------------------")
    hist_stock(STOCKS_LIST)
    calculations(STOCKS_LIST)




#------------------------------- Question 2 ---------------------------------------------

def Q2():
    df = daily_stock(STOCKS_LIST)
    WorkDaysPerYear = 252
    NumOfWin = 200
    NumOfDaysPerWin = int(WorkDaysPerYear * 3.5 / NumOfWin) + 1
    df_into_bins = np.array_split(df,df.shape[0]/int(NumOfDaysPerWin))
    new_df = sim_rand_win(df_into_bins, 1, 200)


    #Part A:

    #Q1:

#------------------------------- Main ---------------------------------------------
Q2()













