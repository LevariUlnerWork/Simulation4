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
        data = ticker.history(interval="1d", start=START_DATE, end=END_DATE)
        data['return_%s' % (name)] = data['Close'].pct_change(1)*100
        returns = returns.join(data[['return_%s' % (name)]], how="outer").dropna()
    returns['Date'] = [date for date in returns.index]
    returns['ID'] = [i for i in range(returns.shape[0])]
    returns = returns.set_index('ID')
    print(returns)
    return returns


def daily_stock(STOCKS_LIST):
    returns = pd.DataFrame({})
    for t in STOCKS_LIST:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",start=START_DATE, end=END_DATE)
        data['return_%s' % (name)] = data['Close'].pct_change(1)
        returns = returns.join(data[['return_%s' % (name)]], how="outer").dropna()
    # print(returns)
    return returns
daily_stock(STOCKS_LIST)

def randWin(df_list, wanted_num_win=200):
    total_num_win = len(df_list)
    rand_num_list = []

    # Random Numbers
    while len(rand_num_list) < wanted_num_win:
        rand_num = randint(0,total_num_win-1)
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

def df_split_by_year(df=None):
    df = daily_stock(STOCKS_LIST)
    it = ""
    first_counter = 0
    last_counter = 0
    df_list = []
    for i in df.index:
        ts = str(i)[:4]
        if (it == ""):
            it = ts
        if( ts != it):
            df_list.append(df.iloc[first_counter:last_counter])
            first_counter = last_counter
            it = ts
        last_counter += 1
    df_list.append(df.iloc[first_counter:last_counter])
    return df_list

#------------------------------- Question 1 ---------------------------------------------

def hist_stock(STOCKS_LIST):
    df = daily_stock_pct(STOCKS_LIST) #not daily_stock??
    for i in df:
        df_new = df[i]
        plt.hist(df_new,bins=50, color='#0080FF', edgecolor='black', linewidth=1.8, alpha=0.65)
        plt.title(i)
        plt.xlabel("Changes (Precentage)")
        plt.ylabel("Amount of Days")
        plt.show()

def calculations(STOCKS_LIST, df_in):
    df = df_in
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
    plt.title('Covariance Matrix\n', fontweight='bold', fontsize=14)
    plt.show()


def Q1(STOCKS_LIST=STOCKS_LIST):
    print("------------------------------- Question 1 ---------------------------------------------")
    hist_stock(STOCKS_LIST)
    df = daily_stock(STOCKS_LIST)
    calculations(STOCKS_LIST,df)

    #Yearly:
    print('\n Yearly Data')
    df = daily_stock(STOCKS_LIST)
    df_year_list = df_split_by_year(df)
    year_list = range(2013,2022)


    for i in range (9):
        means = []
        sds = []
        print('\n year ' + str(year_list[i]) + '\n' )
        calculations(STOCKS_LIST, df_year_list[i])
        print(f'mean of means: {sum(means) / len(means)}')
        print(f'sd of sds: {sum(sds) / len(sds)}')


#------------------------------- Question 2 ---------------------------------------------

def Q2():
    print('\n\n')
    print("------------------------------- Question 2 ---------------------------------------------\n\n")
    df = daily_stock(STOCKS_LIST)
    WorkDaysPerYear = 252
    NumOfWin = 200
    NumOfDaysPerWin = int(WorkDaysPerYear * 3.5 / NumOfWin) + 1
    df_into_bins = np.array_split(df,df.shape[0]/int(NumOfDaysPerWin))
    new_df = sim_rand_win(df_into_bins, 100, 200)
    Q2partA(new_df)
    Q2partB(new_df)
    new_df_part_c = sim_rand_win(df_into_bins, 100, 200)
    Q2partC(new_df_part_c)

def Q2partA(new_df):
    print("------------------------------- Part a ---------------------------------------------\n")
    # calculate returns for each stock
    all_stocks_returns = stock_portfolio_for_each_stock(df=new_df)
    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Portfolio")

def Q2partB(new_df):
    print("\n------------------------------- Part b ---------------------------------------------\n")
    # calculate returns for each stock
    all_stocks_returns = structured_deposit_each_stock(new_df=new_df)

    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Structural deposit")

def Q2partC(new_df):
    print("\n------------------------------- Part c ---------------------------------------------\n")
    # calculate returns for each stock
    all_stocks_returns = stock_portfolio_for_each_stock(df=new_df)
    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Portfolio")

    # calculate returns for each stock
    all_stocks_returns = structured_deposit_each_stock(new_df=new_df)
    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Structural deposit")

def stock_portfolio_for_each_stock(df):
    '''

    :param df: a list of Data Frames for each simulation
    :return: list of portfolio each stock
    '''
    stocks_returns_list = []
    for current_df in df:
        for current_stock in STOCKS_LIST:
            current_stock_name = "return_" + current_stock[0]
            total_return_increment_by_one = list(map(lambda x: x + 1, current_df[current_stock_name]))
            total_return = functools.reduce(lambda x, y: x * y, total_return_increment_by_one)
            stocks_returns_list.append(total_return)

    return stocks_returns_list

def calculate_profits(all_stocks_returns, purpose):
    # get each stock returns for all runs
    bmy_returns = all_stocks_returns[0::7]
    cvs_returns = all_stocks_returns[1::7]
    dal_returns = all_stocks_returns[2::7]
    fb_returns = all_stocks_returns[3::7]
    ug_returns = all_stocks_returns[4::7]
    swks_returns = all_stocks_returns[5::7]
    muv_returns = all_stocks_returns[6::7]

    print(f"all stocks returns len is: {len(all_stocks_returns)}")

    # get profit from all stocks together for each of 100 time windows runs
    mean_return_all_stocks = []
    for i in range(len(bmy_returns)):
        mean_return_all_stocks.append(1 / 7 * (bmy_returns[i] + cvs_returns[i] + dal_returns[i] + fb_returns[i] + ug_returns[i] + swks_returns[i] + muv_returns[i]))

    # count number of runs in the return ranges and print each range probability
    stock_returns_ranges(mean_all_stocks=mean_return_all_stocks, purpose=purpose)

    # calculate mean for all stocks of the selected random runs and confidence interval
    mean_and_confidence_interval(mean_all_stocks=mean_return_all_stocks, purpose=purpose)

def stock_returns_ranges(mean_all_stocks: list, purpose: str):
    print(f'Sections a-d for the {purpose}:')

    res = len([x for x in mean_all_stocks if x <= 1]) / 200
    print(f"a. Probability of 0% profit: : {res * 100}%")

    res = len([x for x in mean_all_stocks if 1.019 <= x <= 1.021]) / 200
    print(f"b. Probability of 2% profit: {res * 100}%")

    res = len([x for x in mean_all_stocks if 1.021 <= x <= 1.2]) / 200
    print(f"b. Probability of (2%, 20%] profit: {res * 100}%")

    res = len([x for x in mean_all_stocks if 1.2 < x < 1.32]) / 200
    print(f"d. Probability of (20%, 32%) profit: {res * 100}%\n")

def mean_and_confidence_interval(mean_all_stocks, purpose):
    print(f'Section e-f for the {purpose}:')
    mean_result = np.mean(mean_all_stocks)
    mean_result_print = "{0:.5}".format((mean_result - 1) * 100)
    print(f'Median profit for chosen random period: {(np.median(mean_all_stocks) - 1) * 100}%')

    mean_all_stocks = sorted(mean_all_stocks)
    bot_10 = "{0:.4}".format((mean_all_stocks[int(len(mean_all_stocks) * 0.1)] - 1) * 100)
    top_10 = "{0:.4}".format((mean_all_stocks[int(len(mean_all_stocks) * 0.9)] - 1) * 100)

    print(f'profit for 10% decile : {bot_10}% , profit for 90% decile : {top_10}%')

    mean_result = np.mean(mean_all_stocks)
    expected_val = 100000 * mean_result
    print(f'Mean Expected Return: {(mean_result - 1) * 100}%')
    print(f'Expected Value of {purpose} is: {expected_val} ILS')
    print('\n')


    #Q1:

def structured_deposit_each_stock(new_df):
    all_stocks_returns = []
    for current_data_frame in new_df:
        stocks_dict = {'BMY': current_data_frame['return_BMY'].tolist(),
                               'CVS': current_data_frame['return_CVS'].tolist(),
                               'DAL': current_data_frame['return_DAL'].tolist(),
                               'FB': current_data_frame['return_FB'].tolist(),
                               'UG': current_data_frame['return_UG'].tolist(),
                               'SWKS': current_data_frame['return_SWKS'].tolist(),
                               'MUV': current_data_frame['return_MUV'].tolist()
                               }

        for current_stock_list in stocks_dict.values():
            current_stock_list = list(map(lambda x: x + 1, current_stock_list))

            accumulator = 1
            is_arrived_36 = False
            for index, current_return in enumerate(current_stock_list, 0):
                if index == len(current_stock_list) - 1:
                    break
                accumulator *= current_return

                if accumulator >= 1.32:
                    all_stocks_returns.append(1.02)
                    is_arrived_36 = True
                    break

            if not is_arrived_36:
                if accumulator < 1:
                    all_stocks_returns.append(1)
                else:
                    all_stocks_returns.append(accumulator)

    return all_stocks_returns

#------------------------------- Question 4 ---------------------------------------------

def Q4(isFive):

    print('\n\n')
    if (isFive == False):
        print("------------------------------- Question 4 ---------------------------------------------\n\n")
    else:
        print("------------------------------- Question 5 ---------------------------------------------\n\n")
    returns, dow_returns, stock_data = data_q4()

    seed = 100
    stock_num = 7 # num of stocks
    T = 880 # time
    N = 880
    Alpha = 1.7
    mean_each_stock_list = []
    sigma_each_stock = []

    for stock in STOCKS_LIST:
        name = stock[0]
        mean_each_stock_list.append(returns[name].mean()) #Mean
        sigma_each_stock.append(returns[name].std()) #std
    mean_each_stock_df = np.array(mean_each_stock_list)
    sigma_each_stock_df = np.array(sigma_each_stock)

    stocks_values = []
    for stock in STOCKS_LIST:
        stock_values = returns[stock[0]].values
        stocks_values.append(stock_values)

    Cov = np.cov(stocks_values)  # covariance matrix

    stocks, time = GBM (seed, stock_num, mean_each_stock_df, sigma_each_stock_df, Cov, T, N, Alpha, isFive)

    np.random.seed(seed)
    simul = int(200)

    SS = np.zeros([simul, stock_num, N])

    SS[0, :, :] = stocks

    for k in range(1, simul):
        seed = int(np.random.uniform(1, 2 ** 32 - 1, 1))
        SS[k, :, :] = GBM(seed, stock_num,mean_each_stock_df, sigma_each_stock_df, Cov, T, N, Alpha, isFive)[0]

    # Question 3:
    '''
    plt.figure(figsize=(16, 8))
    plt.title('Multidimensional Correlated GBM', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    for j in range(stock_num):
        plt.plot(time, SS[10, j, :],label=STOCKS_LIST[j][0])
    plt.legend()
    plt.show()
    '''

    sims_200 = []
    columns = {}
    i = 0
    for st in STOCKS_LIST:
        columns[i] = st[0]
        i += 1
    for sim in SS:
        # print(f'this is the SS: {SS}')
        # print(f'this is the sim: {sim}')
        # print(f'this is the sim transpose: {pd.DataFrame(sim).T}')
        sim_df = pd.DataFrame(sim)
        sim_df_transposed = sim_df.T
        sim_df_transposed.rename(columns=columns, inplace=True)
        for col in sim_df_transposed:
            sim_df_transposed[col] = sim_df_transposed[col].pct_change()
        sim_df_transposed = sim_df_transposed.iloc[1:, :]
        sims_200.append(sim_df_transposed)

    if(isFive == False):
        print('\n- - - - - - - - - - - Question 4 section a: - - - - - - - - - - - \n')
    else:
        print('\n- - - - - - - - - - - Question 5 section a: - - - - - - - - - - - \n')
    Q4partA(new_df=sims_200)

    if (isFive == False):
        print('\n- - - - - - - - - - - Question 4 section b: - - - - - - - - - - - \n')
    else:
        print('\n- - - - - - - - - - - Question 5 section b: - - - - - - - - - - - \n')
    Q4partB(new_df=sims_200)

def GBM(seed, stockPrice, exReturn, sigma, Cov, T, N, Alpha, isFive):

    np.random.seed(seed)
    dim = 7
    t = np.linspace(0., T, int(N))
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N)])
    S[:, 0] = stockPrice
    for i in range(1, int(N)):
        drift = (exReturn - 0.5 * sigma**2) * (t[i] - t[i-1])
        if(isFive == False):
            Z = np.random.normal(0., 1., dim)
        else:
            Z = np.array([Symmetric_Stable_distributions(Alpha) for _ in range(dim)])
        diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t

def data_q4():
    new_stocks_arr = [('BMY', 0.125), ('CVS', 0.125), ('DAL', 0.125), ('DOW', 0.125), ('FB', 0.125),
                             ('UG', 0.125), ('SWKS', 0.125), ('MUV', 0.125)]
    returns = pd.DataFrame({})
    dow_returns = pd.DataFrame({})
    stock_data = []
    for t in new_stocks_arr:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval='1d',
                              start="2013-01-01", end="2021-12-25")
        data['date'] = [date for date in data.index]
        data[name] = data['Close'].pct_change()
        if name == 'DOW':
            dow_returns = dow_returns.join(data[[name]],
                                           how="outer").dropna()
        else:
            f_val = data['Close'].iloc[[0]].values.tolist()[0]
            stock_data.append(f_val)
            returns = returns.join(data[[name]],
                                   how="outer").dropna()
    return returns, dow_returns, stock_data

def Symmetric_Stable_distributions(Alpha):
    half_pi = np.pi/2
    U = np.random.uniform(half_pi*-1, half_pi)
    E = np.random.standard_exponential()
    return (math.sin(Alpha*U)) * (math.cos(U) ** (-1/Alpha)) * ((math.cos(U*(1-Alpha)) / E) ** ((1/Alpha)-1))

def stock_portfolio_for_each_stock4(df):
    '''

    :param df: a list of Data Frames for each simulation
    :return: list of portfolio each stock
    '''
    stocks_returns_list = []
    for current_df in df:
        for current_stock in STOCKS_LIST:
            current_stock_name = current_stock[0]
            total_return_increment_by_one = list(map(lambda x: x + 1, current_df[current_stock_name]))
            total_return = functools.reduce(lambda x, y: x * y, total_return_increment_by_one)
            stocks_returns_list.append(total_return)

    return stocks_returns_list

def structured_deposit_each_stock4(new_df):
    all_stocks_returns = []
    for current_data_frame in new_df:
        stocks_dict = {'BMY': current_data_frame['BMY'].tolist(),
                               'CVS': current_data_frame['CVS'].tolist(),
                               'DAL': current_data_frame['DAL'].tolist(),
                               'FB': current_data_frame['FB'].tolist(),
                               'UG': current_data_frame['UG'].tolist(),
                               'SWKS': current_data_frame['SWKS'].tolist(),
                               'MUV': current_data_frame['MUV'].tolist()
                               }

        for current_stock_list in stocks_dict.values():
            current_stock_list = list(map(lambda x: x + 1, current_stock_list))

            accumulator = 1
            is_arrived_36 = False
            for index, current_return in enumerate(current_stock_list, 0):
                if index == len(current_stock_list) - 1:
                    break
                accumulator *= current_return

                if accumulator >= 1.32:
                    all_stocks_returns.append(1.02)
                    is_arrived_36 = True
                    break

            if not is_arrived_36:
                if accumulator < 1:
                    all_stocks_returns.append(1)
                else:
                    all_stocks_returns.append(accumulator)

    return all_stocks_returns

def Q4partA(new_df):
    print("------------------------------- Part a ---------------------------------------------\n")
    # calculate returns for each stock
    all_stocks_returns = stock_portfolio_for_each_stock4(df=new_df)
    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Portfolio")

def Q4partB(new_df):
    print("\n------------------------------- Part b ---------------------------------------------\n")
    # calculate returns for each stock
    all_stocks_returns = structured_deposit_each_stock4(new_df=new_df)

    # calculate all profits and prints results
    calculate_profits(all_stocks_returns, "Structural deposit")

#------------------------------- Main ---------------------------------------------
# Q1(STOCKS_LIST=STOCKS_LIST)
Q2()
# Q4(False)
# Q4(True)












