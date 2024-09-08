# -*- coding: utf-8 -*-

from tradingview_ta import TA_Handler, Exchange,Interval
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pdb
import time
import os
import copy

today = datetime.now()
data = []
look_back_time = 4000

def get_hist(look_back_time, n):
    start_time= time_change(look_back_time)
        
    data = yf.download('NVDA', start = start_time, end = today)
    chg_rate = data['Close'].pct_change()[1:]
    
    
    most_similar_sequence = find_most_similar_sequence(chg_rate, n, 5)
    print("Most similar sequence:", most_similar_sequence)
    
def euclidean_distance(x, y):
    """计算两个向量之间的欧氏距离"""
    return np.linalg.norm(x.values - y.values)

def find_most_similar_sequence(chg_rate, n, top_k):
    """在 chg_rate 中寻找与最后 n 个数据最相似的连续 N 个元素"""
    last_n_data = chg_rate[-n:]  # 最后 n 个数据
    most_similar_distance = []  # 最相似距离初始化为正无穷
    most_similar_sequence = []  # 最相似序列初始化为空
    
    seleted = []
    # 计算前 N-n 个元素中，与最后 n 个数据最相似的 N 个连续元素
    for i in range(len(chg_rate) - n):
        sequence = chg_rate[i:i+n]  # 获取当前连续的 N 个元素
        distance = euclidean_distance(sequence, last_n_data)  # 计算欧氏距离
        
        if len(most_similar_distance) < top_k:
            most_similar_distance.append(distance)
            most_similar_sequence.append(sequence)
            
            sorted_pairs = sorted(zip(most_similar_distance, most_similar_sequence), key = lambda x: x[0])
            
            most_similar_distance, most_similar_sequence = zip(*sorted_pairs)
            most_similar_distance, most_similar_sequence = list(most_similar_distance), list(most_similar_sequence)
        else:
            for j in range(0, top_k):
                if distance < most_similar_distance[j]:
                    most_similar_distance[j] = copy.deepcopy(distance)
                    most_similar_sequence[j] = copy.deepcopy(sequence)
                    break
    
    return most_similar_sequence

def time_change(per):
    # if i == 1:
    #     end = today
    # else:
    #     end = today - timedelta((i-1) * 30)
    start = today - timedelta(per)

    start = start.strftime('%Y-%m-%d')
    
    return start

get_hist(1000, 5)
# for i in range(1, 3):
    # pdb.set_trace()

# def get_all_stock_codes():




# tickers = [ 'YPF',
#             'BBAR',
#             'GGAL',
#             'MELI',
#             'SUPV',
#             'BMA',
#             'DESP',
#             'LOMA',
#             'PAM',
#             'CEPU',
#             'TGS',
#             'TEO',
#             'BIOX',
#             'CRESY',
#             'IRS',
#             'EDN'
#         ]

# tickers_data = []

# # Iterate through each ticker
# for ticker in tickers:
#     try:
#         # Retrieve data for the ticker from NYSE
#         data = TA_Handler(
#             symbol=ticker,
#             screener="america",
#             exchange="NYSE",
#             interval="1d"
#         )
#         data = data.get_analysis().summary
#         tickers_data.append(data)
        
#     except Exception as e:
#         # If no data is found for the ticker in NYSE, search in NASDAQ
#         print(f"No data found for ticker {ticker} in NYSE. Searching in NASDAQ...")
#         data = TA_Handler(
#             symbol=ticker,
#             screener="america",
#             exchange="NASDAQ",
#             interval="1d"
#         )
#         data = data.get_analysis().summary
#         tickers_data.append(data)

# print("Data successfully imported.")

# recommendations = []
# buys = []
# sells = []
# neutrals = []

# # Iterate through each data in tickers_data
# for data in tickers_data:
#     pdb.set_trace()
#     recommendation = data.get('RECOMMENDATION')
#     buy = data.get('BUY')
#     sell = data.get('SELL')
#     neutral = data.get('NEUTRAL')
    
#     recommendations.append(recommendation)
#     buys.append(buy)
#     sells.append(sell)
#     neutrals.append(neutral)

# data = {
#     'Ticker': tickers,
#     'Recommendations': recommendations,
#     'Buys': buys,
#     'Sells': sells,
#     'Neutrals': neutrals
# }

# df = pd.DataFrame(data)

# order_categories = {
#         'STRONG_BUT' : 5,
#         'BUT' : 4,
#         'NEUTRAL' : 3,
#         'SELL' : 2,
#         'STRONG_SELL' : 1
#     }

# df['order' ] = df['']


# tesla = TA_Handler(
#         symbol = 'TSLA',
#         screener = 'america',
#         exchange = 'NASDAQ',
#         interval = Interval.INTERVAL_1_MINUTE        
#     )

# # tesla.get_analysis()
# tesla.get_indicators()