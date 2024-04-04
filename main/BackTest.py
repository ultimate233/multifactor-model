# 构件因子投资组合并回测
# 运行前先把IC_test.py中的period值改为1
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantstats as qs
import IC_test as Ic
from itertools import combinations
from multiprocessing import Pool, Manager
import sys
import os
import time
warnings.filterwarnings('ignore')

# 选股数量
n = 5

# 历史指数数据
idx_return = pd.read_csv('../data/return/指数.csv', dtype=object)
idx_return.set_index('日期', inplace=True)
idx_return = idx_return.T
idx_return.index = pd.to_datetime(idx_return.index)
idx_return.index = idx_return.index.strftime("%Y/%-m/%-d")
idx_return = idx_return.astype(float)

# 计算指数累计收益率
idx_symbol = ['000001.SH', '399300.SZ']
idx_cum = {}
for symbol in idx_symbol:
    cum_ret = (1 + 0.01 * idx_return[symbol]).cumprod() * 100
    idx_cum[symbol] = cum_ret

# 下一期收益率
stock_return = Ic.get_period_return()


# 找最佳因子组合（非并行运算）
def find_best(factor_list, start_from=6):
    """
    对因子库中的因子进行组合，找到使得收益率最高的合成因子
    :param start_from: 因子数起点
    :param factor_list: 因子库
    :return:
    max_ret: 最高收益率
    max_combination: 最高收益率对应因子组合
    max_cum_ret: 对应累计收益率
    data: 与指数合并的累计收益率dataframe
    """
    max_ret = 0
    max_win_rate = 0
    max_win_rate_ret = None
    max_win_rate_comb = None
    max_combination = None
    max_cum_ret = None
    count = 0
    num = len(factor_list)
    for size in range(start_from, num+1):
        for combination in combinations(range(num), size):
            name = [Ic.factor_name_list[i] for i in combination]
            factors = [Ic.factor_df_list[i] for i in combination]
            IC_df = Ic.summarized_IC(name, factors)
            norm_df = Ic.composite(IC_df, name)
            # norm_df = norm_df * Ic.st
            norm_df = norm_df.replace(0, np.nan)
            long_side = norm_df.apply(lambda row: (row >= row.nlargest(n).iloc[-1]).astype(int), axis=1)
            selected_stock = np.sign(long_side).astype(int)

            # 计算组合收益率dataframe
            date_index = stock_return.index.tolist()

            # 股票池为收益率df和择股df的并集
            stock_return_column = stock_return.columns.tolist()
            selected_stock_column = selected_stock.columns.tolist()
            stock_pool = list(set(stock_return_column) & set(selected_stock_column))

            # 应用股票池
            port_return = pd.DataFrame(selected_stock[stock_pool].values * stock_return[stock_pool].values, columns=stock_pool,
                                       index=date_index)
            port_return = port_return.sum(axis=1) / n
            port_return.fillna(0, inplace=True)
            cum_return = (1 + port_return).cumprod()
            max_drawdown = qs.stats.max_drawdown(cum_return)

            # 选择从2019至今收益率最高的
            start = 2019
            idx = (start - 2010) * 12
            signal = cum_return.iloc[idx:] / cum_return.iloc[idx]
            win_rate = qs.stats.win_rate(signal)

            count += 1
            sys.stdout.write("\r" +
                             " I:" + str(count) +
                             " Comb:" + str(name))

            if win_rate > max_win_rate:
                max_win_rate = win_rate
                max_win_rate_ret = [signal[-1], cum_return[-1]]
                max_win_rate_comb = name

            if signal[-1] > max_ret:
                max_ret = signal[-1]
                max_combination = name
                max_cum_ret = cum_return

    # 最高胜率信息
    win_info = {
        '5yrs max_win_rate': str(max_win_rate),
        '5yrs return': str(max_win_rate_ret[0]),
        'total return': str(max_win_rate_ret[1]),
        'Comb': str(max_win_rate_comb)
    }

    # 上证指数、沪深300、组合收益dataframe
    data = pd.DataFrame(idx_cum).iloc[1:]
    data = pd.concat([data, max_cum_ret], axis=1)
    data.columns = [f'{symbol}_return' for symbol in idx_symbol + ['portfolio', ]]

    # 计算策略表现指标
    result = {}
    for symbol in idx_symbol + ["portfolio", ]:
        column_name = f"{symbol}_return"
        result[column_name] = []
        for stat in ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]:
            r = getattr(qs.stats, stat)(data[column_name])
            result[column_name].append(r)

    result = pd.DataFrame(result, index=["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"])
    result.to_excel('../result/策略表现指标.xlsx')

    return max_ret, max_combination, data, win_info


# 找最佳因子组合（并行运算）
def process_combination(combination, progress):
    name = [Ic.factor_name_list[i] for i in combination]
    factors = [Ic.factor_df_list[i] for i in combination]
    IC_df = Ic.summarized_IC(name, factors)
    norm_df = Ic.composite(IC_df, name)
    norm_df = norm_df.replace(0, np.nan)
    long_side = norm_df.apply(lambda row: (row >= row.nlargest(n).iloc[-1]).astype(int), axis=1)
    selected_stock = np.sign(long_side).astype(int)

    date_index = stock_return.index.tolist()

    stock_return_column = stock_return.columns.tolist()
    selected_stock_column = selected_stock.columns.tolist()
    stock_pool = list(set(stock_return_column) & set(selected_stock_column))

    port_return = pd.DataFrame(selected_stock[stock_pool].values * stock_return[stock_pool].values, columns=stock_pool,
                               index=date_index)
    port_return = port_return.sum(axis=1) / n
    port_return.fillna(0, inplace=True)
    cum_return = (1 + port_return).cumprod()
    max_drawdown = qs.stats.max_drawdown(cum_return)

    # Update progress
    progress.append(1)

    return name, cum_return, max_drawdown


def parallel_find_best(factor_list):
    max_ret = 0
    max_combination = None
    max_cum_ret = None
    num = len(factor_list)

    # Initialize progress manager
    manager = Manager()
    progress = manager.list()

    with Pool() as pool:
        results = pool.starmap(process_combination, [(combination, progress) for size in range(6, num + 1) for
                                                     combination in combinations(range(num), size)])

    for name, cum_return, max_drawdown in results:
        if cum_return[-1] > max_ret and math.fabs(max_drawdown) < 0.25:
            max_ret = cum_return[-1]
            max_combination = name
            max_cum_ret = cum_return

    data = pd.DataFrame(idx_cum).iloc[1:]
    data = pd.concat([data, max_cum_ret], axis=1)
    data.columns = [f'{symbol}_return' for symbol in idx_symbol + ['portfolio', ]]

    result = {}
    for symbol in idx_symbol + ["portfolio", ]:
        column_name = f"{symbol}_return"
        result[column_name] = []
        for stat in ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]:
            r = getattr(qs.stats, stat)(data[column_name])
            result[column_name].append(r)

    result = pd.DataFrame(result, index=["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"])
    result.to_excel('../result/策略表现指标.xlsx')

    return max_ret, max_combination, max_cum_ret, data


if __name__ == '__main__':

    # 开始计时
    start_time = time.time()

    max_ret, max_combination, data, win_info = find_best(Ic.factor_name_list, 6)
    print('', f'5yrs Return: {max_ret:.2f}', f'Combination: {max_combination}', sep='\n')
    for key, value in win_info.items():
        print(f'{key}: {value}')

    # 结束计时
    end_time = time.time()
    # 计算时间差
    elapsed_time = end_time - start_time
    print(f"程序执行时间为: {elapsed_time} 秒")

    # 收益率曲线图
    # 数据起始于2010.2
    start_year = 2022
    period = -1
    start_idx = (start_year - 2010) * 12
    end_idx = start_idx + period * 12 if period != -1 else period
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlabel('Time')
    ax.set_ylabel('Return')
    for symbol in idx_symbol + ['portfolio', ]:
        column_name = f'{symbol}_return'
        cum_ret = data[column_name].iloc[start_idx:end_idx] / data[column_name].iloc[start_idx]
        ax.plot(cum_ret.index, cum_ret.values)
    ax.legend(idx_symbol + ['portfolio'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

    plt.show()