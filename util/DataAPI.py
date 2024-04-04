import numpy as np
import pandas as pd
import os

factor_path = r'../data/factors/'
raw_data_path = r'../data/factors/raw_data/'
date_list = pd.read_csv(os.path.join(raw_data_path, '调仓日期（月）.csv'), encoding='GBK')
date_list = date_list['日期'].tolist()
date_index = pd.to_datetime(date_list).strftime("%Y/%-m/%-d").tolist()


def winsorization(data):
    cap = np.nanpercentile(data, 99)  # 忽略NaN值
    flour = np.nanpercentile(data, 1)
    winsorized = data[(data > flour) & (data < cap)]
    return winsorized


def abnormal_turnover(tmp_factor):
    """
    异常换手率定义为：过去21个交易日的平均换手率和过去252个交易日的平均换手率的比值
    :param tmp_factor: 未处理的月平均换手率（自由流动股本计算）
    :return: 异常换手率
    """
    past_21 = tmp_factor.iloc[11:, ]  # 从第12个月开始
    past_252 = tmp_factor.rolling(12).mean()  # 过去12个月取均值
    past_252.dropna(inplace=True)
    stock_index = past_21.columns.tolist()
    date_index = past_21.index.tolist()
    with np.errstate(divide='ignore', invalid='ignore'):
        abnormal = pd.DataFrame(np.where(past_252 != 0, past_21 / past_252, np.nan),
                                columns=stock_index, index=date_index)

    abnormal.fillna(0, inplace=True)
    return abnormal


def read_df(file_path):
    """
    对iFind抓取dataframe做格式调整
    :param file_path: dataframe's location
    :return: time(row) * stock(col)
    """
    tmp_factor = pd.read_csv(file_path, dtype=object)
    tmp_factor = tmp_factor.set_index(tmp_factor.columns[0]).fillna(0)   # 第一列作为index
    tmp_factor = tmp_factor.astype(str).apply(lambda x: x.str.replace(",", "")).astype(float)
    tmp_factor = tmp_factor.T
    tmp_factor.index = pd.to_datetime(tmp_factor.index)
    tmp_factor.index = tmp_factor.index.strftime("%Y/%-m/%-d")
    tmp_factor.fillna(0, inplace=True)

    return tmp_factor


# 量价同步因子
def corr_ret_turned(window_size):
    ret_path = os.path.join(raw_data_path, '日收益率.csv')
    turned_path = os.path.join(raw_data_path, '日换手率.csv')
    print('正在读取：日收益率')
    ret = read_df(ret_path)
    print('正在读取：日换手率')
    turned = read_df(turned_path)

    # 创建一个空的dataframe储存相关性结果
    corr_results = pd.DataFrame(index=ret.index[window_size - 1:],
                                columns=ret.columns)

    # 滑动窗口计算相关性
    for i in range(len(ret) - window_size + 1):
        window_ret = ret.iloc[i:i + window_size]
        window_turned = turned.iloc[i:i + window_size]

        # 计算当前窗口内的两个dataframe之间的相关性
        corr_matrix = window_ret.corrwith(window_turned)
        corr_results.iloc[i] = corr_matrix

    corr_results.fillna(0, inplace=True)
    corr_results = corr_results.loc[date_index]
    corr_results = corr_results.T

    return corr_results


# 振幅的标准差因子
def high_low_std(window_size):
    """
    振幅的标准差因子 vol_high_low_std_3M 在全市场的有效性较好
    :param window_size: 窗口大小
    :return: 过去三个月振幅的标准差
    """
    high_low_path = os.path.join(raw_data_path, '日振幅.csv')
    high_low = read_df(high_low_path)
    std = high_low.rolling(window_size).std()
    std.dropna(inplace=True)
    std = std.loc[date_index]
    std = std.T

    return std


# 月平均大单净买入量
def net_buy_vol(window_size):
    vol_daily_path = os.path.join(raw_data_path, '日大单净买入量.csv')
    vol_daily = read_df(vol_daily_path)
    vol_mo = vol_daily.rolling(window_size).mean()
    vol_mo.dropna(inplace=True)
    vol_mo = vol_mo.loc[date_index]
    vol_mo = vol_mo.T

    return vol_mo


# 大单买入的位移路程比因子
def big_shift_dist(window_size):
    numerator_path = os.path.join(raw_data_path, '大单净买入金额.csv')
    denominator_path = os.path.join(raw_data_path, '大单主动净买入金额.csv')
    numerator = read_df(numerator_path)
    denominator = read_df(denominator_path)
    denominator = denominator.abs()
    num_window = numerator.rolling(window_size).sum()
    den_window = denominator.rolling(window_size).sum()
    num_window.fillna(0, inplace=True)
    den_window.fillna(0, inplace=True)
    stock_column = numerator.columns.tolist()
    date_row = numerator.index.tolist()
    with np.errstate(divide='ignore', invalid='ignore'):
        index = pd.DataFrame(np.where(den_window != 0, num_window / den_window, np.nan),
                             columns=stock_column, index=date_row)

    index.fillna(0, inplace=True)
    index = index.loc[date_index]
    index = index.T

    return index


if __name__ == '__main__':
    # corr_ret_turned_1M = corr_ret_turned(21)
    # corr_ret_turned_1M.to_csv(os.path.join(factor_path, 'corr_ret_turned_1M.csv'))
    # high_low_3M = high_low_std(63)
    # high_low_3M.to_csv(os.path.join(factor_path, 'high_low_3M.csv'))
    # net_buy_vol_1M = net_buy_vol(20)
    # net_buy_vol_1M.to_csv(os.path.join(factor_path, 'net_buy_vol_1M.csv'))
    big_shift_dist_I = big_shift_dist(20)
    big_shift_dist_I.to_csv(os.path.join(factor_path, 'big_shift_dist_I.csv'))