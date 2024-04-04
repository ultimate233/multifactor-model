import pandas as pd
import numpy as np
import os
import math
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from datetime import datetime
import util.DataAPI as Api
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
zh_font1 = matplotlib.font_manager.FontProperties(fname=r"../font/SourceHanSansSC-Normal.otf")

factor_path = r'../data/factors/'
return_path = r'../data/return/'
raw_data_path = r'../data/factors/raw_data/'
result_dir = r'../result/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def get_period_return(n=1):
    """
    :param n: 区间长度
    :return: 区间收益率
    """

    # 整理收盘价dataframe
    df = pd.read_csv(os.path.join(return_path, '2015-2024周收盘价.csv'), dtype=object)
    df = df.set_index('日期').fillna(0)
    df = df.astype(str).apply(lambda x: x.str.replace(",", "")).astype(float)
    df = df.T
    df.index = pd.to_datetime(df.index)
    df.index = df.index.strftime("%Y/%-m/%-d")

    # 整体向下平移n个单位
    begin = df.copy().shift(n).dropna()
    end = df.iloc[n:, :]
    stock_index = end.columns.tolist()
    date_index = end.index.tolist()
    with np.errstate(divide='ignore', invalid='ignore'):  # 阻止除以零的警告
        period_return = pd.DataFrame(np.where(begin.values != 0, end.values / begin.values - 1, np.nan),
                                     columns=stock_index, index=date_index)
    # 将NaN替换为0
    period_return.fillna(0, inplace=True)

    # 删除0多的股票
    zero_counts = period_return.eq(0).sum()
    selected_stocks = zero_counts[zero_counts < 0.4 * len(period_return)].index.tolist()
    period_return = period_return[selected_stocks]

    return period_return


# 设定观测间隔（以月记）
period = 1

# 下一期收益率 n=1
next_period_return = get_period_return()
fund_pool = next_period_return.columns.tolist()

# 是否ST
st = pd.read_csv(os.path.join(raw_data_path, 'st.csv'))
st = st.set_index(st.columns[0]).fillna(0)
st = st.replace(['否', '是'], [1, 0])
st = st.T
st.index = pd.to_datetime(st.index)
st.index = st.index.strftime("%Y/%-m/%-d")
st = st.iloc[:-1].filter(items=fund_pool)

# 读取因子值
factor_csv_list = os.listdir(factor_path)
factor_csv_list = [x for x in factor_csv_list if x[-3:] == "csv"]
factor_name_list = [x[:-4] for x in factor_csv_list]
factor_df_list = []
for i in tqdm(range(len(factor_csv_list)), desc='Reading'):
    file_path = os.path.join(factor_path, factor_csv_list[i])
    tmp_factor = Api.read_df(file_path)
    tmp_factor = tmp_factor.filter(items=fund_pool)
    tmp_factor.fillna(0, inplace=True)
    tmp_factor = tmp_factor.iloc[11:] if factor_name_list[i] == '月平均换手率' else tmp_factor
    # 舍弃最靠近现在的部分，对齐数据
    tmp_factor = tmp_factor.iloc[:-period, :]
    factor_df_list.append({factor_name_list[i]: tmp_factor})


# IC测试
# 方法是对所有股票的当期因子值、后一期收益率（横截面数据）取平均
# 对两个数列计算斯皮尔曼相关系数，得到IC时间序列
# 对IC时间序列计算每个因子的IC mean，IC std，和 IC_IR= IC_mean/IC_std
def get_RankIC(factor_name_list, factor_df_list):
    # 后1年收益率
    ICmean_list = np.zeros(len(factor_name_list))
    ICstd_list = np.zeros(len(factor_name_list))
    ICIR_list = np.zeros(len(factor_name_list))
    average_sample = np.zeros(len(factor_name_list))
    for i in range(len(factor_name_list)):
        factor_name = factor_name_list[i]
        factor_df = factor_df_list[i][factor_name]
        tmp_corr = np.zeros(factor_df.shape[0])
        sum_sample = 0
        for day in range(factor_df.shape[0]):
            # 取横截面数据
            return_series = next_period_return.iloc[day, :] * st.iloc[day, :]
            factor_series = factor_df.iloc[day, :] * st.iloc[day, :]
            # 只留下非0列
            return_series_nonzero = return_series[return_series != 0]
            factor_series_nonzero = factor_series[factor_series != 0]
            winsorized_return = pd.DataFrame(Api.winsorization(return_series_nonzero))
            winsorized_factor = pd.DataFrame(Api.winsorization(factor_series_nonzero))
            # 只留下共同部分
            combine = pd.merge(winsorized_return, winsorized_factor,
                               how='inner', left_index=True, right_index=True)
            # 将因子值和收益率的共同部分转化为array
            return_list = combine.iloc[:, 0].to_list()
            factor_list = combine.iloc[:, 1].to_list()
            num_sample = len(return_list)

            # 将每一期的样本量加和，用于计算平均样本量
            sum_sample += num_sample
            rank_ic = stats.spearmanr(return_list, factor_list)[0]
            tmp_corr[day] = 0 if math.isnan(rank_ic) else rank_ic

        average_sample[i] = sum_sample / factor_df.shape[0]
        ICmean = np.mean(tmp_corr)
        ICstd = np.std(tmp_corr)
        ICIR = ICmean / ICstd
        # 填入列表
        ICmean_list[i] = ICmean
        ICstd_list[i] = ICstd
        ICIR_list[i] = ICIR

    IC_summary = pd.DataFrame(
        [ICmean_list, ICstd_list, ICIR_list, average_sample]
    )
    return IC_summary


# 分位数组合测试
def quantile_test(IC_df):
    target_dict = {}
    for i in range(len(factor_name_list)):
        # 构造T*5的矩阵
        target = np.zeros([next_period_return.shape[0], 5])
        factor_name = factor_name_list[i]
        factor_df = factor_df_list[i][factor_name]
        IC_mean = IC_df.loc[factor_name, 'ICmean']
        for day in range(factor_df.shape[0]):
            # 比较分位数组合净值
            factor_value = factor_df.iloc[day, :] * st.iloc[day, :]
            return_value = next_period_return.iloc[day, :] * st.iloc[day, :]
            # 只留下非0列
            return_series_nonzero = return_value[return_value != 0].dropna()
            factor_series_nonzero = factor_value[factor_value != 0].dropna()
            winsorized_return = pd.DataFrame(Api.winsorization(return_series_nonzero))
            winsorized_factor = pd.DataFrame(Api.winsorization(factor_series_nonzero))
            # 只留下共同部分
            combine = pd.merge(winsorized_return, winsorized_factor,
                               how='inner', left_index=True, right_index=True)
            factor_series = combine.iloc[:, 1]
            return_series = combine.iloc[:, 0]
            # 用np.sign调整符号
            quantile = pd.qcut(np.sign(IC_mean) * factor_series, q=5, labels=False, duplicates='drop')
            quantile_df = pd.concat([factor_series, return_series, quantile], axis=1)
            quantile_df.columns = [factor_name, 'return_value', 'quantile']
            quantile_df = quantile_df.sort_values(by='quantile', ascending=True)
            target[day] = quantile_df.groupby('quantile')['return_value'].mean().tolist()

        tmp = pd.DataFrame(target, columns=['Group 1', 'Group 2', 'Group 3', 'Group 4',
                                            'Group 5'], index=next_period_return.index)
        target_dict[factor_name] = (1 + tmp).cumprod()
    return target_dict


# 因子合成
def composite(IC_df, name):
    weight = IC_df.iloc[:, 0] / sum(IC_df.iloc[:, 0])
    norm_factor_list = []
    for i in range(len(name)):
        factor_name = name[i]
        factor_idx = factor_name_list.index(factor_name)
        factor_df = factor_df_list[factor_idx][factor_name]
        norm_factor = factor_df.copy() * st
        # 标准化为排名百分比
        nonzero = norm_factor[norm_factor != 0]
        norm_factor[norm_factor != 0] = nonzero.apply(lambda x: x.rank(pct=True), axis=1)
        norm_factor = np.sign(IC_df.iloc[i, 0]) * norm_factor
        norm_factor_list.append({factor_name: pd.DataFrame(norm_factor, dtype=float)})

    # 保证权重为正数，IC_mean为负已有调整
    norm_df = sum([math.fabs(weight[i]) * norm_factor_list[i][name[i]] for i in range(len(name))])

    return norm_df


def summarized_IC(factor_name_list, factor_df_list):
    col_names = ['ICmean', 'ICstd', 'ICIR', 'AvgSample']
    IC_df = get_RankIC(factor_name_list, factor_df_list).T
    IC_df.columns = col_names
    IC_df.index = factor_name_list
    return IC_df


if __name__ == '__main__':
    # IC测试
    # IC_df: row - 因子名, col - 测量值
    IC_df = summarized_IC(factor_name_list, factor_df_list)
    print(f'区间间隔为n={period}')
    print(IC_df)
    os.chdir(result_dir)
    IC_df.to_excel("IC测试.xlsx")

    # 分位数组合测试 - 画图
    target = quantile_test(IC_df)
    for factor in factor_name_list:
        df = target[factor]
        dates = [datetime.strptime(date_str, '%Y/%m/%d').date() for date_str in df.index.tolist()]
        plt.figure(figsize=(10, 5))
        for group in df.columns:
            plt.subplot(1, 2, 1)
            plt.plot(dates, df[group], label=group)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.subplot(1, 2, 2)
            plt.bar(group, np.mean(df[group]), label=group)
        plt.legend()
        plt.title(factor, fontproperties=zh_font1)
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.show()

    with pd.ExcelWriter(os.path.join(result_dir, '分位数组合测试.xlsx'), engine='xlsxwriter') as writer:
        for factor in factor_name_list:
            target[factor].to_excel(writer, sheet_name=factor, index=True)

    # 因子合成
    norm_df = composite(IC_df, factor_name_list)
    composed = summarized_IC(['合成因子'], [{'合成因子': norm_df}])
    print(composed)

