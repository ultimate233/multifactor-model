# 只需三个因子，周成交量、周平均换手率、两个月振幅标准差
# 周成交量、周平均换手率均直接拉取

import os
import IC_test as Ic
import numpy as np
import pandas as pd
import math
import warnings
import akshare as ak
warnings.filterwarnings('ignore')

latest_path = r'../data/latest/'
result_path = r'../result/record/'

month = 202403

# 读取因子值
factor_name_list = ['tapi', 'cci', 'vma', 'srmi', 'cdp', 'expma']
factor_df_list = []
data = pd.read_csv(os.path.join(latest_path, f'{month}.csv'), dtype=object)
data = data.set_index(data.columns[0]).fillna(0)
data = data.astype(str).apply(lambda x: x.str.replace(",", "")).astype(float)


for i in range(len(factor_name_list)):
    factor_name = factor_name_list[i]
    tmp_factor = pd.DataFrame(data[factor_name].values).T
    tmp_factor.columns = data.index.tolist()
    tmp_factor = tmp_factor.filter(items=Ic.fund_pool)
    tmp_factor.fillna(0, inplace=True)
    tmp_factor = tmp_factor.iloc[11:] if factor_name_list[i] == '月平均换手率' else tmp_factor

    factor_df_list.append({factor_name_list[i]: tmp_factor})

# 获取历史数据IC测试的结果(确定权重)
IC_df = Ic.summarized_IC(Ic.factor_name_list, Ic.factor_df_list)
weight = IC_df.iloc[:, 0] / sum(IC_df.iloc[:, 0])
norm_factor_list = []

# 用本月数据计算合成因子值
for i in range(len(factor_name_list)):
    factor_name = factor_name_list[i]
    factor_df = factor_df_list[i][factor_name]
    norm_factor = factor_df.copy()
    # 标准化为排名百分比
    nonzero = norm_factor[norm_factor != 0]
    norm_factor[norm_factor != 0] = nonzero.apply(lambda x: x.rank(pct=True), axis=1)
    norm_factor = np.sign(IC_df.iloc[i, 0]) * norm_factor
    norm_factor_list.append({factor_name: pd.DataFrame(norm_factor, dtype=float)})

# 保证权重为正数，IC_mean为负已有调整
norm_df = sum([math.fabs(weight[i]) * norm_factor_list[i][factor_name_list[i]] for i in range(len(factor_name_list))])
norm_df = norm_df.T
norm_df.index = norm_df.index.astype(object)
norm_df.columns = [month]

# 剔除ST股
current_st = ak.stock_zh_a_st_em()['代码'].values.tolist()
no_st_pool = [stock.split('.')[0] for stock in Ic.fund_pool]
no_st_pool = [code for code in no_st_pool if code not in current_st]
filtered_norm = norm_df[norm_df.index.map(lambda x: x.split('.')[0] in no_st_pool)]
filtered_norm = filtered_norm.sort_values(by=month, ascending=False)

filtered_norm.to_csv(os.path.join(result_path, f'{month}.csv'))

