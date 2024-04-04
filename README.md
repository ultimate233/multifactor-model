---
tag: 因子投资
---

## 1. 因子IC测试

假设时间$t=t_1,t_2,t_3,\cdots,t_T$，观测到$t=t_1$时刻的因子值，按排序法建立多空组合，观察其在$t_1\sim t_2$期的收益率，计算秩相关系数。接着观测$t=t_2$的因子值，观察$t_2\sim t_3$的收益率，以此类推。这是间隔期为1个单位的情况。如果间隔期为2个单位，则$t=t_1$的因子值，对应$t_1\sim t_3$的收益率；$t=t_2$的因子值，对应$t_2\sim t_4$的收益率，此时收益率有部分重叠，但没关系，因为在凭因子值买入时我们是没有用到未来收益率的，不存在数据窥探。

这里麻烦的地方在数据对齐。记因子数据框为$factor$，收盘价数据框为$price$

- $factor$：如果间隔1期，最后一行因子值不能用；如果间隔n期，最后n行因子值不能用

  ```python
  factor = factor.iloc[:-n, :]
  ```

- $price$：用收盘价计算收益率。

  - 用`price.iloc[1,:] / price.iloc[0,:]`算出的是$t_1\sim t_2$期的收益率，用`price.iloc[2,:] / price.iloc[0,:]`是$t_1\sim t_3$期的收益率。推广到n期，$t_1\sim t_{1+n}$​的表达式为`price.iloc[0+n,:] / price.iloc[0,:]`

  - 用shift函数，形参为向下平移的格数。需要两个新的dataframe分别实现末期`.iloc[0+n:,]`和初期`.iloc[0,:]`。假定index保留末期的数值，即$t_1\sim t_{1+n}$的收益率最终对应的index为$t_{1+n}$。

    - 对$price$​​向下平移n个单位，舍弃nan，得到初期dataframe。
    - 对$price$取$t_{1+n}$及以后的数据，作为末期dataframe

    ```python
    begin = price.shift(n).dropna()
    end = price.iloc[n:, :]
    ```

  - 末期除以初期再减1，得到区间收益率。做除法时要考虑到分母存在0值，

    ```python
    with np.errstate(divide='ignore', invalid='ignore'):
      # np.where(条件, 成立时的值, 不成立的值)
      period_return = pd.DataFrame(np.where(begin != 0, end / begin - 1, np.nan),
                                  columns=stock_index, index=date_index)
    ```

    因为用到np.where，要先把行列名提取出来并转化成list，存储在$stock\_index$和$date\_index$​两个变量中。

  - 此时得到的$period\_return$​​​符合要求


Note: 用间隔n期的区间收益率只能理论上做IC测试研究，可以比较因子长期的有效性表现。但区间收益率不能用在回测或构件投资组合上，而是只能用$n=1$，即后一期的收益率



IC测试：

|                    | ICmean   | ICstd    | ICIR     | AvgSample |
| ------------------ | -------- | -------- | -------- | --------- |
| avg_turnover(mo)   | -0.08967 | 0.149429 | -0.60009 | 1765.846  |
| net_buy_vol_1M     | 0.054085 | 0.095857 | 0.564223 | 1765.911  |
| volumn(mo)         | -0.0799  | 0.135576 | -0.58933 | 1767.911  |
| high_low_std_3M    | -0.0809  | 0.142698 | -0.56694 | 1774.272  |
| PB                 | -0.04502 | 0.148095 | -0.30402 | 1780.657  |
| corr_ret_turned_1M | -0.04202 | 0.077142 | -0.5447  | 1768.669  |

合成因子IC：

|          | ICmean   | ICstd    | ICIR     | AvgSample   |
| -------- | -------- | -------- | -------- | ----------- |
| 合成因子 | 0.107158 | 0.140357 | 0.763469 | 1780.195266 |

以下为常见因子的IC和分位数组合测试结果

### 1.1 换手率因子

- 平均换手率

![image-20240313022214787](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022214787.png)

- 异常换手率

  ```
  异常换手率定义为：过去21个交易日的平均换手率和过去252个交易日的平均换手率的比值
  ```

- 换手率标准差因子

![abnormal](/Users/zdf/Library/Application Support/typora-user-images/image-20240311105959010.png)

|                         | ICmean   | ICstd    | ICIR     | AvgSample |
| ----------------------- | -------- | -------- | -------- | --------- |
| Avg_turnover (abnormal) | -0.06385 | 0.11821  | -0.5401  | 1765.16   |
| Avg_turnover            | -0.08967 | 0.149429 | -0.60009 | 1765.846  |

*问题* ：平均换手率比异常换手率效果好，与所处市场是否有关（沪深300，中证500，中证1000）



### 1.2 价值因子

![image-20240310203924341](/Users/zdf/Library/Application Support/typora-user-images/image-20240310203924341.png)

|      | ICmean   | ICstd    | ICIR     | AvgSample |
| ---- | -------- | -------- | -------- | --------- |
| PB   | -0.04502 | 0.148095 | -0.30402 | 1780.657  |



### 1.3 波动率因子

振幅的标准差因子 high_low_std_3M ，取过去3个月振幅的标准差（频率为周）

![image-20240313022440850](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022440850.png)

### 1.4 资金流向因子

净买入量（大单）一个月的平均值

![image-20240313022340907](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022340907.png)

### 1.5 成交量因子

![image-20240313022419342](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022419342.png)

### 1.6 量价同步因子

本期换手率和本期收益率的相关系数

![image-20240313022531504](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022531504.png)



## 2. 因子多空组合

先计算股票后一期的收益率。每一期结束时买入本期合成因子

多空组合十三年收益大约为40倍，即4000%

![image-20240313022850908](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022850908.png)

但由于无法做空，仅靠多头的收益仅大约5.5倍，即550%。且股灾时出现巨幅回撤

![image-20240313022925731](/Users/zdf/Library/Application Support/typora-user-images/image-20240313022925731.png)

## 3. 组合调整

程序选出收益最高且最大回撤在25%以内的因子组合

### 3.1 组合1

这是从2010-至今累计收益最高的组合，且最大回撤较低。该策略是每月取因子值最高的12只股票

```
['月平均换手率', 'tapi', 'atr', 'vstd', 'vma', 'srmi', '月成交量', 'cdp', 'bias', 'expma']
```

![image-20240324153208218](/Users/zdf/Library/Application Support/typora-user-images/image-20240324153208218.png)

|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| ------------ | ---------------- | ---------------- | ---------------- |
| avg_return   | 0.001583         | 0.002419         | 0.048382         |
| volatility   | 0.913805         | 1.01247          | 1.864632         |
| sharpe       | 0.434051         | 0.598537         | $6.499972$       |
| max_drawdown | -0.45923         | -0.40558         | $-0.19039$       |
| win_rate     | 0.5              | 0.505952         | 0.654762         |

但是近5年表现并不理想

![image-20240324223140470](/Users/zdf/Library/Application Support/typora-user-images/image-20240324223140470.png)

### 3.2 组合2

这是2019-至今累计收益最高的组合，相比于之前更有时效性。每月取因子值最高的8只股票。

```
['rsi', 'atr', 'vstd', 'srmi', 'cdp', 'bias', 'expma']
```

![image-20240324195241359](/Users/zdf/Library/Application Support/typora-user-images/image-20240324195241359.png)

|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| ------------ | ---------------- | ---------------- | ---------------- |
| avg_return   | 0.001583         | 0.002419         | 0.040124         |
| volatility   | 0.913805         | 1.01247          | 1.918075         |
| sharpe       | 0.434051         | 0.598537         | 5.240331         |
| max_drawdown | -0.45923         | -0.40558         | $-0.2136$        |
| win_rate     | 0.5              | 0.505952         | 0.625            |

近5年收益率曲线

![image-20240324222646461](/Users/zdf/Library/Application Support/typora-user-images/image-20240324222646461.png)

表现还可以，但注意到胜率只有62.5%，因此将胜率也考虑进筛选条件中

### 3.3 组合3

选出收益最高且最大回撤在25%以内，且胜率大于70%的因子组合

```
5yrs Return: 2.95
Combination: ['rsi', 'tapi', 'vstd', 'vma', 'srmi', '月成交量', 'cdp', 'expma']
win_rate: 0.709090909090909
max_drawdown: -0.227437275044878

max_win_rate: 0.7151515151515152
5yrs return: 2.6377866986444527
total return: 947.7118308473921
Comb: ['rsi', '月平均换手率', 'tapi', 'cci', 'vma', 'srmi', '月成交量', 'cdp']
```

对于组合`['rsi', 'tapi', 'vstd', 'vma', 'srmi', '月成交量', 'cdp', 'expma']`，近5年收益2.95倍，14年总收益853.63倍

![image-20240325100146855](/Users/zdf/Library/Application Support/typora-user-images/image-20240325100146855.png)

### 3.4 组合4

```
5yrs Return: 3.67
Combination: ['月平均换手率', 'cci', 'atr', 'vma', 'srmi', 'cdp', 'bias', 'expma']
win_rate: 0.618055555555556

5yrs max_win_rate: 0.6666666666666666
5yrs return: 2.784380230157084
total return: 14016.412313202307
Comb: ['tapi', 'cci', 'vma', 'srmi', 'cdp', 'expma']
```

剔除ST股后，5年内收益最高



![image-20240326092201160](/Users/zdf/Library/Application Support/typora-user-images/image-20240326092201160.png)

|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| ------------ | ---------------- | ---------------- | ---------------- |
| avg_return   | 0.001583         | 0.002419         | 0.05274          |
| volatility   | 0.913805         | 1.01247          | 2.578328         |
| sharpe       | 0.434051         | 0.598537         | 4.39218          |
| max_drawdown | -0.45923         | -0.40558         | -0.24664         |
| win_rate     | 0.5              | 0.505952         | 0.618056         |

17年至今收益

![image-20240326093811204](/Users/zdf/Library/Application Support/typora-user-images/image-20240326093811204.png)

尽管其五年收益确实略高，但全样本内表现实在平庸



### 3.5 组合5

在每一期合成因子时都剔除ST股，即ST股不参与因子值排序，完全不参与计算。

以下策略5年内胜率65%，10年内胜率约70%，且最大回撤为20.8%。

```
['tapi', 'cci', 'vma', 'srmi', 'cdp', 'expma']
```

![image-20240326134930163](/Users/zdf/Library/Application Support/typora-user-images/image-20240326134930163.png)

|                  | **000001.SH_return** | **399300.SZ_return** | **portfolio_return** |
| ---------------- | -------------------- | -------------------- | -------------------- |
| **avg_return**   | 0.001583             | 0.002419             | 0.073308             |
| **volatility**   | 0.913805             | 1.01247              | 3.120535             |
| **sharpe**       | 0.434051             | 0.598537             | 5.674841             |
| **max_drawdown** | -0.45923             | -0.40558             | $-0.20853$           |
| **win_rate**     | 0.5                  | 0.505952             | $0.697531$           |

17年至今

![image-20240326134710591](/Users/zdf/Library/Application Support/typora-user-images/image-20240326134710591.png)

若在2024.2月末以收盘价买入对应股票，截止2024.3.26月收益达到
$$
\frac 15(14.48+23.96+6.56+7.52+2.27)\%=10.958\%
$$

### 3.6 组合6

我觉得也许用五年收益作为衡量指标还是太偏离了，于是该用两年收益，即2022～2024年，正是经济逐步衰退的阶段。

选用因子为：`['tapi', 'atr', 'vma', 'cdp', 'expma']`

![image-20240328002234516](/Users/zdf/Library/Application Support/typora-user-images/image-20240328002234516.png)

```
2yrs Return: 2.50
Combination: ['tapi', 'atr', 'vma', 'cdp', 'expma']

2yrs max_win_rate: 1.0
2yrs return: 0.6036739937832657
total return: 12366.184783704079
Comb: ['rsi', 'tapi', 'vstd', 'vma', 'srmi']
```

虽然胜率最高能达到1.0，但由于时间区间小（24次调仓），过拟合的可能性较大。

|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| ------------ | ---------------- | ---------------- | ---------------- |
| avg_return   | 0.001583         | 0.002419         | 0.04413          |
| volatility   | 0.913805         | 1.01247          | 1.656833         |
| sharpe       | 0.434051         | 0.598537         | 6.036841         |
| max_drawdown | -0.45923         | -0.40558         | $-0.21862$       |
| win_rate     | 0.5              | 0.505952         | $0.644737$       |

策略表现还算不错

![image-20240328002803260](/Users/zdf/Library/Application Support/typora-user-images/image-20240328002803260.png)

![image-20240328002919036](/Users/zdf/Library/Application Support/typora-user-images/image-20240328002919036.png)

由上图也可看出，该组合仅在近两年表现良好，**22年以前不尽人意**。

再输入2024.2的数据，选2024.3应持有的股票，结果为
$$
\frac 15(-2.9+1.74-71.79+13.5+1.01)\%=-11.69\%
$$
其中新海退（002089）截止3.27的月收益率为$-71.79\%$，**逼近退市**。从以上两点推断，该组合过拟合的可能性较大，不具有有效性。

## 4. 因子分析

因子分类如下

| 序号 | 因子名称           | 分类         |
| ---- | ------------------ | ------------ |
| 1    | TAPI加权指数成交量 | 成交量指标   |
| 2    | CCI顺势指标        | 超买超卖指标 |
| 3    | VMA量简单移动平均  | 成交量指标   |
| 4    | SRMI               | 摆动指标     |
| 5    | CDP逆势操作        | 压力支撑指标 |
| 6    | EXPMA指数平均数    | 趋向指标     |

可见以上因子类别较为分散，除了TAPI和VMA同属于成交量指标，其余指标均表征股票的不同技术特点。所有因子IC都大于3%，虽然也考虑过IC更高（6%以上）的其他因子，但由于存在多重共线性，IC较高的因子比较拥挤，组合并不能起到很好的效果。

|           | **ICmean** | **ICstd** | **ICIR** | **AvgSample** |
| --------- | ---------- | --------- | -------- | ------------- |
| **tapi**  | -0.07412   | 0.13397   | -0.55328 | 1893.734      |
| **cci**   | -0.0403    | 0.132958  | -0.30312 | 1845.343      |
| **vma**   | -0.05984   | 0.12915   | -0.46334 | 1860.633      |
| **srmi**  | -0.03899   | 0.137975  | -0.28256 | 1856.367      |
| **cdp**   | -0.03888   | 0.155553  | -0.24994 | 1892.302      |
| **expma** | -0.03037   | 0.155089  | -0.19582 | 1892.497      |
| 合成因子  | 0.08444    | 0.132918  | 0.635276 | 1894.248521   |

根据每个因子IC值进行加权平均后，得到合成因子。其IC均值超过8%，ICIR大于6%，有效性较高。

### 4.1 TAPI加权指数成交量

TAPI指标是根据股票的每日成交值与指数间的关系，来反映股市买气的强弱程度及未来股价展望的技术指标。其理论分析重点为成交值。 TAPI指标认为成交量是股市生命的源泉。成交量值的变化会反映出股市购买股票的强弱程度及对未来股价的愿望，因而可以通过分析每日成交值和加权指数间的关系来研判未来大势变化。
$$
TAPI = \frac{每日成交总值}{当日加权指数}=\frac{A}{PI}
$$
A表示每日的成交金额，PI表示当天的股价指数即指收盘价



### 4.2 CCI顺势指标

顺势指标是由DonaldLambert所创，专门测量股价是否已超出常态分布范围。CCI（Commodity Channel Index，商品通道指数）是一种用于衡量商品价格与其统计平均数之间的偏离程度的技术指标。属于超买超卖类指标中较特殊的一种，波动于正无限大和负无限小之间。但是，又不须要以0为中轴线，这一点也和波动于正无限大和负无限小的指标不同。
$$
TYP:=(HIGH+LOW+CLOSE)/3
$$

$$
CCI:=(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N))
$$

其中$AVEDEV(\cdot)$是平均绝对偏差，公式为
$$
AVEDEV(X) = \frac 1n\sum_{i=1}^n|X_i-\bar{X}|
$$

### 4.3 VMA

成交量的简单算术平均值。
$$
VMA=MA(volume,N) 
$$
其中参数$N=5$

### 4.4 SRMI指标

MI修正指标（SRMI）是表示价格的涨跌速度。如果价格能始终不渝地上升则动力指数继续向上发展，就说明价格上升的速度在加快。
$$
SRMI=\frac{C - C_n}{\max(C, C_n)}
$$
其中$C$指当月收盘价，$C_n$指$n$​月前的收盘价。参数$N=9$​

### 4.5 CDP指标

又称中心点指标（Central Pivot Point）。中心点指标主要用于确定可能的支撑位和阻力位，以及市场趋势的方向。
$$
CDP=(High+Low+Close)/3
$$
CDP是当月最高价、最低价和收盘价的平均值

### 4.6 EXPMA

指数平均数（EMA），其构造原理是对股票收盘价进行算术平均，并根据计算结果来进行分析，用于判断价格未来走势的变动趋势。

下面是指数移动平均数（EMA）的计算公式：

1. 今日EMA（N）的基本计算公式：

$$
\text{今日EMA}(N) = \frac{2}{N+1} \times \text{今日收盘价} + \frac{N-1}{N+1} \times \text{昨日EMA}(N)
$$

2. EMA(X, N) 的通用计算公式：

$$
\text{EMA}(X, N) = \frac{2X + (N - 1) \times \text{EMA}(\text{ref}(X), N)}{N + 1}
$$

其中：

- $\text{今日EMA}(N)$代表当前时间点 \( N \) 天的指数移动平均数。
- $\text{今日收盘价}$是当天的收盘价。
- $\text{昨日EMA}(N)$是前一天的指数移动平均数值。
- $X$ 代表当前的价格数据。
- $\text{ref}(X)$是上一个时间点的价格数据。
- $N$是用来调整计算的时间窗口大小的参数

参数$N=5$​​



## 5. 一些其他技术指标

以下总结若干在挑选组合时考虑到、但未被纳入组合的其他因子

### 5.1 RSI相对强弱指标

全称Relative Strength Index，计算公式为
$$
\frac{上升幅度}{上升幅度+下跌幅度}\times 100
$$
RSI值越大，表示在过去一段时间，上涨几率较大

RSI范围在0～100，一般以70、30为区分，当$RSI\geqslant 70$时，表示价格被高估；当$RSI\leqslant 30$时，表示价格被低估

具体计算方法：

```
LC = REF(CLOSE,1);
RSI = SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
SMA（C,N,M）=M/N×今日收盘价+(N-M)/N×昨日SMA（N）
```

- LC：代表昨日的收盘价，通过 REF(CLOSE,1) 计算得到。
  
- RSI：相对强弱指标的计算公式，它是一个在0到100之间波动的指标，用于衡量股价涨跌的强度。RSI的计算涉及两个参数：N 和 N1，它们是用来调整计算的时间窗口大小的参数。
  
  - SMA：代表简单移动平均线，这是一种常见的技术指标，表示一段时间内价格的平均值。在这个公式中，使用的是加权移动平均线。公式中的 M 是加权系数，N 是时间窗口大小。SMA(C, N, M) 表示计算最新的简单移动平均值。
  
  - `MAX(CLOSE - LC, 0)`：这部分计算得到的是当日收盘价与昨日收盘价之差的正值（如果为负值则置为0），代表当日的涨幅。
  
  - `ABS(CLOSE - LC)`：这部分计算得到的是当日收盘价与昨日收盘价之差的绝对值，代表当日价格变动的幅度。
  
  - RSI 计算公式中的分子部分：`SMA(MAX(CLOSE - LC, 0), N, 1)` 代表了收盘价涨幅的加权移动平均值。
  
  - RSI 计算公式中的分母部分：`SMA(ABS(CLOSE - LC), N1, 1) `代表了收盘价变动幅度的加权移动平均值。
  
  - `RSI=SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N1,1)*100`；最终，将分子除以分母，并乘以100，得到的值就是相对强弱指标（RSI）。

其中，参数$N=6$

### 5.2 ATR真实波动幅度均值

ATR衡量的是在过去一段时间内价格的波动情况

真实波幅（ATR average trade range）主要应用于了解股价的震荡幅度和节奏，在窄幅整理行情中用于寻找突破时机。通常情况下股价的波动幅度会保持在一定常态下，但是如果有主力资金进出时，股价波幅往往会加剧。
$$
TR = \max (\max(High-Low), |Close-High|, |Close-Low|)
$$

$$
ATR = MA(TR, N)
$$

其中TR是当月数据，MA为移动平均线，即对过去N期的TR取平均，这样更加稳健。参数$N=14$。

然而本策略调仓周期为月，因此选用TR，若使用N=14的ATR将考虑过去14个月的走势，引入噪音过大。

### 5.3 VSTD成交量标准差

VSTD指标又名成交量标准差，是指N周期的成交量的估算标准差。
$$
VSTD=STD(V, N)=\sqrt{\frac 1N\sum (V - MA(V, N))^2}
$$
其中V指月成交量（volumn），参数$N=10$​



### 5.4 BIAS乖离率

乖离率（BIAS）是测量股价偏离均线大小程度的指标。当股价偏离市场平均成本太大时，都有一个回归的过程，即所谓的“物极必反”。

计算方法：
$$
BIAS=\frac{当月股市收盘价-N月移动平均值}{N月移动平均值}
$$
参数$N=12$





## 6. 总结

至此策略基本成型。本策略每月调仓一次，持有合成因子值最高的8只股票，直到下一个调仓日。选用以下6个因子：

```
['tapi', 'cci', 'vma', 'srmi', 'cdp', 'expma']
```

近10年收益率曲线如下

![image-20240326134930163](/Users/zdf/Library/Application Support/typora-user-images/image-20240326134930163.png)

近5年的收益率曲线如下

![image-20240328010021147](/Users/zdf/Library/Application Support/typora-user-images/image-20240328010021147.png)

近2年收益率曲线如下

![image-20240328010104448](/Users/zdf/Library/Application Support/typora-user-images/image-20240328010104448.png)

近1年收益率曲线如下

![image-20240328010140055](/Users/zdf/Library/Application Support/typora-user-images/image-20240328010140055.png)

|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| :----------: | ---------------- | ---------------- | ---------------- |
|  avg_return  | 0.001583         | 0.002419         | 0.073308         |
|  volatility  | 0.913805         | 1.01247          | 3.120535         |
|    sharpe    | 0.434051         | 0.598537         | 5.674841         |
| max_drawdown | -0.45923         | -0.40558         | $-0.20853$       |
|   win_rate   | 0.5              | 0.505952         | $0.697531$       |
