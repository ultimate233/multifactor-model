---
tag: 因子投资
---

本策略每月调仓一次，持有合成因子值最高的8只股票，直到下一个调仓日。详见[因子投资手册](https://zzzzdf.page/static/html/%E5%9B%A0%E5%AD%90%E6%8A%95%E8%B5%84.html)。

选用以下6个因子：

```
['tapi', 'cci', 'vma', 'srmi', 'cdp', 'expma']
```

近10年收益率曲线如下

<img width="1154" alt="image-20240326134930163" src="https://github.com/ultimate233/multifactor-model/assets/156023557/942fd2a8-65ad-4a13-98f0-ed5a34e84f1a">

近5年的收益率曲线如下

<img width="1140" alt="image-20240328010021147" src="https://github.com/ultimate233/multifactor-model/assets/156023557/3c39edc2-4edb-4ba6-9531-36610187aeb1">


|              | 000001.SH_return | 399300.SZ_return | portfolio_return |
| :----------: | ---------------- | ---------------- | ---------------- |
|  avg_return  | 0.001583         | 0.002419         | 0.073308         |
|  volatility  | 0.913805         | 1.01247          | 3.120535         |
|    sharpe    | 0.434051         | 0.598537         | 5.674841         |
| max_drawdown | -0.45923         | -0.40558         | $-0.20853$       |
|   win_rate   | 0.5              | 0.505952         | $0.697531$       |
