import numpy as np


def buy_sell_smart(today, pred, balance, shares, risk=5):
    diff = pred * risk / 100
    if today > pred + diff:
        balance += shares * today
        shares = 0
    elif today > pred:
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        shares += balance / today
        balance = 0
    return balance, shares


def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        shares += balance / today
        balance = 0
    else:
        balance += shares * today
        shares = 0
    return balance, shares


# def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close'):
#     balance_in_time = [balance]
#     shares = 0

#     for ts, target, pred in zip(timstamps, targets, preds):
#         today = data[data[time_key] == int(ts - 24 * 60 * 60)].iloc[0][y_key]
#         assert round(target, 2) == round(data[data[time_key] == int(ts)].iloc[0][y_key], 2)
#         if mode == 'smart':
#             balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
#         elif mode == 'vanilla':
#             balance, shares = buy_sell_vanilla(today, pred, balance, shares)
#         elif mode == 'no_strategy':
#             shares += balance / today
#             balance = 0
#         balance_in_time.append(shares * today + balance)

#     balance += shares * targets[-1]
#     return balance, balance_in_time


def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close', risk_free_rate=0):
    balance_in_time = [balance]
    shares = 0
    max_balance = balance  # 初始余额即为最高余额
    max_drawdown = 0  # 初始最大回撤为0
    daily_returns = []  # 用于计算日收益率

    for ts, target, pred in zip(timstamps, targets, preds):
        today = data[data[time_key] == int(ts - 24 * 60 * 60)].iloc[0][y_key]
        assert round(target, 2) == round(data[data[time_key] == int(ts)].iloc[0][y_key], 2)
        
        # 执行买卖操作
        if mode == 'smart':
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        elif mode == 'vanilla':
            balance, shares = buy_sell_vanilla(today, pred, balance, shares)
        elif mode == 'no_strategy':
            shares += balance / today
            balance = 0

        # 计算当前总资产
        current_balance = shares * today + balance
        balance_in_time.append(current_balance)

        # 更新历史最高余额
        if current_balance > max_balance:
            max_balance = current_balance
        
        # 计算当前回撤
        drawdown = (max_balance - current_balance) / max_balance
        
        # 更新最大回撤
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        # 计算日收益率
        if len(balance_in_time) > 1:
            daily_return = (current_balance - balance_in_time[-2]) / balance_in_time[-2]
            daily_returns.append(daily_return)

    # 计算最终余额
    balance += shares * targets[-1]
    
    # 计算 Sharpe Ratio
    if len(daily_returns) > 1:  # 确保有足够的日收益率数据来计算 Sharpe Ratio
        mean_return = np.mean(daily_returns)  # 日收益率的均值
        std_dev = np.std(daily_returns)  # 日收益率的标准差
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    else:
        sharpe_ratio = 0  # 如果没有足够的数据，返回0
    
    # 返回最终余额、余额时间序列、最大回撤和 Sharpe Ratio
    return balance, balance_in_time, max_drawdown, sharpe_ratio
