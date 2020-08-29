#
# Monte Carlo valuation of European call option
# in Black-Scholes-Merton model
# bsm_mcs_euro_demo.py
# 
import math
import numpy as np

# 初始股票指数水平S0=100；欧式看涨期权的行权价格K=105；到期时间T=1年；固定无风险短期利率r=5%；固定波动率σ=20%。
def valuation(s0 = 100, k = 105, t = 1.0, r = 0.05, sigma = 0.2, times = 100000):
    # Valuation Algorithm
    z = np.random.standard_normal(times)  # pseudo-random numbers
    st = s0 * np.exp((r - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)
    ht = np.maximum(st - k, 0)  # payoff at maturity
    c0 = math.exp(-r * t) * np.mean(ht)  # Monte Carlo estimator
    return c0
