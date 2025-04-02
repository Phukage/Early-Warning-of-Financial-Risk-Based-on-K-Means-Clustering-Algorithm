import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import random

index = np.array([random.randint(0,1) for _ in range(10)])
print(index)

int_ls = np.array([i for i in range(10)])
print(int_ls[index == 1])


