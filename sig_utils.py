import math
import numpy as np
import scipy.stats as stats

def ttest(x0, x1):
    res = stats.ttest_ind(x0, x1)
    return round(res[1], 4)

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(y) - np.mean(x)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    

def significance_eval(known, unknown):
    
    sig = []
    for i, j in zip(known, unknown):
        sig.append(ttest(i, j))
    
    return sig