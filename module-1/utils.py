from scipy.special import comb, factorial
from scipy.stats import fisher_exact
from scipy import stats
import numpy as np


def binomial(n, k):
    """Simple calc of binomial pmf"""
    p = k/n
    return comb(n, k) * p**k * (1 - p)**(n - k)


def poisson(n, k):
    """Simple calc of poisson pmf"""
    p = k/n
    l = p*n
    return (np.e**(-l) * l**(k)) / factorial(k)


def hipergeo_dist(K, k , N, n):
    return (comb(K, k) * comb(N-K, n-k)) / comb(N, n)

n, k = 31000, 63
"""diferences between binomial & poisson with large n is not much"""
print(binomial(n, k) - poisson(n, k))

# K, k = 31000, 39
# N, n = 62000, 102
# hipergeo_dist(K, k, N, n)

# Module 1.4 Fisher's exact test practice
table = np.array([[39, 31000-39], [102-39, 62000-(31000+102)+39]])
oddsr, pval = fisher_exact(table=table, alternative='less')

# Testing the efficacy of a sleeping drug
drug = np.array([6.1, 7.0, 8.2, 7.6, 6.5, 7.8, 6.9, 6.7, 7.4, 5.8])
plac = np.array([5.2, 7.9, 3.9, 4.7, 5.3, 4.8, 4.2, 6.1, 3.8, 6.3])
diff_mean = drug - plac
n = drug.__len__()
sigma_hat = (1/(n-1)) * np.sum((diff_mean - diff_mean.mean())**2)
sigma_hat = np.sqrt((1/(n-1)) * np.sum((diff_mean - diff_mean.mean())**2))
Ts = (diff_mean.mean() - 0)/ (sigma_hat / np.sqrt(n))
print(Ts, diff_mean.mean(), sigma_hat)

Tx, pvalue = stats.ttest_1samp(diff_mean, popmean=0)
print(pvalue/2)