from logparser.kp_hdbscan.thesis import KpHdbscan
from logparser.kp_hdbscan.thesis import KpPandasDataFrameHelper
import sys
import os
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import hypertools as hyp
import pandas as pd
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import time
import pickle

log = 'Jul 17 06:14:36 combo ftpd[23580]: connection from 83.116.207.11 (aml-sfh-3310b.adsl.wanadoo.nl) at Sun Jul 17 06:14:36 2005'

x1 = KpHdbscan.tokenize_logs(n_gram=5, original_message=log, encoded=True, encoding_method="default",
                             tokenizing_method='sliding_window')
x2 = KpHdbscan.tokenize_logs(n_gram=5, original_message=log, encoded=False, encoding_method="default",
                             tokenizing_method='sliding_window')
x3 = KpHdbscan.tokenize_logs(n_gram=5, original_message=log, encoded=True, encoding_method="default",
                             tokenizing_method='fixed_length')
x4 = KpHdbscan.tokenize_logs(n_gram=5, original_message=log, encoded=False, encoding_method="default",
                             tokenizing_method='fixed_length')
print(x1)


print(x2)

print(x3)

print(x4)
