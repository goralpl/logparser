from logparser.kp_hdbscan.thesis import KpHdbscan
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


benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log'
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log'
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log'
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log'
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log'
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log'
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log'
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log'
    },

    'Linux': {
        'log_file': 'Linux/Linux_full.log'
    },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log'
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log'
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log'
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log'
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log'
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log'

    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log'

    }
}

# Size of our sliding window
n_gram_size = 10

# Reset Timer
start_time = time.time()

# Instantiate KpHdbscan
k = KpHdbscan("test", "test2", os.path.join('../logs/', benchmark_settings['Linux']['log_file']))

print("Instantiate KpHdbscan \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Load the raw logs
logs = k.load_logs()

print("KpHdbscan.load_logs() \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Tokenize the log message using sliding window method.
# Each log is a list with the following:
# [
# {'line': 0,
#  'log': 'Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 ',
#  'tokens': [ [token 1], [token 2], etc]
#  'relevant_tokens: [[token 1], [token 2], etc ]
# Reset Timer
start_time = time.time()
for log in logs:
    log['tokens'] = k.tokenize_logs(n_gram_size, log['log'])
    log['tokens_decoded'] = k.tokenize_logs(n_gram_size, log['log'], encoded=False)
    log['relevant_tokens'] = []
    log['relevant_tokens_padded'] = []
    log['log_as_lst'] = [None] * len(log['log'])

print("Tokenizing Logs \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Make a list of tokens
vec_logs = []
for log in logs:
    for token in log['tokens']:
        vec_logs.append(token)

print("Creating vec_logs list \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Make a numpy array of the tokens
numpy_vec_logs = np.array(vec_logs)

print("Converting vec_logs to numpy_vec_logs \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Instantiate the clustering algorithm
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=15)

print("Instantiated hdbscan \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Cluster the data
clusterer.fit(numpy_vec_logs)

print("Clustering completed hdbscan \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Returns an array of integers, each entry corresponds to the data point and tells
# you which cluster it belongs to
# labels will return an array [ 1, 2, 4, ... ]
labels = clusterer.labels_

# probabilities will return an array [ 1, 2, 4, ... ]
probabilities = clusterer.probabilities_

# Reset Timer
start_time = time.time()

# Zip labels with probabilities so you get [ [label,prob] , [label,prob] ]
labels_probabilities = [list(a) for a in zip(labels, probabilities)]

print("Zipped labels with probabilities \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Each row in the Pandas DataFrame will contain ['label','probabilities', 'c1','c2','c3','cn']
labels_probabilities_vectors = pd.concat([pd.DataFrame(labels_probabilities), pd.DataFrame(vec_logs)], axis=1)

print("Concatenated labels + probabilities + vec_logs \t\t\t\t %s seconds ---" % (time.time() - start_time))

# Reset Timer
start_time = time.time()

# Create the labels for the Pandas Data Frame
labels_probabilities_vectors_columns = ['cluster', 'cluster_probability']
vector_columns = []
for n_gram in range(1, n_gram_size + 1, 1):
    column_name = "c{}".format(n_gram)
    labels_probabilities_vectors_columns.append(column_name)
    vector_columns.append(column_name)

# Assign the column labels to the DataFrame
labels_probabilities_vectors.columns = labels_probabilities_vectors_columns

# Add empty cluster_variance column
labels_probabilities_vectors['cluster_variance'] = -1

# Add empty decoded_vector column
labels_probabilities_vectors['decoded_vector'] = "decoded"

# Get the column indexes for for the vector columns
col_indexes_vector_columns = []
for vector_column in vector_columns:
    col_indexes_vector_columns.append(labels_probabilities_vectors.columns.get_loc(vector_column))

print("Created column labels for Data Frame \t\t\t\t %s seconds ---" % (time.time() - start_time))

zero_variance_cluster_rows = []
for idx, cluster_number in enumerate(range(labels.min(), labels.max() + 1, 1)):

    print("Running {} of {}".format(idx, labels.max() + 1))

    # Reset Timer
    start_time = time.time()

    # tmp_list will contain the label, probability and vectors that belong to cluster_number
    tmp_list = labels_probabilities_vectors[labels_probabilities_vectors['cluster'] == cluster_number]

    print("Created tmp_list \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the Index values and store them
    tmp_indexes = list(tmp_list.index.values)

    print("Created tmp_indexes \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Only keep the vectors
    tmp_list_vectors = tmp_list[labels_probabilities_vectors.columns[-n_gram_size:]]

    print("Created tmp_list_vectors \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the variance for each column
    variance = list(tmp_list_vectors.var(axis=0))

    print("Calculated variance \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the column index for cluster_variance so we can use it later
    col_index_cluster_variance = labels_probabilities_vectors.columns.get_loc('cluster_variance')

    print("Created col_index_cluster_variance \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Assign the sum of column variances to each row that was in the cluster
    labels_probabilities_vectors.iloc[tmp_indexes, col_index_cluster_variance] = sum(variance)

    print("Calculated sum(variance) \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the column index for decoded_vector so we can use it later.
    col_index_decoded_vector = labels_probabilities_vectors.columns.get_loc('decoded_vector')

    print("Calculated col_index_decoded_vector \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    for tmp_index in tmp_indexes:
        labels_probabilities_vectors.iloc[tmp_index, col_index_decoded_vector] = k.decode_token(
            labels_probabilities_vectors.iloc[tmp_index, col_indexes_vector_columns].tolist())

    print("Calculated decoded_token \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # If the variance is 0 for all columns then add the pandas index to a list
    # if sum(variance) < 20:
    if sum(variance) < 20:
        for t in tmp_indexes:
            zero_variance_cluster_rows.append(t)

    print("Calculated zero_variance_cluster_rows \t\t\t\t %s seconds ---" % (time.time() - start_time))

labels_probabilities_vectors_zero_variance = labels_probabilities_vectors.iloc[zero_variance_cluster_rows, :]

labels_probabilities_vectors_zero_variance.to_csv("variance_inspection.csv")

# vecs = labels_probabilities_vectors_zero_variance[['c1', 'c2', 'c3', 'c4', 'c5']].to_numpy()
#
# lab = labels_probabilities_vectors_zero_variance['cluster'].tolist()
# print("hello")
# #hyp.plot(vecs, fmt=".", hue=lab, reduce='TSNE', ndims=2)
# hyp.plot(vecs, fmt=".", hue=lab)

labels_probabilities_vectors_zero_variance.drop_duplicates(keep='first', inplace=True)

with open('filename.pickle', 'wb') as handle:
    pickle.dump(labels_probabilities_vectors_zero_variance, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     labels_probabilities_vectors_zero_variance = pickle.load(handle)_


relevent_tokens = labels_probabilities_vectors_zero_variance['decoded_vector'].tolist()

for relevent_token in relevent_tokens:
    for index, log in enumerate(logs):
        l_len = len(log['log'])
        if relevent_token in log['log']:
            logs[index]['relevant_tokens'].append(relevent_token)
            left_padding = log['log'].find(relevent_token)
            right_padding = l_len - left_padding
            logs[index]['relevant_tokens_padded'].append("*" * left_padding + relevent_token + "*" * right_padding)
            start_index = log['log'].find(relevent_token)
            end_index = start_index + len(relevent_token)
            logs[index]['log_as_lst'][start_index:end_index] = list(relevent_token)


def convert(s):
    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        if x is None:
            x = "*"
        new += x

        # return string
    return new


for log_info in logs:
    print("original: \t {}".format(log_info['log']))
    print("processed: \t {}".format(convert(log_info['log_as_lst'])))
