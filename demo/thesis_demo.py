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
        'log_file': 'Linux/Linux_2k.log'
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

hdbscan_parameters = [{
    "n_gram_size": 10,
    "encoding_method": "default",
    "hdb_scan_min_cluster_size": 2,
    "hdb_scan_min_samples": 15
}]

# Define the hdbscan minimum cluster size
hdb_scan_min_cluster_size = 2

# Define the hdbscan minimum sample size
hdb_scan_min_samples = 15

# Define the encoding method
encoding_method = "default"

# Size of our sliding window
n_gram_size = 10

# Function Timer
ft = True

# Instantiate KpHdbscan
k = KpHdbscan("test", "test2", os.path.join('../logs/', benchmark_settings['Linux']['log_file']), function_timer=ft)

# Load the raw logs
logs = k.load_logs(function_timer=ft)

# Create a standard list of dictionaries for the log messages. See the function to figure out what "standard" is.
logs = KpHdbscan.create_standard_log_dictionary(logs, n_gram_size, encoding_method=encoding_method, function_timer=ft)

# Return a regular list of tokens, we need this for numpy.
vec_logs = KpHdbscan.get_list_of_tokens(logs, function_timer=ft)

# Make a numpy array of the tokens
numpy_vec_logs = np.array(vec_logs)

# Instantiate the clustering algorithm
kp_hdbscan = KpHdbscan.initialize_hdbscan(hdb_scan_min_cluster_size=hdb_scan_min_cluster_size,
                                          hdb_scan_min_samples=hdb_scan_min_samples, function_timer=ft)

# Cluster the data
kp_hdbscan = KpHdbscan.cluster_data(hdbscan_object=kp_hdbscan, data=numpy_vec_logs, function_timer=ft)

# Get a list of cluster labels and their probabilities
labels_probabilities = KpHdbscan.get_cluster_labels_probabilities(hdbscan_object=kp_hdbscan)

# Get a list of just the cluster labels. This is needed further down the script.
labels = KpHdbscan.get_cluster_labels(hdbscan_object=kp_hdbscan)

# Each row in the Pandas DataFrame will contain ['label','probabilities', 'c1','c2','c3','cn']
labels_probabilities_vectors = pd.concat([pd.DataFrame(labels_probabilities), pd.DataFrame(vec_logs)], axis=1)

user_defined_pd_columns = ['cluster', 'cluster_probability']

print("debug")

# Get columns for new Data Frame.
labels_probabilities_vectors_columns = KpPandasDataFrameHelper.create_column_headers(
    user_defined_columns=user_defined_pd_columns, n_gram_size=n_gram_size, prepend_str='c{}',
    function_timer=ft)

print("debug")

# Assign the column labels to the DataFrame
labels_probabilities_vectors.columns = labels_probabilities_vectors_columns

# Add empty cluster_variance column
labels_probabilities_vectors['cluster_variance'] = -1
user_defined_pd_columns.append('cluster_variance')

# Add empty decoded_vector column
labels_probabilities_vectors['decoded_vector'] = "decoded"
user_defined_pd_columns.append('decoded_vector')

# Get the column indexes for for the vector columns
col_indexes_vector_columns = [col for col in labels_probabilities_vectors.columns if col not in user_defined_pd_columns]

print("debug")

zero_variance_cluster_rows = []
for idx, cluster_number in enumerate(range(labels.min(), labels.max() + 1, 1)):

    print("Running {} of {}".format(idx, labels.max() + 1))

    # Reset Timer
    start_time = time.time()

    # tmp_list will contain the label, probability and vectors that belong to cluster_number
    tmp_list = labels_probabilities_vectors[labels_probabilities_vectors['cluster'] == cluster_number]

    print("debug")

    print("Created tmp_list \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the Index values and store them
    tmp_indexes = list(tmp_list.index.values)

    print("debug")

    print("Created tmp_indexes \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Only keep the vectors
    tmp_list_vectors = tmp_list[labels_probabilities_vectors.columns[-n_gram_size:]]

    print("debug")

    print("Created tmp_list_vectors \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the variance for each column
    variance = list(tmp_list_vectors.var(axis=0))

    print("debug")

    print("Calculated variance \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the column index for cluster_variance so we can use it later
    col_index_cluster_variance = labels_probabilities_vectors.columns.get_loc('cluster_variance')

    print("debug")

    print("Created col_index_cluster_variance \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Assign the sum of column variances to each row that was in the cluster
    labels_probabilities_vectors.iloc[tmp_indexes, col_index_cluster_variance] = sum(variance)

    print("debug")

    print("Calculated sum(variance) \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Get the column index for decoded_vector so we can use it later.
    col_index_decoded_vector = labels_probabilities_vectors.columns.get_loc('decoded_vector')

    print("debug")

    print("Calculated col_index_decoded_vector \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # Decode Token
    for tmp_index in tmp_indexes:
        labels_probabilities_vectors.iloc[tmp_index, col_index_decoded_vector] = k.decode_token(
            labels_probabilities_vectors.iloc[tmp_index, col_indexes_vector_columns].tolist())

    print("debug")

    print("Calculated decoded_token \t\t\t\t %s seconds ---" % (time.time() - start_time))

    # Reset Timer
    start_time = time.time()

    # If the variance is 0 for all columns then add the pandas index to a list
    # if sum(variance) < 20:
    if sum(variance) < 20:
        for t in tmp_indexes:
            zero_variance_cluster_rows.append(t)

    print("debug")

    print("Calculated zero_variance_cluster_rows \t\t\t\t %s seconds ---" % (time.time() - start_time))

labels_probabilities_vectors_zero_variance = labels_probabilities_vectors.iloc[zero_variance_cluster_rows, :]

labels_probabilities_vectors_zero_variance.to_csv("variance_inspection.csv")

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
