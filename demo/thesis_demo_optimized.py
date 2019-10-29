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

hdbscan_parameters = [
    {
        "experiment": 1,
        "n_gram_size": 2,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 2,
        "n_gram_size": 3,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 3,
        "n_gram_size": 4,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 4,
        "n_gram_size": 5,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 5,
        "n_gram_size": 6,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 6,
        "n_gram_size": 7,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 7,
        "n_gram_size": 8,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 8,
        "n_gram_size": 9,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 9,
        "n_gram_size": 10,
        "encoding_method": "default",
        "tokenizing_method": "sliding_window",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 10,
        "n_gram_size": 2,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 11,
        "n_gram_size": 3,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 12,
        "n_gram_size": 4,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 13,
        "n_gram_size": 5,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 14,
        "n_gram_size": 6,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 15,
        "n_gram_size": 7,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },
    {
        "experiment": 16,
        "n_gram_size": 8,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 17,
        "n_gram_size": 9,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    },

    {
        "experiment": 18,
        "n_gram_size": 10,
        "encoding_method": "default",
        "tokenizing_method": "fixed_length",
        "hdb_scan_min_cluster_size": 2,
        "hdb_scan_min_samples": 15
    }

]

for hdbscan_parameter in hdbscan_parameters:

    print(hdbscan_parameter)

    # Define the hdbscan minimum cluster size
    hdb_scan_min_cluster_size = hdbscan_parameter['hdb_scan_min_cluster_size']

    # Define the hdbscan minimum sample size
    hdb_scan_min_samples = hdbscan_parameter['hdb_scan_min_samples']

    # Define the encoding method
    encoding_method = hdbscan_parameter['encoding_method']

    # Size of our sliding window
    n_gram_size = hdbscan_parameter['n_gram_size']

    tokenizing_method = hdbscan_parameter['tokenizing_method']

    # Function Timer
    ft = True

    # Instantiate KpHdbscan
    k = KpHdbscan("test", "test2", os.path.join('../logs/', benchmark_settings['Linux']['log_file']), function_timer=ft)

    # Load the raw logs
    logs = k.load_logs(function_timer=ft)

    # Create a standard list of dictionaries for the log messages. See the function to figure out what "standard" is.
    logs = KpHdbscan.create_standard_log_dictionary(logs, n_gram_size, encoding_method=encoding_method,
                                                    function_timer=ft, tokenizing_method=tokenizing_method)

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

    # Get columns for new Data Frame.
    labels_probabilities_vectors_columns = KpPandasDataFrameHelper.create_column_headers(
        user_defined_columns=user_defined_pd_columns, n_gram_size=n_gram_size, prepend_str='c{}',
        function_timer=ft)

    # Assign the column labels to the DataFrame
    labels_probabilities_vectors.columns = labels_probabilities_vectors_columns

    # Add empty decoded_vector column
    labels_probabilities_vectors['decoded_vector'] = "decoded"
    user_defined_pd_columns.append('decoded_vector')

    # Get the column indexes for for the vector columns.
    # Vector columns are the ones we will be performing operations on.
    col_indexes_vector_columns_names = [col for col in labels_probabilities_vectors.columns if
                                        col not in user_defined_pd_columns]

    # Get the column index numbers for for the vector columns.
    # Vector columns are the ones we will be performing operations on.
    col_indexes_vector_columns = [labels_probabilities_vectors.columns.get_loc(col) for col in
                                  labels_probabilities_vectors.columns if col not in user_defined_pd_columns]
    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    cluster_similarity_sum_of_variances = labels_probabilities_vectors.groupby('cluster')[
        col_indexes_vector_columns_names].var().sum(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_sum_of_variances'] = labels_probabilities_vectors['cluster'].map(
        cluster_similarity_sum_of_variances.to_dict())

    if ft:
        print("cluster_similarity_sum_of_variances Computed \t\t\t\t %s seconds ---" % (time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    cluster_similarity_mean_of_variances = labels_probabilities_vectors.groupby('cluster')[
        col_indexes_vector_columns_names].var().mean(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_mean_of_variances'] = labels_probabilities_vectors['cluster'].map(
        cluster_similarity_mean_of_variances.to_dict())

    if ft:
        print("cluster_similarity_mean_of_variances Computed \t\t\t\t %s seconds ---" % (time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    cluster_similarity_mean_of_std = labels_probabilities_vectors.groupby('cluster')[
        col_indexes_vector_columns_names].std().mean(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_mean_of_std'] = labels_probabilities_vectors['cluster'].map(
        cluster_similarity_mean_of_std.to_dict())

    if ft:
        print("cluster_similarity_mean_of_std Computed \t\t\t\t %s seconds ---" % (time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    # Get our decoded tokens into a dictionary. We will map back to the Pandas Data Frame using th index.
    decoded_tokens = labels_probabilities_vectors[col_indexes_vector_columns_names].applymap(
        KpHdbscan.decode_token).sum(
        axis=1).to_dict()

    # Create a new column for the decoded tokens
    labels_probabilities_vectors['decoded_vector'] = labels_probabilities_vectors.index.to_series().map(decoded_tokens)

    if ft:
        print(
            "labels_probabilities_vectors['decoded_vector'] Computed \t\t\t\t %s seconds ---" % (
                    time.time() - start_time))

    labels_probabilities_vectors.to_csv("labels_probabilities_vectors_{}.csv".format(hdbscan_parameter['experiment']),
                                        index=False)

    relevant_tokens = labels_probabilities_vectors['decoded_vector'].tolist()

    # for relevent_token in relevent_tokens:
    #     for index, log in enumerate(logs):
    #         l_len = len(log['log'])
    #         if relevent_token in log['log']:
    #             logs[index]['relevant_tokens'].append(relevent_token)
    #             left_padding = log['log'].find(relevent_token)
    #             right_padding = l_len - left_padding
    #             logs[index]['relevant_tokens_padded'].append("*" * left_padding + relevent_token + "*" * right_padding)
    #             start_index = log['log'].find(relevent_token)
    #             end_index = start_index + len(relevent_token)
    #             logs[index]['log_as_lst'][start_index:end_index] = list(relevent_token)
    #
    #
    # def convert(s):
    #     # initialization of string to ""
    #     new = ""
    #
    #     # traverse in the string
    #     for x in s:
    #         if x is None:
    #             x = "*"
    #         new += x
    #
    #         # return string
    #     return new
    #
    #
    # for log_info in logs:
    #     print("original: \t {}".format(log_info['log']))
    #     print("processed: \t {}".format(convert(log_info['log_as_lst'])))
