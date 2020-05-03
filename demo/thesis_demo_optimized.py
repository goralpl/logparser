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
import re
import csv
from logparser import Drain, evaluator


def find_all_indexes(input_str, search_str):
    """
    Found this function on https://www.journaldev.com/23666/python-string-find
    :param input_str:
    :param search_str:
    :return:
    """
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def replace_none(token):
    if token is None:
        return "*"
    else:
        return str(token)


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

benchmark_settings = {'Linux': {
    'log_file': 'Apache/Apache_2k.log'
}

}

input_dir = './logs/'  # The input directory of log file
output_dir = './demo/'  # The output directory of parsing results

indir = os.path.join(input_dir, os.path.dirname(benchmark_settings['Linux']['log_file']))
log_file = os.path.basename(benchmark_settings['Linux']['log_file'])

# n-gram 2 to 20 inclusive
# n_gram_sizes = range(2, 21, 1)
n_gram_sizes = [5]

# Different methods of tokenizing the log messages.
# tokenizing_methods = ['sliding_window', 'fixed_length']
tokenizing_methods = ['fixed_length']
# Encoding Methods
encoding_methods = ['default']

# HDBSCAN min_cluster_size 2 to 10
# hdb_scan_min_cluster_sizes = range(2, 11, 2)
hdb_scan_min_cluster_sizes = [5]

# HDBSCAN min_sample_size 2 to 31
# hdb_scan_min_samples = range(2, 31, 2)
hdb_scan_min_samples = [5]

experiments = []
experiment = 0
for log, log_path in benchmark_settings.items():
    for n_gram_size in n_gram_sizes:
        for tokenizing_method in tokenizing_methods:
            for encoding_method in encoding_methods:
                for hdb_scan_min_cluster_size in hdb_scan_min_cluster_sizes:
                    for hdb_scan_min_sample in hdb_scan_min_samples:
                        experiment = experiment + 1

                        # print("{}:{} {}:{} {}:{} {}:{} {}:{} {}:{}".format(
                        #     "experiment", experiment,
                        #     log, log_path,
                        #     "n_gram_size", n_gram_size,
                        #     "tokenizing_method", tokenizing_method,
                        #     "hdb_scan_min_cluster_size", hdb_scan_min_cluster_size,
                        #     "hdb_scan_min_sample", hdb_scan_min_sample))

                        experiments.append({
                            "experiment": experiment,
                            "log": log,
                            "log_path": log_path['log_file'],
                            "n_gram_size": n_gram_size,
                            "tokenizing_method": tokenizing_method,
                            "encoding_method": encoding_method,
                            "hdb_scan_min_cluster_size": hdb_scan_min_cluster_size,
                            "hdb_scan_min_sample": hdb_scan_min_sample
                        })
experiment = 0

for ex in experiments:
    experiment = experiment + 1
    print("Running experiment {experiment} of {experiments} {percent}%".format(experiment=experiment,
                                                                               experiments=len(experiments),
                                                                               percent=round(
                                                                                   experiment / len(experiments) * 100
                                                                                   , ndigits=2)))
    print(ex)

    # Define the hdbscan minimum cluster size
    hdb_scan_min_cluster_size = ex['hdb_scan_min_cluster_size']

    # Define the hdbscan minimum sample size
    hdb_scan_min_samples = ex['hdb_scan_min_sample']

    # Define the encoding method
    encoding_method = ex['encoding_method']

    # Size of our sliding window
    n_gram_size = ex['n_gram_size']

    # Set the tokenizing method
    tokenizing_method = ex['tokenizing_method']

    # log file path
    log_file_type = ex['log_path']

    # Function Timer
    ft = True

    # Instantiate KpHdbscan
    k = KpHdbscan("test", "test2", os.path.join('../logs/', log_file_type), function_timer=ft)

    # Load the raw logs
    logs = k.load_logs(function_timer=ft)

    print('debug')

    # Create a standard list of dictionaries for the log messages. See the function to figure out what "standard" is.
    logs = KpHdbscan.create_standard_log_dictionary(logs, n_gram_size, encoding_method=encoding_method,
                                                    function_timer=ft, tokenizing_method=tokenizing_method)

    print('debug2')

    # Return a regular list of tokens, we need this for numpy.
    vec_logs = KpHdbscan.get_list_of_tokens(logs, function_timer=ft)

    print('debug2')

    # Make a numpy array of the tokens
    numpy_vec_logs = np.array(vec_logs)

    print('debug2')

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
    labels_probabilities_vectors['cluster_similarity_sum_of_variances'] = labels_probabilities_vectors[
        'cluster'].map(
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
    labels_probabilities_vectors['cluster_similarity_mean_of_variances'] = labels_probabilities_vectors[
        'cluster'].map(
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
    labels_probabilities_vectors['decoded_vector'] = labels_probabilities_vectors.index.to_series().map(
        decoded_tokens)

    if ft:
        print(
            "labels_probabilities_vectors['decoded_vector'] Computed \t\t\t\t %s seconds ---" % (
                    time.time() - start_time))

    csv_filename = "{}labels_probabilities_vectors_{}.csv".format(log_file_type.replace("/", "_").replace(".", "_"),
                                                                  ex['experiment'])

    print("Writing to: {}".format(csv_filename))

    # Create a new column for the experiment number
    labels_probabilities_vectors['experiment'] = ex['experiment']

    # Create a new column for the tokenizing_method
    labels_probabilities_vectors['tokenizing_method'] = ex['tokenizing_method']

    # Create a new column for the encoding_method
    labels_probabilities_vectors['encoding_method'] = ex['encoding_method']

    # Create a new column for the hdb_scan_min_cluster_size
    labels_probabilities_vectors['hdb_scan_min_cluster_size'] = ex['hdb_scan_min_cluster_size']

    # Create a new column for the hdb_scan_min_sample
    labels_probabilities_vectors['hdb_scan_min_sample'] = ex['hdb_scan_min_sample']

    # Write the final Data Frame To CSV
    labels_probabilities_vectors.to_csv(csv_filename, index=False)

    # Possibly write labels_probabilities_vectors to the database
    # Do it here

    # Here we will extract the relevant clusters by specifying the measure we will use and a cutoff value
    similarity_score_metric = 'cluster_similarity_mean_of_std'
    cutoff = 0

    # Get the list of relevant tokens
    relevant_tokens = labels_probabilities_vectors.loc[labels_probabilities_vectors[similarity_score_metric] <= cutoff]
    relevant_tokens = labels_probabilities_vectors['decoded_vector'].unique().tolist()
    relevant_tokens.sort()

    csv_lines = []

    cluster_patterns = []
    # Iterate through each log entry
    for line in logs:
        cluster_number = 1000

        # print(line)

        lnn = line['log']

        # print(str(line['line']) + "\t" + str(line['log']))
        # Create an empty list the size of the log entry
        tmp_lst = [None] * len(lnn)

        # For each relevant_token
        for relevant_token in relevant_tokens:
            # If the relevant_token is in the log entry
            if relevant_token in lnn:
                # Add the relevant_token to the empty list in the same position as the log

                # Get a list of the lowest indexes for the relevant_tokens found in the line
                low_index_substrings = find_all_indexes(lnn, relevant_token)

                for low_index_substring in low_index_substrings:
                    tmp_lst[low_index_substring:low_index_substring + len(relevant_token)] = list(relevant_token)

        cluster_pattern = ''.join([replace_none(token) for token in tmp_lst])
        cluster_pattern = re.sub('\*\*+', '*', cluster_pattern)

        if cluster_pattern not in cluster_patterns:
            cluster_patterns.append(cluster_pattern)

        csv_line = {"line": line["original_log"].strip(), 'cluster_id': cluster_patterns.index(cluster_pattern),
                    'pattern': str(cluster_pattern).strip()}

        csv_lines.append(csv_line)

    print(os.path.dirname(os.path.realpath(__file__)))

    fh = os.path.join("{log_file}_structured.csv".format(log_file=log_file[:-4]))
    print("Writing to:{}".format(fh))
    f = open(fh, "w")

    writer = csv.DictWriter(f, csv_lines[0].keys())

    writer.writeheader()
    writer.writerows(csv_lines)
    f.close()

    f = open(os.path.join(input_dir,"Apache",os.path.dirname("Apache_2k.log_structured.csv")), "w")
    print("Opened groundtruth")
    F1_measure, accuracy = evaluator.evaluate(
        groundtruth=os.path.join(input_dir,"Apache","Apache_2k.log_structured.csv"),
        parsedresult=os.path.join("{log_file}_structured.csv".format(log_file=log_file[:-4]))
    )
