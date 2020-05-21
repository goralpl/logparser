#!/usr/bin/env python

import sys

sys.path.append('../')
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

sys.path.append('../')
from logparser import Drain, evaluator
import os
import pandas as pd
import csv
import requests
import json
import datetime
from itertools import zip_longest
import logging


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


input_dir = '../logs/'  # The input directory of log file
output_dir = 'thesis_result/'  # The output directory of parsing results


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


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


while True:

    domain = 'www.kpthesisexperiments.com'

    # domain = 'host.docker.internal:8000'

    api_key = 'SKCeSpB1.0Xeu6zlzixpvIQ9kgqxQ7RhmkvVUaeBr'

    # api_key = 'uO5ehOvH.W77UFS3TEgztGQ3V5D70SDtIiFKUEha0'

    # URL to get a random experiment
    get_url = 'https://{domain}/api/v1/experiments/get_random'.format(domain=domain)

    # get_url = 'http://{domain}/api/v1/experiments/get_random'.format(domain=domain)

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Get the experiment as a Dictionary
    ex_tmp = r.json()

    # Assign the values to the local experiment dictionary
    ex = {
        "id": ex_tmp['id'],
        "log_type": ex_tmp['log_type'],
        "log_file": ex_tmp['log_file'],
        "n_gram_size": ex_tmp['n_gram_size'],
        "tokenizing_method": ex_tmp['tokenizing_method'],
        "encoding_method": ex_tmp['encoding_method'],
        "hdb_scan_min_cluster_size": ex_tmp['hdb_scan_min_cluster_size'],
        "hdb_scan_min_sample": ex_tmp['hdb_scan_min_sample'],
        "regex_version": ex_tmp['regex_version'],
        "regex": ex_tmp['regex'].split(ex_tmp['regex_separator']),
        'regex_separator': ex_tmp['regex_separator'],
        "experiment_status": ex_tmp['experiment_status'],
        "start_time": ex_tmp['start_time'],
        "end_time": ex_tmp['end_time'],
        "similarity_score_metric": ex_tmp['similarity_score_metric'],
        "cutoff": ex_tmp['cutoff'],
        "precision": ex_tmp['precision'],
        "recall": ex_tmp['recall'],
        "accuracy": ex_tmp['accuracy'],
        "f_measure": ex_tmp['f_measure']
    }

    # Set the experiment status to 'started'
    put_url = 'https://{domain}/api/v1/experiments/{experiment_id}'.format(domain=domain, experiment_id=ex['id'])
    # put_url = 'http://{domain}/api/v1/experiments/{experiment_id}'.format(domain=domain, experiment_id=ex['id'])

    # Set the experiment status to 'started'
    r = requests.put(url=put_url,
                     data={'experiment_status': 'started'},
                     headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Directory where the raw log files are contained in
    indir = os.path.join(input_dir, os.path.dirname(ex['log_file']))

    # Path to the actual log file
    log_file = os.path.basename(ex['log_file'])

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
    log_file_type = ex['log_file']

    # Function Timer
    ft = True

    # Instantiate KpHdbscan
    k = KpHdbscan(os.path.join('../logs/', log_file_type), function_timer=ft)

    # Set the regex_mode to False if we have no regular expressions defined.
    if ex['regex']:
        regex_mode = True
    else:
        regex_mode = False

    # Load the raw logs
    logs = k.load_logs(regexes=ex['regex'], regex_mode=regex_mode, function_timer=ft)

    # Create a standard list of dictionaries for the log messages. See the function to figure out what "standard" is.
    logs = KpHdbscan.create_standard_log_dictionary(logs, n_gram_size,
                                                    encoding_method=encoding_method,
                                                    function_timer=ft,
                                                    tokenizing_method=tokenizing_method)

    # Return a regular list of tokens, we need this for numpy.
    vec_logs = KpHdbscan.get_list_of_tokens(logs, function_timer=ft)

    # Make a numpy array of the tokens
    numpy_vec_logs = np.array(vec_logs)

    # Instantiate the clustering algorithm
    kp_hdbscan = KpHdbscan.initialize_hdbscan(hdb_scan_min_cluster_size=hdb_scan_min_cluster_size,
                                              hdb_scan_min_samples=hdb_scan_min_samples,
                                              function_timer=ft)

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

    csv_filename = "{}labels_probabilities_vectors.csv".format(log_file_type.replace("/", "_").replace(".", "_"))

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

    # Write the result to the database
    tmp_columns_upload_db = ['cluster',
                             'cluster_probability',
                             'decoded_vector',
                             'cluster_similarity_sum_of_variances',
                             'cluster_similarity_mean_of_variances',
                             'cluster_similarity_mean_of_std']

    # Slice out the columns we want to upload.
    tmp_labels_probabilities_vectors = labels_probabilities_vectors[tmp_columns_upload_db]

    # Add the experiment id
    tmp_labels_probabilities_vectors['experiment'] = ex['id']

    # Get a list of dictionaries
    tmp_labels_probabilities_vectors = tmp_labels_probabilities_vectors.to_dict(orient='records')

    n = 1000

    # using list comprehension
    upload_chunks = [tmp_labels_probabilities_vectors[i * n:(i + 1) * n] for i in
                     range((len(tmp_labels_probabilities_vectors) + n - 1) // n)]

    post_url = 'https://{domain}/api/v1/experimentclusters/new'.format(domain=domain)

    # post_url = 'http://{domain}/api/v1/experimentclusters/new'.format(domain=domain)

    num_experimentclusters = len(upload_chunks)

    for key, upload_chunk in enumerate(upload_chunks):
        print("Request {} of {}".format(key + 1, num_experimentclusters))
        post_data = upload_chunk
        print(post_data)
        r = requests.post(url=post_url,
                          json=post_data,
                          headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})
        print(r.request)
        print(r.text)

    # Here we will extract the relevant clusters by specifying the measure we will use and a cutoff value
    similarity_score_metric = ex['similarity_score_metric']

    cutoff = ex['cutoff']

    # Get the list of relevant tokens
    relevant_tokens = labels_probabilities_vectors.loc[labels_probabilities_vectors[similarity_score_metric] <= cutoff]

    relevant_tokens = labels_probabilities_vectors['decoded_vector'].unique().tolist()

    relevant_tokens.sort()

    csv_lines = []

    cluster_patterns = []

    # Iterate through each log entry
    for line in logs:

        lnn = line['log']

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

        csv_line = {'LineId': line['line'] + 1, "original_log": line["original_log"].strip(),
                    'EventId': cluster_patterns.index(cluster_pattern),
                    'cluster_pattern': str(cluster_pattern).strip()}

        csv_lines.append(csv_line)

    fh = os.path.join(output_dir, "{log_file}_structured.csv".format(log_file=log_file[:-4]))
    f = open(fh, "w")
    writer = csv.DictWriter(f, csv_lines[0].keys())
    writer.writeheader()
    writer.writerows(csv_lines)
    f.close()

    upload_lines = []
    for c in csv_lines:
        tmp = {
            "experiment": ex['id'],
            "line_id": c['LineId'],
            "original_log": c['original_log'],
            "event_id": c['EventId'],
            "cluster_pattern": c['cluster_pattern']
        }
        upload_lines.append(tmp)

    # Upload CSV
    n = 200

    # using list comprehension
    upload_chunks = [upload_lines[i * n:(i + 1) * n] for i in range((len(upload_lines) + n - 1) // n)]

    post_url = 'https://{domain}/api/v1/experimentstructured/new'.format(domain=domain)

    # post_url = 'http://{domain}/api/v1/experimentstructured/new'.format(domain=domain)

    num_experimentclusters = len(upload_chunks)

    for key, upload_chunk in enumerate(upload_chunks):
        print("Request {} of {}".format(key + 1, num_experimentclusters))
        post_data = upload_chunk
        #print(post_data)
        r = requests.post(url=post_url,
                          json=post_data,
                          headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})
        #print(r.request)
        print("HTTP Response Code {}".format(r.status_code))

    # END OF MY CODE

    precision, recall, f_measure, accuracy = evaluator.evaluate(
        groundtruth=os.path.join(indir, log_file[:-4] + '.log_structured.csv'),
        parsedresult=os.path.join(output_dir, log_file[:-4] + '_structured.csv')
    )

    # Set the experiment precision value
    ex['precision'] = precision

    # Set the experiment recall value
    ex['recall'] = recall

    # Set the experiment f_measure value
    ex['f_measure'] = f_measure

    # Set the experiment accuracy value
    ex['accuracy'] = accuracy

    # Set the experiment status value
    ex['experiment_status'] = 'finished'

    ex['end_time'] = datetime.datetime.utcnow()

    # Construct the PUT url
    put_url = 'https://{domain}/api/v1/experiments/{experiment_id}'.format(domain=domain, experiment_id=ex['id'])

    # Exclude some keys for being updated
    put_ex = without_keys(ex, {'id', 'regex'})

    # Update the database
    r = requests.put(url=put_url, data=put_ex, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    print("Experiment Update Response Code {}".format(r.status_code))
