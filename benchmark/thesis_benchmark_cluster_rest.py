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
import json
import gzip

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
from retrying import retry
import shutil
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score


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


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_random_experiment(api_key, domain):
    """
    This function will reach out to the API server and grab a random experiment for the machine to work on.
    :param api_key:
    :param domain:
    :return:
    """

    # URL to get a random experiment
    get_url = '{domain}/api/v1/experimentclusterparameters/get_random'.format(domain=domain)

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code != 200:
        time.sleep(600)
        raise Exception("Unable to obtain an experiment!")
    else:
        resp = r.json()
        print("Obtained an experiment. ID {}".format(resp['id']))
        return resp


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def acknowledge_experiment(api_key, domain, experiment_cluster_parameter):
    """
    This function will reach out to the API server to confirm that the experiment_cluster_parameter has been received.
    :param api_key:
    :param domain:
    :param id:
    :return:
    """

    experiment_cluster_parameter['experiment_cluster_status'] = "started"

    # Exclude some keys for being updated
    put_ex = without_keys(experiment_cluster_parameter, {'id', 'regex'})

    # Set the experiment_cluster_parameter status to 'started'
    put_url_ack_exp = '{domain}/api/v1/experimentclusterparameters/{id}'.format(domain=domain,
                                                                                id=experiment_cluster_parameter['id'])

    # Set the experiment_cluster_parameter status to 'started'
    r = requests.put(url=put_url_ack_exp,
                     data=put_ex,
                     headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        print(r.content)
        print(r.status_code)
        raise Exception("Unable to acknowledge experiment! Response Code {}".format(r.status_code))
    else:
        print("Acknowledged experiment. ID {}".format(experiment_cluster_parameter['id']))
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def update_experiment(api_key, domain, experiment_cluster_parameter):
    """
    This function will reach out to the API server to confirm that the experiment_cluster_parameter has been received.
    :param api_key:
    :param domain:
    :param id:
    :return:
    """

    experiment_cluster_parameter['experiment_cluster_status'] = "started"

    # Exclude some keys for being updated
    put_ex = without_keys(experiment_cluster_parameter, {'id', 'regex'})

    # Set the experiment_cluster_parameter status to 'started'
    put_url_ack_exp = '{domain}/api/v1/experimentclusterparameters/{id}'.format(domain=domain,
                                                                                id=experiment_cluster_parameter['id'])

    # Set the experiment_cluster_parameter status to 'started'
    r = requests.put(url=put_url_ack_exp,
                     data=put_ex,
                     headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        print(r.content)
        print(r.status_code)
        raise Exception("Unable to acknowledge experiment! Response Code {}".format(r.status_code))
    else:
        print("Updated experiment. ID {}".format(experiment_cluster_parameter['id']))
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def upload_experiment_cluster(api_key, domain, upload_chunk):
    """

    :param api_key:
    :param domain:
    :param upload_chunk:
    :return:
    """
    post_url_new_exp_clstr = '{domain}/api/v1/experimentclusters/new'.format(domain=domain)

    print("Attempting to upload experiment cluster")

    r = requests.post(url=post_url_new_exp_clstr,
                      json=upload_chunk,
                      headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        print(r.status_code)
        raise Exception("Unable to upload cluster experiment")
    else:
        print("Uploaded experiment cluster for experiment id {}".format(upload_chunk[0]['experiment']))
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def upload_experiment_cluster_file(api_key, domain, upload_file, data):
    """

    :param api_key:
    :param domain:
    :param upload_chunk:
    :return:
    """
    post_url_new_exp_clstr = '{domain}/api/v1/experimentclusters/upload'.format(domain=domain)

    print("Attempting to upload experiment cluster")

    r = requests.post(url=post_url_new_exp_clstr,
                      files=upload_file,
                      data=data,
                      headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        print(r.status_code)
        raise Exception("Unable to upload cluster experiment")
    else:

        print("Uploaded experiment cluster for experiment id")
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def upload_experiment_structured(api_key, domain, upload_chunk):
    """

    :param api_key:
    :param domain:
    :param upload_chunk:
    :return:
    """

    post_url = '{domain}/api/v1/experimentstructured/new'.format(domain=domain)

    print("Attempting to upload experiment structured")

    r = requests.post(url=post_url,
                      json=upload_chunk,
                      headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        raise Exception("Unable to upload experiment structured")
    else:
        print("Uploaded experiment structured for experiment id {}".format(upload_chunk[0]['experiment']))
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def finish_experiment(api_key, domain, experiment):
    """

    :param api_key:
    :param domain:
    :param experiment:
    :return:
    """

    # Construct the PUT url
    put_url = '{domain}/api/v1/experimentclusterparameters/{experiment_id}'.format(domain=domain,
                                                                                   experiment_id=experiment['id'])

    # Exclude some keys for being updated
    put_ex = without_keys(experiment, {'id', 'regex'})

    print("Attempting to finish experiment")

    r = requests.put(url=put_url, data=put_ex, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        raise Exception("Unable to finish experiment")
    else:
        print("Finished Experiment")
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_experiment_cluster_count(api_key, domain, experiment):
    """

    :param api_key:
    :param domain:
    :param experiment:
    :return:
    """

    # URL to get a random experiment
    get_url = '{domain}/api/v1/experimentclusters/count/?experiment={id}'.format(domain=domain,
                                                                                 id=experiment['id'])

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code != 200:
        raise Exception("Unable to obtain experiment clusters count!")
    else:
        resp = r.json()
        print("Obtained an experiment. ID {exid} cluster count".format(exid=experiment['id']))
        return resp


while True:

    # Set this to False for production
    local_conn = False

    if local_conn:
        domain = 'http://host.docker.internal:8000'

        api_key = 'V8D90F1k.LlUUj8j0KEsCqpuPjWGuVGj4yAKIbAsm'
    else:

        # Domain for the REST API server
        domain = 'https://www.kpthesisexperiments.com'

        # API Key for the server
        api_key = 'ZCA4Wluw.wlBavJG6xD18LzdhYxmPfMvUzL7Apwsr'

    # Get the experiment as a Dictionary
    ex_tmp = get_random_experiment(api_key, domain)

    # Assign the values to the local experiment dictionary
    ex = {
        "id": ex_tmp['id'],
        "log_type": ex_tmp['log_type'],
        "log_file": ex_tmp['log_file'],
        "log_file_lines": ex_tmp['log_file_lines'],
        "n_gram_size": ex_tmp['n_gram_size'],
        "tokenizing_method": ex_tmp['tokenizing_method'],
        "encoding_method": ex_tmp['encoding_method'],
        "hdb_scan_min_cluster_size": ex_tmp['hdb_scan_min_cluster_size'],
        "hdb_scan_min_sample": ex_tmp['hdb_scan_min_sample'],
        "hdb_scan_distance_metric": ex_tmp['hdb_scan_distance_metric'],
        "regex_version": ex_tmp['regex_version'],
        "regex": ex_tmp['regex'].split(ex_tmp['regex_separator']),
        'regex_separator': ex_tmp['regex_separator'],
        "experiment_cluster_status": ex_tmp['experiment_cluster_status'],
        "clustering_duration_seconds": ex_tmp['clustering_duration_seconds'],
        "cluster_count": ex_tmp['cluster_count'],
        "log_chunk_count": ex_tmp['log_chunk_count'],
        "start_time": ex_tmp['start_time'],
        "end_time": ex_tmp['end_time'],
        "iteration_level": ex_tmp['iteration_level']
    }

    # Assign the values to the local experiment dictionary
    # ex = {
    #     "id": 58179,
    #     "log_type": 'Linux',
    #     "log_file": 'Linux/Linux_Full.log',
    #     "log_file_lines": 0,
    #     "n_gram_size": 37,
    #     "tokenizing_method": 'fixed_length',
    #     "encoding_method": 'minimaxir_char_embedding',
    #     "hdb_scan_min_cluster_size": 52,
    #     "hdb_scan_min_sample":12,
    #     "hdb_scan_distance_metric": 'euclidean',
    #     "regex_version": 0,
    #     "regex": 'no_regex',
    #     'regex_separator': '||||||',
    #     "experiment_cluster_status": 'not started',
    #     "clustering_duration_seconds": 0,
    #     "cluster_count": 0,
    #     "log_chunk_count": 0,
    #     "start_time": '2020-08-15 02:49:00.071319',
    #     "end_time": '2020-01-01 00:00:00.000000',
    #     "iteration_level": 1
    # }

    # Once we acknowledge the experiment we can continue.
    acknowledge_experiment(api_key, domain, ex)

    # Directory where the raw log files are contained in
    indir = os.path.join(input_dir, os.path.dirname(ex['log_file']))

    # Path to the actual log file
    log_file = os.path.basename(ex['log_file'])

    # Define the hdbscan minimum cluster size
    hdb_scan_min_cluster_size = ex['hdb_scan_min_cluster_size']

    # Define the hdbscan minimum sample size
    hdb_scan_min_samples = ex['hdb_scan_min_sample']

    # Define the hdbscan distance metric
    hdb_scan_distance_metric = ex['hdb_scan_distance_metric']

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
    if 'no_regex' in ex['regex']:
        regex_mode = False
        print("regex mode is False: {}".format(ex['regex']))
    else:
        regex_mode = True
        print("regex mode is True: {}".format(ex['regex']))

    # Load the raw logs
    logs = k.load_logs(regexes=ex['regex'], regex_mode=regex_mode, function_timer=ft)

    # Get the number of log entries in the file
    ex['log_file_lines'] = len(logs)

    # Create a standard list of dictionaries for the log messages. See the function to figure out what "standard" is.
    logs = KpHdbscan.create_standard_log_dictionary(logs, n_gram_size,
                                                    encoding_method=encoding_method,
                                                    function_timer=ft,
                                                    tokenizing_method=tokenizing_method)

    # Return a regular list of tokens, we need this for numpy.
    vec_logs = KpHdbscan.get_list_of_tokens(logs, function_timer=ft)

    if encoding_method == 'default':
        # If we use the default encoding method then we have one number per column so we can simply
        # convert it to a numpy array for clustering.
        numpy_vec_logs = np.array(vec_logs)
    elif encoding_method == 'minimaxir_char_embedding':

        # Make a numpy array of the tokens
        tmp_numpy_vec_logs = np.array(vec_logs,dtype=object)
        test = tmp_numpy_vec_logs.shape
        # Get the shape of numpy array
        chunks, character, character_embedding = tmp_numpy_vec_logs.shape

        # Combine the character embedding of each chunk into one dimension
        numpy_vec_logs = tmp_numpy_vec_logs.reshape(chunks, character * character_embedding)

    # Instantiate the clustering algorithm
    kp_hdbscan = KpHdbscan.initialize_hdbscan(hdb_scan_min_cluster_size=hdb_scan_min_cluster_size,
                                              hdb_scan_min_samples=hdb_scan_min_samples,
                                              hdb_scan_distance_metric=hdb_scan_distance_metric,
                                              function_timer=ft)

    # Cluster the data
    if ft:
        # Start Timer
        start_time = time.time()

    kp_hdbscan = KpHdbscan.cluster_data(hdbscan_object=kp_hdbscan, data=numpy_vec_logs, function_timer=ft)

    if ft:
        ex['clustering_duration_seconds'] = time.time() - start_time

    # Pickle the clustering result
    # pickle.dump(kp_hdbscan, open("save.p", "wb"))
    # kp_hdbscan = pickle.load(open("save.p", "rb"))

    # Get a list of cluster labels and their probabilities
    labels_probabilities = KpHdbscan.get_cluster_labels_probabilities(hdbscan_object=kp_hdbscan)

    # Get a list of just the cluster labels. This is needed further down the script.
    labels = KpHdbscan.get_cluster_labels(hdbscan_object=kp_hdbscan)

    # Each row in the Pandas DataFrame will contain ['label','probabilities', 'c1','c2','c3','cn']
    if encoding_method == 'default':
        labels_probabilities_vectors = pd.concat([pd.DataFrame(labels_probabilities), pd.DataFrame(vec_logs)], axis=1)
    elif encoding_method == 'minimaxir_char_embedding':
        labels_probabilities_vectors = pd.concat(
            [pd.DataFrame(labels_probabilities), pd.DataFrame(vec_logs), pd.DataFrame(numpy_vec_logs)], axis=1)

    # The silhouette_score gives the average value for all the samples. This gives a perspective into the density and
    # separation of the formed clusters. The best value is 1 and the worst value is -1. Values near 0 indicate
    # overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster,
    # as a different cluster is more similar.
    if ft:
        # Start Timer
        start_time = time.time()

    # cluster_similarity_overall_silhouette_score = silhouette_score(numpy_vec_logs, labels)
    cluster_similarity_overall_silhouette_score = -1

    if ft:
        cluster_similarity_overall_silhouette_score_duration_seconds = time.time() - start_time
        print("cluster_similarity_overall_silhouette_score_duration_seconds: {}".format(
            cluster_similarity_overall_silhouette_score_duration_seconds))

    # Compute the silhouette scores for each sample. The best value is 1 and the worst value is -1.
    # Values near 0 indicate overlapping clusters.
    if ft:
        # Start Timer
        start_time = time.time()

    # cluster_similarity_silhouette_samples = silhouette_samples(numpy_vec_logs, labels)
    cluster_similarity_silhouette_samples = -1

    if ft:
        cluster_similarity_silhouette_samples_duration_seconds = time.time() - start_time
        print("cluster_similarity_silhouette_samples_duration_seconds:{}".format(
            cluster_similarity_silhouette_samples_duration_seconds))

    # Computes the Davies-Bouldin score. The minimum score is zero, with lower values indicating better clustering.
    if ft:
        # Start Timer
        start_time = time.time()

    cluster_similarity_overall_davies_bouldin_score = davies_bouldin_score(numpy_vec_logs, labels)


    if ft:
        cluster_similarity_overall_davies_bouldin_score_duration_seconds = time.time() - start_time
        print("cluster_similarity_overall_davies_bouldin_score_duration_seconds: {}".format(
            cluster_similarity_overall_davies_bouldin_score_duration_seconds))

    # An array of user defined columns
    user_defined_pd_columns = ['cluster', 'cluster_similarity_each_sample_hdbscan_probability']

    # Get columns for new Data Frame.
    labels_probabilities_vectors_columns = KpPandasDataFrameHelper.create_column_headers(
        user_defined_columns=user_defined_pd_columns, n_gram_size=n_gram_size, prepend_str='c{}',
        encoding_method=encoding_method, function_timer=ft)

    # Assign the column labels to the DataFrame
    labels_probabilities_vectors.columns = labels_probabilities_vectors_columns

    # Add empty decoded_vector column
    labels_probabilities_vectors['decoded_vector'] = "decoded"
    user_defined_pd_columns.append('decoded_vector')

    # At this point we can update the database with the number of log chunks and the number of clusters
    ex['log_chunk_count'] = len(labels_probabilities_vectors)
    ex['cluster_count'] = labels_probabilities_vectors['cluster'].nunique()
    update_experiment(api_key, domain, ex)

    # Get the column indexes for for the vector columns.
    # Vector columns are the ones we will be performing operations on.
    if encoding_method == 'default':
        col_indexes_vector_columns_names = [col for col in labels_probabilities_vectors.columns if
                                            col not in user_defined_pd_columns]

    elif encoding_method == 'minimaxir_char_embedding':
        col_indexes_vector_columns_names = [col for col in labels_probabilities_vectors.columns if
                                            col not in user_defined_pd_columns and 'ce_' in col]

    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    if ex['encoding_method'] == 'minimaxir_char_embedding':
        cluster_similarity_each_cluster_sum_of_variances = labels_probabilities_vectors.groupby('cluster')[
            col_indexes_vector_columns_names].var().sum(axis=1)
    elif ex['encoding_method'] == 'default':
        cluster_similarity_each_cluster_sum_of_variances = labels_probabilities_vectors.groupby('cluster')[
            col_indexes_vector_columns_names].var().sum(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_each_cluster_sum_of_variances'] = labels_probabilities_vectors[
        'cluster'].map(
        cluster_similarity_each_cluster_sum_of_variances.to_dict())

    user_defined_pd_columns.append('cluster_similarity_each_cluster_sum_of_variances')

    if ft:
        print("cluster_similarity_each_cluster_sum_of_variances Computed \t\t\t\t %s seconds ---" % (
                time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    cluster_similarity_each_cluster_mean_of_variances = labels_probabilities_vectors.groupby('cluster')[
        col_indexes_vector_columns_names].var().mean(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_each_cluster_mean_of_variances'] = labels_probabilities_vectors[
        'cluster'].map(
        cluster_similarity_each_cluster_mean_of_variances.to_dict())

    user_defined_pd_columns.append('cluster_similarity_each_cluster_mean_of_variances')

    if ft:
        print("cluster_similarity_each_cluster_mean_of_variances Computed \t\t\t\t %s seconds ---" % (
                time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    # We will use group by to perform operations on each cluster
    cluster_similarity_each_cluster_mean_of_std = labels_probabilities_vectors.groupby('cluster')[
        col_indexes_vector_columns_names].std().mean(axis=1)

    # Create a new column for the similarity measure
    labels_probabilities_vectors['cluster_similarity_each_cluster_mean_of_std'] = labels_probabilities_vectors[
        'cluster'].map(
        cluster_similarity_each_cluster_mean_of_std.to_dict())

    user_defined_pd_columns.append('cluster_similarity_each_cluster_mean_of_std')

    if ft:
        print(
            "cluster_similarity_each_cluster_mean_of_std Computed \t\t\t\t %s seconds ---" % (time.time() - start_time))

    if ft:
        # Start Timer
        start_time = time.time()

    if encoding_method == 'default':
        # Get our decoded tokens into a dictionary. We will map back to the Pandas Data Frame using th index.
        decoded_tokens = labels_probabilities_vectors[col_indexes_vector_columns_names].applymap(
            KpHdbscan.decode_token_default).sum(axis=1).to_dict()

        # Create a new column for the decoded tokens
        labels_probabilities_vectors['decoded_vector'] = labels_probabilities_vectors.index.to_series().map(
            decoded_tokens)

    elif encoding_method == 'minimaxir_char_embedding':
        decode_column_names = [col for col in labels_probabilities_vectors.columns if
                               col not in user_defined_pd_columns and 'ce_' not in col]

        # Get our decoded tokens into a dictionary. We will map back to the Pandas Data Frame using th index.
        decoded_tokens = labels_probabilities_vectors[decode_column_names].applymap(
            KpHdbscan.decode_token_minimaxir_char_embedding).sum(axis=1).to_dict()

        # Create a new column for the decoded tokens
        labels_probabilities_vectors['decoded_vector'] = labels_probabilities_vectors.index.to_series().map(
            decoded_tokens)

    if ft:
        print(
            "labels_probabilities_vectors['decoded_vector'] Computed \t\t\t\t %s seconds ---" % (
                    time.time() - start_time))

    # csv_filename = "{}labels_probabilities_vectors.csv".format(log_file_type.replace("/", "_").replace(".", "_"))

    # Create a new column for the tokenizing_method
    labels_probabilities_vectors['tokenizing_method'] = ex['tokenizing_method']

    # Create a new column for the encoding_method
    labels_probabilities_vectors['encoding_method'] = ex['encoding_method']

    # Create a new column for the hdb_scan_min_cluster_size
    labels_probabilities_vectors['hdb_scan_min_cluster_size'] = ex['hdb_scan_min_cluster_size']

    # Create a new column for the hdb_scan_min_sample
    labels_probabilities_vectors['hdb_scan_min_sample'] = ex['hdb_scan_min_sample']

    # Create a new column for the cluster_similarity_each_sample_silhouette_samples
    labels_probabilities_vectors[
        'cluster_similarity_each_sample_silhouette_samples'] = cluster_similarity_silhouette_samples

    # Create a new column for the cluster_similarity_overall_silhouette_score
    labels_probabilities_vectors[
        'cluster_similarity_overall_silhouette_score'] = cluster_similarity_overall_silhouette_score

    # Create a new column for the cluster_similarity_overall_davies_bouldin_score
    labels_probabilities_vectors[
        'cluster_similarity_overall_davies_bouldin_score'] = cluster_similarity_overall_davies_bouldin_score

    # Write the final Data Frame To CSV
    # labels_probabilities_vectors.to_csv(csv_filename, index=False)

    # Write the result to the database
    tmp_columns_upload_db = ['cluster',
                             'cluster_similarity_each_sample_hdbscan_probability',
                             'decoded_vector',
                             'cluster_similarity_each_cluster_sum_of_variances',
                             'cluster_similarity_each_cluster_mean_of_variances',
                             'cluster_similarity_each_cluster_mean_of_std',
                             'cluster_similarity_each_sample_silhouette_samples',
                             'cluster_similarity_overall_davies_bouldin_score',
                             'cluster_similarity_overall_silhouette_score'
                             ]

    # Slice out the columns we want to upload.
    tmp_labels_probabilities_vectors = labels_probabilities_vectors[tmp_columns_upload_db]

    # Copy the Data Frame to avoid warnings
    tmp_labels_probabilities_vectors = tmp_labels_probabilities_vectors.copy()

    # Add the experiment id
    tmp_labels_probabilities_vectors['experiment_cluster_parameter'] = ex['id']

    # Get a list of dictionaries
    tmp_labels_probabilities_vectors = tmp_labels_probabilities_vectors.to_dict(orient='records')

    # Filename of the plain json file
    json_file_name = 'experiment_cluster_{}.json'.format(ex['id'])

    # Filename of the plain compressed json file
    json_file_name_gzip = 'experiment_cluster_{}.json.gz'.format(ex['id'])

    print("Attempting to write to GZIP")
    with gzip.open(json_file_name_gzip, 'wt', encoding="utf-8") as zipfile:
        json.dump(tmp_labels_probabilities_vectors, zipfile)

    print("Calling Upload Function")
    # Upload the Compressed File to the Server
    upload_experiment_cluster_file(api_key,
                                   domain,
                                   upload_file={'filename': open(json_file_name_gzip, 'rb')},
                                   data={'experiment_cluster_id': ex['id']}
                                   )

    os.remove(json_file_name_gzip)

    # Set the experiment status value
    ex['experiment_cluster_status'] = 'finished'

    # Update the database
    finish_experiment(api_key, domain, experiment=ex)
