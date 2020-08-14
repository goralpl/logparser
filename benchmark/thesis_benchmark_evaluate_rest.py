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


# Need to create a separate table that will contain the evaluation variables.
# Which metric do we use to determine relevant clusters? What is the cutoff value
# For Iterative partitioning we need to come up with a way to do each type and keep track of that


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


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


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_random_experiment_evaluation_parameters(api_key, domain):
    """
    This function will reach out to the API server and grab a random experiment for the machine to work on.
    :param api_key:
    :param domain:
    :return:
    """

    # URL to get a random experiment
    get_url = 'https://{domain}/api/v1/experimentevaluationparameters/get_random'.format(domain=domain)

    print(get_url)

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code != 200:
        raise Exception("Unable to obtain an experiment!")
    else:
        resp = r.json()
        print("Obtained Experiment Parameters for experiment_cluster_parameter!. ID {}".format(
            resp['results'][0]['experiment_cluster_parameter']))
        return resp


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_experiments_experimentclusters(api_key, domain, experiment_cluster_parameter):
    """
    This function will reach out to the API server and grab a random experiment for the machine to work on.
    :param api_key:
    :param domain:
    :return:
    """
    experiment_clusters = []

    # URL to get a random experiment
    get_url = 'https://{domain}/api/v1/experimentclusters'.format(domain=domain)

    params = {'experiment_cluster_parameter': experiment_cluster_parameter}

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)}, params=params)

    r = r.json()

    next_page = r['next']

    while next_page:

        print(next_page)

        # Make the API Call
        r = requests.get(next_page, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

        r = r.json()

        next_page = r['next']

        for res in r['results']:
            experiment_clusters.append(res)

    return experiment_clusters


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_experiments_experimentclusters_spaces(domain, experiment_cluster_parameter):
    """
    This function will reach out to the API server and grab a random experiment for the machine to work on.
    :param api_key:
    :param domain:
    :return:
    """
    experiment_clusters = []

    # URL to get a random experiment
    get_url = 'https://{domain}/media/experiment_cluster_{experiment_cluster_parameter}.json.gz'.format(domain=domain,
                                                                                                        experiment_cluster_parameter=experiment_cluster_parameter)

    # Make the API Call
    r = requests.get(get_url)
    print('this ran')
    experiment_clusters = r.json()

    return experiment_clusters


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def get_experiment(api_key, domain, id):
    """
    This function will reach out to the API server and grab a random experiment for the machine to work on.
    :param api_key:
    :param domain:
    :return:
    """

    # URL to get a random experiment
    get_url = 'https://{domain}/api/v1/experimentclusterparameters/'.format(domain=domain)

    params = {'id': id}

    # Make the API Call
    r = requests.get(get_url, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)}, params=params)

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code != 200:
        raise Exception("Unable to obtain an experiment!")
    else:
        resp = r.json()
        return resp


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def finish_experiment_evaluation_parameter(api_key, domain, experiment):
    """

    :param api_key:
    :param domain:
    :param experiment:
    :return:
    """

    # Construct the PUT url
    put_url = 'https://{domain}/api/v1/experimentevaluationparameters/{experiment_id}'.format(domain=domain,
                                                                                              experiment_id=experiment[
                                                                                                  'id'])

    # Exclude some keys for being updated
    put_ex = without_keys(experiment, {'id'})

    print("Attempting to finish experiment")

    r = requests.put(url=put_url, data=put_ex, headers={'Authorization': 'Api-Key {api_key}'.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        raise Exception("Unable to finish experiment")
    else:
        print("Finished Experiment")
        return r.json()


@retry(stop_max_attempt_number=7, wait_random_min=1000, wait_random_max=15000)
def upload_experiment_result_file(api_key, domain, upload_file, data):
    """

    :param api_key:
    :param domain:
    :param upload_chunk:
    :return:
    """
    post_url_new_exp_clstr = 'https://{domain}/api/v1/experimentresults/upload'.format(domain=domain)

    print("Attempting to upload experimentresults")

    r = requests.post(url=post_url_new_exp_clstr,
                      files=upload_file,
                      data=data,
                      headers={'Authorization': 'Api-Key {api_key} '.format(api_key=api_key)})

    # Check to see if the response is a 200 OK. If not then retry or error out
    if r.status_code not in [200, 201]:
        print(r.status_code)
        raise Exception("Unable to upload cluster experiment")
    else:

        print("Uploaded experiment cluster {}".format(data))
        return r.json()


def evaluate_v1(exp_eval_params, experiment_cluster_parameter, experiment_clusters):
    # Create Data Frame of experiment clusters
    labels_probabilities_vectors = pd.DataFrame(experiment_clusters)

    # Directory where the raw log files are contained in
    indir = os.path.join(input_dir, os.path.dirname(experiment_cluster_parameter['log_file']))

    # Get the log file we are working on
    log_file = experiment_cluster_parameter['log_file']

    if experiment_cluster_parameter['regex'] == 'no_regex':
        regex_mode = False
        print("regex mode is False: {}".format(experiment_cluster_parameter['regex']))
    else:
        regex_mode = True
        experiment_cluster_parameter['regex'] = experiment_cluster_parameter['regex'].split(
            experiment_cluster_parameter['regex_separator'])
        print("regex mode is True: {}".format(experiment_cluster_parameter['regex']))

    for ex in exp_eval_params['results']:
        print("Running Experiment Evaluation Parameters: {}".format(ex))

        similarity_score_metric = ex['similarity_metric']

        cutoff = ex['similarity_metric_cutoff_value']

        # Get the list of relevant tokens

        relevant_tokens = labels_probabilities_vectors.loc[labels_probabilities_vectors['cluster'] != -1]

        relevant_tokens = relevant_tokens.loc[relevant_tokens[similarity_score_metric] <= cutoff]

        # Get the list of relevant tokens in a list
        relevent_token_list = relevant_tokens['decoded_vector'].unique().tolist()

        # Sort the list of relevant tokens
        relevent_token_list.sort()

        csv_lines = []

        cluster_patterns = []

        # Maps the log_type field in the experiments_experimentclusterparameter table to the ground truth logs
        ground_truth_logs = {'Linux_ground_truth': 'Linux_2k.log_structured.csv',
                             'Linux_2k': 'Linux_2k.log'
                             }
        # Get the file name for the ground truth log from the above dictionary
        ground_truth_log = ground_truth_logs[experiment_cluster_parameter['log_type'] + '_ground_truth']

        # Load the original 2K logs.
        original_2k_logs = ground_truth_logs[experiment_cluster_parameter['log_type'] + '_2k']

        # Instantiate KpHdbscan
        k = KpHdbscan(os.path.join(indir, original_2k_logs))

        # Load the raw logs
        logs = k.load_logs(regexes=experiment_cluster_parameter['regex'], regex_mode=regex_mode)

        # Iterate through each log entry
        for line in logs:

            lnn = line['log']

            # Create an empty list the size of the log entry
            tmp_lst = [None] * len(lnn)

            # For each relevant_token
            for relevant_token in relevent_token_list:
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
                        'EventId': 'E' + str(cluster_patterns.index(cluster_pattern) + 1),
                        'cluster_pattern': str(cluster_pattern).strip()}

            csv_lines.append(csv_line)

        fn = "{log_file}_structured.csv".format(log_file=log_file[:-4].split("/")[1])

        fh = os.path.join(output_dir, fn)
        f = open(fh, "w")
        writer = csv.DictWriter(f, csv_lines[0].keys())
        writer.writeheader()
        writer.writerows(csv_lines)
        f.close()

        file_name_gzip = "{log_file}_{experiment_cluster_id}_{experiment_evaluation_parameter_id}_structured.csv.gz".format(
            log_file=log_file[:-4].split("/")[1],
            experiment_cluster_id=experiment_cluster_parameter['id'],
            experiment_evaluation_parameter_id=ex['id']
        )

        with open(os.path.join(output_dir, fn), 'rb') as f_in:
            with gzip.open(file_name_gzip, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Write csv lines to the webserver
        upload_experiment_result_file(api_key=api_key,
                                      domain=domain,
                                      upload_file={'filename': open(file_name_gzip, 'rb')},
                                      data={'experiment_cluster_id': experiment_cluster_parameter['id'],
                                            'experiment_evaluation_parameter_id': ex['id']})
        # Delete the GZ file
        os.remove(file_name_gzip)

        precision, recall, f_measure, accuracy = evaluator.evaluate(
            groundtruth=os.path.join(indir, ground_truth_log),
            parsedresult=os.path.join(output_dir, fn)
        )

        print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f' % (
            precision, recall, f_measure, accuracy))

        # # Set the experiment precision value
        ex['precision'] = precision
        #
        # # Set the experiment recall value
        ex['recall'] = recall
        #
        # # Set the experiment f_measure value
        ex['f_measure'] = f_measure
        #
        # # Set the experiment accuracy value
        ex['accuracy'] = accuracy
        #
        # # Set the experiment status value
        ex['experiment_evaluation_status'] = 'finished'

        # Update the database
        finish_experiment_evaluation_parameter(api_key, domain, experiment=ex)


# The input directory of log file. This is where the raw logs are stored
input_dir = '../logs/'

# The output directory of parsing results
output_dir = 'thesis_result'

# Domain for the REST API server
domain = 'www.kpthesisexperiments.com'

# API Key for the server
api_key = 'ZCA4Wluw.wlBavJG6xD18LzdhYxmPfMvUzL7Apwsr'

# Get the experiment evaluation parameters as a Dictionary
exp_eval_params = get_random_experiment_evaluation_parameters(api_key, domain)

# From the experiment evaluation parameters, get the first item and extract the experiment_cluster_parameter_id
experiment_cluster_parameter_id = exp_eval_params['results'][0]['experiment_cluster_parameter']

# Get the experiment_cluster_parameter form the database
experiment_cluster_parameter = get_experiment(api_key, domain, experiment_cluster_parameter_id)['results'][0]

# Get the experiment clusters for that parameter
experiment_clusters = get_experiments_experimentclusters_spaces(
    domain='kpthesisexperiments.nyc3.digitaloceanspaces.com', experiment_cluster_parameter=experiment_cluster_parameter_id)

evaluate_v1(exp_eval_params, experiment_cluster_parameter, experiment_clusters)
