import pandas as pd
import hdbscan
import numpy as np
import time
import pickle
import struct


class KpHdbscan:
    def __init__(self, indir, outdir, log_path, function_timer=False):
        if function_timer:
            start_time = time.time()

        self.indir = indir
        self.outdir = outdir
        self.log_path = log_path
        self.log_list = None
        self.df = None

        if function_timer:
            print("Instantiate KpHdbscan \t\t\t\t %s seconds ---" % (time.time() - start_time))

    def load_logs(self, function_timer=False):
        """
        This function simply loads the log files form the text files. The result is a list
        of dictionaries with a key that identifies each log line.
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        logs = []
        line_count = 0
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for num, line in enumerate(f):
                logs.append({"line": num, "log": line[16:]})
                line_count = line_count + 1
                # if line_count > 10:
                #     break

        self.log_list = logs

        if function_timer:
            print("KpHdbscan.load_logs() \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return logs

    @staticmethod
    def create_standard_log_dictionary(logs, n_gram_size, encoding_method="default", function_timer=False,
                                       tokenizing_method='sliding_window'):
        """
        Tokenize the log message using sliding window method.
        Each log is a list with the following:
        [
        {'line': 0,
         'log': 'Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0',
         'tokens': [ [token 1], [token 2], etc]
         'relevant_tokens: [[token 1], [token 2], etc ]
         'relevant_tokens_padded : [ [token 1 padded], [token 2 padded], etc]
         'log_as_lst : log message as list
         }
         ]
        :param logs:
        :param n_gram_size:
        :param encoding_method:
        :param function_timer:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        for log in logs:
            log['tokens'] = KpHdbscan.tokenize_logs(n_gram_size, log['log'], encoded=True,
                                                    encoding_method=encoding_method, tokenizing_method=tokenizing_method)
            log['tokens_decoded'] = KpHdbscan.tokenize_logs(n_gram_size, log['log'], encoded=False,tokenizing_method=tokenizing_method)
            log['relevant_tokens'] = []
            log['relevant_tokens_padded'] = []
            log['log_as_lst'] = [None] * len(log['log'])

        if function_timer:
            print("Tokenizing Logs \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return logs

    @staticmethod
    def get_list_of_tokens(logs, function_timer=False):
        """

        When we create our standard log dictionary, it is a list of dictionaries. Sometimes we just want a plain
        list of the log tokens.

        This function will take the standard logs dictionary and return a plain list of log tokens.

        :param logs:
        :param function_timer:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        # Make a list of tokens
        vec_logs = []
        for log in logs:
            for token in log['tokens']:
                vec_logs.append(token)

        if function_timer:
            print("Created vec_logs list \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return vec_logs

    @staticmethod
    def tokenize_logs(n_gram, original_message, encoded=True, encoding_method="default",
                      tokenizing_method="sliding_window"):
        """

        :param n_gram:
        :param original_message:
        :param encoded:
        :param encoding_method:
        :param tokenizing_method:
        :return:
        """

        # Empty list to store the final tokens
        tokens = []

        if tokenizing_method == "sliding_window":

            # Take the original log message and create candidate tokens. We create candidate tokens by applying
            # A sliding window to the original log message of size n_gram.
            candidate_tokens = [original_message[i:i + n_gram] for i in range(len(original_message) - 2)]

            # Iterate over each candidate token to see if it's going ot be added to tokens array.
            for candidate_token in candidate_tokens:

                # If the n-gram matches our windows size then encode and store the result.
                if len(candidate_token) == n_gram:

                    # Store the token as encoded or regular string.
                    if encoded:

                        encoded_token = KpHdbscan.encode_token(candidate_token, encoding_method)

                        tokens.append(encoded_token)
                    else:
                        tokens.append(candidate_token)
        elif tokenizing_method == "fixed_length":

            if encoded:
                tokens = [KpHdbscan.encode_token(original_message[i: i + n_gram].ljust(n_gram)) for i in
                          range(0, len(original_message), n_gram)]
            else:
                tokens = [original_message[i: i + n_gram].ljust(n_gram) for i in
                          range(0, len(original_message), n_gram)]

        elif tokenizing_method == 'word_boundaries':

            # Figure out how to tokenize log entries by word boundaries. We probably have to pad the words
            # to the biggest length word so that we can use the clustering algorithm.

            pass




        return tokens

    @staticmethod
    def encode_token(token, encoding_method='default'):
        """
        This function shall take in a token, and then encode that token depending on the encoding method specified.

        The function returns a list.
        :param token:
        :param encoding_method:
        :return:
        """

        # list to store encoded array
        encoded_string = []

        # encode the string into number representation
        for char in token:

            # Depending on the encoding scheme, use a different encoding method.
            if encoding_method == 'default':
                encoded_string.append(ord(char))

        return encoded_string

    @staticmethod
    def decode_token(token):
        """
        This function shall take in a token, and then decode that token depending on the decoding method specified.

        The function return a string
        :param token:
        :param decoding_method:
        :return:
        """
        # Initialize the decoded string to empty variable.
        decoded_token = ""
        # if decoding_method == 'default':
        if True:
            decoded_token = str(chr(token))

        return decoded_token

    @staticmethod
    def initialize_hdbscan(hdb_scan_min_cluster_size, hdb_scan_min_samples, function_timer=False):
        """

        :param hdb_scan_min_cluster_size:
        :param hdb_scan_min_samples:
        :param function_timer:
        :return:
        """
        if function_timer:
            # Start Timer
            start_time = time.time()

        hdbs = hdbscan.HDBSCAN(min_cluster_size=hdb_scan_min_cluster_size, min_samples=hdb_scan_min_samples)

        if function_timer:
            print("Instantiated hdbscan hdb_scan_min_cluster_size {} hdb_scan_min_samples {} took {} seconds".format(
                hdb_scan_min_cluster_size, hdb_scan_min_samples, time.time() - start_time))

        return hdbs

    @staticmethod
    def cluster_data(hdbscan_object, data, function_timer=False):
        """

        :param hdbscan_object:
        :param data:
        :param function_timer:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        # Cluster the data
        hdbscan_object.fit(data)

        if function_timer:
            print("Clustering completed hdbscan \t\t\t\t %s seconds ---" % (time.time() - start_time))

        # Return the hdbscan object
        return hdbscan_object

    @staticmethod
    def get_cluster_labels_probabilities(hdbscan_object, function_timer=False):
        """

        :param hdbscan_object:
        :param function_timer:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        # Returns an array of integers, each entry corresponds to the data point and tells
        # you which cluster it belongs to
        # labels will return an array [ 1, 2, 4, ... ]
        labels = hdbscan_object.labels_

        # probabilities will return an array [ 1, 2, 4, ... ]
        probabilities = hdbscan_object.probabilities_

        # Zip labels with probabilities so you get [ [label,prob] , [label,prob] ]
        labels_probabilities = [list(a) for a in zip(labels, probabilities)]

        if function_timer:
            print("get_cluster_labels_probabilities \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return labels_probabilities

    @staticmethod
    def get_cluster_labels(hdbscan_object, function_timer=False):
        """

        :param hdbscan_object:
        :param function_timer:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        labels = hdbscan_object.labels_

        if function_timer:
            print("get_cluster_labels \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return labels

    def print(self):
        for line in self.log_list:
            print(line)


class KpPandasDataFrameHelper:
    def __init__(self):
        pass

    @staticmethod
    def create_column_headers(user_defined_columns, n_gram_size, prepend_str='c{}', function_timer=False):
        """

        :param user_defined_columns:
        :param n_gram_size:
        :param prepend_str:
        :return:
        """

        if function_timer:
            # Start Timer
            start_time = time.time()

        # Create the labels for the Pandas Data Frame. Here we create c1, c2, c3, .. cn
        labels_probabilities_vectors_columns = user_defined_columns[:]
        for n_gram in range(1, n_gram_size + 1, 1):
            column_name = prepend_str.format(n_gram)
            labels_probabilities_vectors_columns.append(column_name)

        if function_timer:
            # End Timer
            print(
                "KpPandasDataFrameHelper.create_column_headers() \t\t\t\t %s seconds ---" % (time.time() - start_time))

        return labels_probabilities_vectors_columns
