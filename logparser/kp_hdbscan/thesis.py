import pandas as pd
import hdbscan
import numpy as np
import time
import pickle
import struct


class KpHdbscan():
    def __init__(self, indir, outdir, log_path):
        self.indir = indir
        self.outdir = outdir
        self.log_path = log_path
        self.log_list = None

    def load_logs(self):
        """
        This function simply loads the log files form the text files. The result is a list
        of dictionaries with a key that identifies each log line.
        :return:
        """
        logs = []
        line_count = 0
        with open(self.log_path, 'r',encoding='utf-8') as f:
            for num, line in enumerate(f):
                logs.append({"line": num, "log": line[16:]})
                line_count = line_count + 1
                # if line_count > 10:
                #     break

        self.log_list = logs

        return logs

    def tokenize_logs(self, n_gram, original_message, encoded=True):

        tokens = []

        sliding_windows_messages = [original_message[i:i + n_gram] for i in range(len(original_message) - 2)]

        # Iterate each n-gram.
        for sliding_windows_message in sliding_windows_messages:

            # If the n-gram matches our windows size then encode and store the result.
            if len(sliding_windows_message) == n_gram:

                if encoded:

                    # list to store encoded array
                    encoded_string = []

                    # encode the string into number representation
                    for char in sliding_windows_message:
                        encoded_string.append(ord(char))

                    # sliding_windows_messages_all.append(sliding_windows_message)

                    tokens.append(encoded_string)
                else:
                    tokens.append(sliding_windows_message)

        return tokens

    def decode_token(self, token):

        decoded_string = ""

        for char in token:
            decoded_string += str(chr(char))

        return decoded_string

    def print(self):
        for line in self.log_list:
            print(line)
