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

# Size of our sliding window
n_gram_size = 5

# Instanitate KpHdbscan
k = KpHdbscan("test", "test2", os.path.join('../logs/', benchmark_settings['Linux']['log_file']))

# Load the raw logs
logs = k.load_logs()

# Tokenize the log message using sliding window method.
# Each log is a list with the following:
# [
# {'line': 0,
#  'log': 'Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 ',
#  'tokens': [ [token 1], [token 2], etc]
for log in logs:
    log['tokens'] = k.tokenize_logs(n_gram_size, log['log'])

# Make a list of tokens
vec_logs = []
for log in logs:
    for token in log['tokens']:
        vec_logs.append(token)

# Make a numpy array of the tokesn
numpy_vec_logs = np.array(vec_logs)

# Instantiate the clustering algorithm
clusterer = hdbscan.HDBSCAN()

# Cluster the data
clusterer.fit(numpy_vec_logs)

# Returns an array of integers, each entry corresponds to the data point and tells
# you which cluster it belongs to
# labels will return an array [ 1, 2, 4, ... ]
labels = clusterer.labels_

# probabilities will return an array [ 1, 2, 4, ... ]
probabilities = clusterer.probabilities_

# Zip labels with probabilities so you get [ [label,prob] , [label,prob] ]

labels_probabilities = [list(a) for a in zip(labels, probabilities)]

labels_probabilities_vectors = pd.concat([pd.DataFrame(labels_probabilities), pd.DataFrame(vec_logs)], axis=1)

labels_probabilities_vectors.columns = ['cluster', 'cluster_probability', 'c1', 'c2', 'c3', 'c4', 'c5']

#mean_silhouette_coefficients = metrics.silhouette_score(numpy_vec_logs, labels, metric="euclidean")
calinski_harabasz_score = metrics.calinski_harabaz_score(numpy_vec_logs,labels)
print("hello")

# for cluster_number in range(labels.min(), labels.max() + 1, 1):
#     # tmp_list will contain the label, probability and vectors that belong to cluster_number
#     tmp_list = labels_probabilities_vectors[labels_probabilities_vectors['cluster'] == cluster_number]
#
#     # Only keep the vectors
#     tmp_list_vectors = tmp_list[labels_probabilities_vectors.columns[-5:]]
#
#     # Get the average of each column
#     average = tmp_list_vectors.mean(axis=0)
#
#     print(average)
#
#     print(tmp_list_vectors)
#     print(np.power(tmp_list_vectors, 2))

#
# plt.show()

# hyp.plot(numpy_vec_logs, fmt=".", hue=clusterer.labels_, reduce='TSNE', ndims=2)
# hyp.plot(numpy_vec_logs, fmt=".", hue=clusterer.labels_)


# # Zip labels with probabilities so you get [ [label,prob] , [label,prob] ]
#
# x = [list(a) for a in zip(labels, probabilities)]
#
# z = pd.DataFrame(x)
#
# decoded_logs = []
# for vec_log in vec_logs:
#     decoded_logs.append(k.decode_token(vec_log))
#
# y = pd.DataFrame(decoded_logs)
#
# final = pd.concat([z, y], axis=1)
#
# final.columns = ['cluster_membership', 'probability', 'token']
#
#
# final.to_csv(
#     'kristest.csv'
# )
