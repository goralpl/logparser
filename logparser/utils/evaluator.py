"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import pandas as pd
from collections import defaultdict
import scipy.misc


def evaluate(groundtruth, parsedresult):
    """ Evaluation function to benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """

    df_groundtruth = pd.read_csv(groundtruth)

    df_parsedlog = pd.read_csv(parsedresult)

    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index

    df_groundtruth = df_groundtruth.loc[null_logids]

    df_parsedlog = df_parsedlog.loc[null_logids]

    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])

    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f' % (
        precision, recall, f_measure, accuracy))

    return precision, recall, f_measure, accuracy


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """

    # Group by the ground truth event IDs and count how many per ID
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    # A sum of possible combinations using the ground truth event IDs
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.misc.comb(count, 2)

    # Group by the parsed event IDs and count how many per Id
    series_parsedlog_valuecounts = series_parsedlog.value_counts()

    # A sum of possible combinations using the parsed event IDs
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.misc.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed

    # For each group_by of parsed_eventIds
    for parsed_eventId in series_parsedlog_valuecounts.index:

        # Get the line numbers of all the logs that match the current parsed_eventId
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index

        # Get the event_id + count from the ground truth series based on the logIds from the experiment.
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()

        # Broke out this step into separate temporary variable. debug_tmp holds a list of the ground_truth event ids
        # that were mapped to the experiment ground truth ids. I think if the list has 1 element then the accuracy is
        # 100%
        debug_tmp = series_groundtruth_logId_valuecounts.index.tolist()

        # A tuple that holds the current parsed event_id and the list of ground_truth event ids that were matched
        # according to the line numbers.
        error_eventIds = (parsed_eventId, debug_tmp)

        error = True

        # If the experiment event_id matches exactly to the ground truth.
        if series_groundtruth_logId_valuecounts.size == 1:

            # Just holds the ground truth eventId
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]

            # If the number of log entries with the current event_id matches that of the ground truth.
            # increment the accurate_events with the number of log entries and set error to false.
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        # Ignore this since it's just debug code so it doesn't actually matter for the calculation.
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')

        # This runs regardless if the number of parsed event ids == 1
        for count in series_groundtruth_logId_valuecounts:

            if count > 1:
                # Get the number of 2 combinations for the count of each ground truth EventId
                accurate_pairs += scipy.misc.comb(count, 2)

    # End of for loop
    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs

    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size

    return precision, recall, f_measure, accuracy
