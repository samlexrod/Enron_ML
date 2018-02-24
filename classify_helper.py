from feature_format import featureFormat, targetFeatureSplit
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from my_tester import my_test_classifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}

SIZE_AND_TIME_STRING = "\tAccuracy: {:0.2%}\tDatatest Size: {:>0.{dec_points}f}\t\n\
\tTime to fit: {:0.0f} minute(s) and {:0.0f} second(s)\t\
Time to score: {:0.0f} minute(s) and {:0.0f} second(s)\nPredictions:"

def feature_scaling(classifier, data):
    classifier_name = str(classifier)[0:str(classifier).find('(')]
    if classifier_name == 'SVC' or classifier_name == 'KMean':
        print "Applying Feature Scaling..."
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data
    return data

def classify_tuner(clf, dataset, optimal_features, parameters={}, tune_size=1):

    '''
    # testing classifier before tuning
    print "\n-> Test before tuning:"
    features_train, features_test, labels_train, labels_test = \
        my_test_classifier(clf, dataset, optimal_features, returns='feat')
        '''

    data = featureFormat(dataset, optimal_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    clf = GridSearchCV(clf, parameters)

    '''
    # Tune Size of Training
    features_train = features_train[:int(len(features_train)*tune_size)]
    labels_train = labels_train[:int(len(labels_train)*tune_size)]
    '''

    print "-> Finding best parameters..."
    t = time()
    clf.fit(features, labels)
    train_time = min_sec(time(), t)

    print "Best parameters are:"
    pprint(clf.best_params_)
    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])

    print "\n-> Test after tuning..."
    my_test_classifier(clf.best_estimator_, dataset, optimal_features)
    print "-" * 50

    return clf.best_estimator_

# this function converts the time to minutes and seconds and returns tuple
def min_sec(from_time, to_time):
    return floor((from_time - to_time) / 60), (from_time - to_time) % 60

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def auto_feature(clf, dataset, aditional_features, initial_features, folds=1000, iterate=5, max_eval_foc='both'):

    '''
    Conditions of test:
    Accuracy >= .80
    Precision >= .30
    Recall >= .30

    :param clf: This is the ML model selected.
    :param dataset: This is the dataset gathered.
    :param aditional_features: This are the features to be tested on a one-on-one basis and evaluated for
        performance.
    :param initial_features: This are the first features to be tested.
    :param folds: This is the number of randomized indexes used for the StratifiedShuffleSplit.
        The higher the number the more predictions are store and increases the chance of randomization.
    :param iterate: This is the number of times the best feature-finding test will be repeated.
        A dictionary of evaluation metrics and relative best features is created and then compared
        to find the optimal iteration.
    :param max_eval_foc: This is the focus of the evaluation metrics conditions:
        'both' focus on both, precision and recall.
        'reca' focus only on recall while maintaining precision above .30
        'prec' focus on ly on precision while maintaining recall above .30
        All focuses also maximize accuracy.
    :return: It returns the best features that meet the maximizing evaluation metric rule
    '''

    # initiating the dictionary for the iterative tests
    # it will hold the test number, accuracy, precision,
    # recall, and best parameters
    feature_test_dict = {}

    # if the test meet the conditions stated:    #
    return_initial = False

    classifier_name = str(clf)[:str(clf).find("(")]

    # TEST LEVEL
    # starting the number of
    # test stated in the parameters
    for i in range(iterate):

        testing_features  = [feature for feature in initial_features]
        accuracy_tracker  = []
        precision_tracker = []
        recall_tracker    = []
        remove_idx = 1

        # prints the name of the classifier
        # it also prints the progress of the test
        print "\nTesting on {}: " \
              "\n-> Starting {} out of {}, {} folds focusing on {}..."\
            .format(classifier_name, i+1, iterate, folds, max_eval_foc)


        # FEATURE LEVEL
        # starting the test of features
        # the first one is the one with initial features
        # the next ones are one by one comparing the best scores with the next

        t = time()
        for j in range(len(aditional_features)+1):

            # prints the progress of the features and the features being tested
            print "--> Testing {} out of {} predictor features" \
                  "\n\tTesting on {}".format(j+1, len(aditional_features)+1, testing_features)

            # index to remove bad features
            # adjusted to iteration
            # to ensure the last feature
            # gets removed
            use_idx = j + remove_idx

            # allow evaluation calculations
            # only if there is not too many missing cases
            # which is handle by the try error handling
            skip_feature = False

            # extracting data and formatting the data
            data = featureFormat(dataset, testing_features, sort_keys=True)
            data = feature_scaling(clf, data)
            labels, features = targetFeatureSplit(data)

            # on error handling, it fits the data and folds to create more predictions
            # if there is an error in the process, the feature is removed
            try:
                cv = StratifiedShuffleSplit(labels, folds, random_state=42)

                # initiating variables
                true_positives  = 0
                true_negatives  = 0
                false_positives = 0
                false_negatives = 0

                for train_strat_idxs, test_strat_idxs in cv:
                    # resetting lists for next stratified index number
                    features_train = []
                    features_test  = []
                    labels_train   = []
                    labels_test    = []

                    # Creating Stratified Features and Labels
                    # assigning stratified features and labels
                    # related to stratified indexes
                    for train_idx in train_strat_idxs:
                        features_train.append(features[train_idx])
                        labels_train.append(labels[train_idx])
                    for test_idx in test_strat_idxs:
                        features_test.append(features[test_idx])
                        labels_test.append(labels[test_idx])

                    # fitting and predicting on specific indexes for training and labels
                    clf.fit(features_train, labels_train)

                    preds = clf.predict(features_test)

                    for pred, actual in zip(preds, labels_test):
                        if pred == actual and pred == 1:
                            true_positives += 1
                        elif pred == actual and pred == 0:
                            true_negatives += 1
                        elif pred <> actual and pred == 1:
                            false_positives += 1
                        elif pred <> actual and pred == 0:
                            false_negatives += 1
                        else:
                            print "{} is not valid. Prediction should be classified as 0 or 1"

            except:
                print "Error using {}. The feature will be removed from test.".format(testing_features)
                testing_features.remove(testing_features[use_idx])
                remove_idx -= 1

                # prevent the test
                # from continuing
                skip_feature = True

            # if there is no errors on the process above,
            # it calculates the evaluation metrics
            if not skip_feature:

                # parameters for console messages
                perform = 'increased'
                target_feature = testing_features[len(testing_features)-1]

                # on error handling, it calculates the evaluation metrics
                # if there is an error when creating the evaluation metrics,
                # the feature is removed
                try:
                    total_predictions = true_negatives + false_negatives + false_positives + true_positives
                    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
                    precision = 1.0*true_positives/(true_positives+false_positives)
                    recall = 1.0*true_positives/(true_positives+false_negatives)
                    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
                    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

                    # if this is not the first test,
                    # then start tracking the maximum values
                    if j <> 0:
                        max_accuracy = max(accuracy_tracker)
                        max_precision = max(precision_tracker)
                        max_recall = max(recall_tracker)

                    # MAXIMUM ACCURACY AND EVALUATION FOCUS
                    # here is where the magic happens,
                    # features are removed if they do not beat
                    # the max values above depending on the focus of the test
                    if j == 0:
                        pass
                    elif accuracy <= max_accuracy:
                        testing_features.remove(testing_features[use_idx])
                        perform = 'decreased'
                        remove_idx -= 1
                    elif accuracy > max_accuracy:
                        if max_eval_foc.lower() == 'both':
                            if precision < max_precision or recall < max_recall:
                                testing_features.remove(testing_features[use_idx])
                                perform = 'decreased'
                                remove_idx -= 1
                        elif max_eval_foc.lower() == 'prec':
                            if precision < max_precision or recall < .30:
                                testing_features.remove(testing_features[use_idx])
                                perform = 'decreased'
                                remove_idx -= 1
                        elif max_eval_foc.lower() == 'reca':
                            if precision < .30 or recall < max_recall:
                                testing_features.remove(testing_features[use_idx])
                                perform = 'decreased'
                                remove_idx -= 1

                    # parameters for console messages
                    if perform == 'decreased':
                        outcome = 'removed'
                    else:
                        outcome = 'appended'

                    # console message including evaluation metrics at the feature level
                    print "\tEvaluation:" \
                          "\n\tAdding {} {} performance. Feature will be {} for next test." \
                          "\n\tAccuracy: {:0{dec}f}" \
                          "\tPrecision: {:0{dec}f}" \
                          "\tRecall: {:0{dec}f}" \
                          "\tf1: {:0{dec}f}" \
                          "\tf2: {:0{dec}f}" \
                        .format(target_feature, perform, outcome,
                                accuracy, precision, recall, f1, f2, dec=2)

                    # tracking progress to later calculate the maximum values
                    accuracy_tracker.append(round(accuracy, 2))
                    precision_tracker.append(round(precision, 2))
                    recall_tracker.append(round(recall, 2))

                except:
                    print "Got a divide by zero when trying out:", clf
                    print "Precision or recall may be undefined due to a lack of true positive predictions."
                    accuracy_tracker.append(accuracy)
                    precision_tracker.append('NaN')
                    recall_tracker.append('NaN')
                    # automatic removal
                    testing_features.remove(testing_features[use_idx])
                    remove_idx -= 1


            # if the feature is not the last,
            # it counties to append the following features
            # if all predictor features are removed
            # it reinstates initial features sets to return them
            if j <> len(aditional_features):
                testing_features.append(aditional_features[j])
            elif len(testing_features) == 1:
                return_initial = True
                testing_features = initial_features


        print "---> Final test gathering and collecting test {} evaluation metrics:".format(i+1)
        # on the test level, it provides with an example test and gathers evaluation metric tracker
        # and max values

        # this test gathers the new scores at test level with best parameters per test
        accuracy, precision, recall = my_test_classifier(clf, dataset, testing_features, returns='eval')

        # stores test level evaluation metrics
        # and best features in a dictionary
        test = "test_num{}".format(i+1)
        feature_test_dict.update({test: {'accuracy': accuracy,
                                         'precision': precision,
                                         'recall': recall,
                                         'features': testing_features}})

        # tracks the time of the test(s)
        print "\tProcessed in {:0.0f} minute(s) and {:0.0f} second(s)"\
            .format(min_sec(time(), t)[0], min_sec(time(), t)[1])

        # console printing the trackers and max values
        print "\tAccuracy Tracker {}" \
              "\n\tPrecision Tracker: {}" \
              "\n\tMax Precision: {}" \
              "\n\tRecall Tracker: {}" \
              "\n\tMax Recall: {}".format(accuracy_tracker, precision_tracker, max(precision_tracker),
                                          recall_tracker, max(recall_tracker))


    # extract a list of accuracies from the dictionary of tests
    test_accuracy_list = [feature_test_dict[test_num]['accuracy'] for test_num in feature_test_dict.keys()]

    # gets the maximum test accuracy
    max_test_accuracy = max(test_accuracy_list)

    # gets the features with the best maximum accuracy that meets the .30 precision and recall rule
    best_features = [feature_test_dict[name]['features'] for name in feature_test_dict.keys()
                     if feature_test_dict[name]['accuracy'] == max_test_accuracy
                     and feature_test_dict[name]['precision'] >= .30
                     and feature_test_dict[name]['precision'] <> 'NaN'
                     and feature_test_dict[name]['recall'] <> 'NaN'
                     and feature_test_dict[name]['recall'] >= .30]

    # it takes out the list of best features from the list
    if best_features <> []: best_features = best_features[0]

    # if no best features are found,
    # it returns the initial features
    if best_features <> []:
        # best features
        print "\nOptimal Features Returned"
        print best_features
        return best_features
    else:
        # last features used
        if not return_initial:
            print "\nWARNING! At least one evaluation metric did not met the .30 rule" \
                  "\nLast Features Used Returned"
        elif return_initial:
            print "\nWARNING! The model returns no or no significant evaluation metrics." \
                  "\nInitial Features Return"
        print testing_features
        return testing_features

def avg_eval_metrics(clf, dataset, optimal_features, folds=1000, sampling_size=30):

    print "\n :) Starting sampling of {} test(s) to get the sampling averages...".format(sampling_size)

    samp_accuracy_avg_track = []
    samp_precision_avg_track = []
    samp_recall_avg_track = []

    for i in range(sampling_size):
        # extracting data
        data = featureFormat(dataset, optimal_features, sort_keys=True)
        data = feature_scaling(clf, data)
        labels, features = targetFeatureSplit(data)
        cv = StratifiedShuffleSplit(labels, folds, random_state=42)

        # initiating variables
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        # Creating Stratified Features and Labels

        for train_strat_idxs, test_strat_idxs in cv:
            # resetting lists for next stratified index number
            features_train = []
            features_test = []
            labels_train = []
            labels_test = []

            # assigning stratified features and labels
            # related to stratified indexes
            for train_idx in train_strat_idxs:
                features_train.append(features[train_idx])
                labels_train.append(labels[train_idx])
            for test_idx in test_strat_idxs:
                features_test.append(features[test_idx])
                labels_test.append(labels[test_idx])

            # fitting and predicting on specific indexes for training and labels
            clf.fit(features_train, labels_train)
            preds = clf.predict(features_test)

            for pred, actual in zip(preds, labels_test):
                if pred == actual and pred == 1:
                    true_positives += 1
                elif pred == actual and pred == 0:
                    true_negatives += 1
                elif pred <> actual and pred == 1:
                    false_positives += 1
                elif pred <> actual and pred == 0:
                    false_negatives += 1
                else:
                    print "{} is not valid. Prediction should be classified as 0 or 1"

        try:
            total_predictions = true_negatives + false_negatives + false_positives + true_positives
            accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
            precision = 1.0 * true_positives / (true_positives + false_positives)
            recall = 1.0 * true_positives / (true_positives + false_negatives)
            f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
            f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)

        except:
            print "Got a divide by zero when trying out:", clf
            print "Precision or recall may be undefined due to a lack of true positive predictions."
            if not accuracy: accuracy = 0
            if not precision: precision = 0
            if not recall: recall = 0

        samp_accuracy_avg_track.append(accuracy)
        samp_precision_avg_track.append(precision)
        samp_recall_avg_track.append(recall)

    samp_accuracy_avg = np.mean(samp_accuracy_avg_track)
    samp_precision_avg = np.mean(samp_precision_avg_track)
    samp_recall_avg = np.mean(samp_recall_avg_track)

    print "\n-> Final Test..."
    my_test_classifier(clf, dataset, optimal_features)
    AVERAGE_MESSAGE = "{}\n\tSampling Size: {:0.0f}" \
                      "\tSampling Accuracy Average: {:0.{dec}f}" \
                      "\tSampling Precision Average: {:0.{dec}f}" \
                      "\tSampling Recall Average: {:0.{dec}f}"

    print AVERAGE_MESSAGE.format("*"*60,sampling_size, samp_accuracy_avg,
                                  samp_precision_avg, samp_recall_avg, dec=2)


