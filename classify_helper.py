from feature_format import featureFormat, targetFeatureSplit
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import test_classifier
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

    # testing classifier before tuning
    print "\n-> Test before tuning:"
    features_train, features_test, labels_train, labels_test = \
        test_classifier(clf, dataset, optimal_features, returns='feat')

    clf = GridSearchCV(clf, parameters)

    # Tune Size of Training
    features_train = features_train[:int(len(features_train)*tune_size)]
    labels_train = labels_train[:int(len(labels_train)*tune_size)]

    print "-> Finding best parameters..."
    t = time()
    clf.fit(features_train, labels_train)
    train_time = min_sec(time(), t)

    print "Best parameters are:"
    pprint(clf.best_params_)
    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])

    print "\n-> Test after tuning..."
    test_classifier(clf.best_estimator_, dataset, optimal_features)
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

    feature_test_dict = {}
    return_initial = False

    for i in range(iterate):

        # break if initial_features condition is not met
        # it should be two features
        if len(initial_features) <> 2:
            print "There should be only two initial features"
            break

        testing_features  = [feature for feature in initial_features]
        accuracy_tracker  = []
        precision_tracker = []
        recall_tracker    = []
        remove_idx = 1

        # STRATIFIED TEST
        # Extracting and Stratifying Data
        print "\n-> Starting {} out of {}, {} folds focusing on {}...".format(i+1, iterate, folds, max_eval_foc)
        t = time()
        for j in range(len(aditional_features)+1):
            print "--> Testing {} out of {} predictor features".format(j+1, len(aditional_features)+1)

            use_idx = j + remove_idx

            # allow evaluation calculations
            # only if there is not too many missing cases
            skip_feature = False

            # extracting data
            data = featureFormat(dataset, testing_features, sort_keys=True)
            data = feature_scaling(clf, data)
            labels, features = targetFeatureSplit(data)

            # debug
            pass
            if j == len(aditional_features)-1:
                pass
            if testing_features[1] == 'total_compensation_abs':
                pass

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
                print "Too much missing data using just {}.".format(testing_features)
                testing_features.remove(testing_features[use_idx])
                remove_idx -= 1
                skip_feature = True

            if not skip_feature:

                try:
                    total_predictions = true_negatives + false_negatives + false_positives + true_positives
                    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
                    precision = 1.0*true_positives/(true_positives+false_positives)
                    recall = 1.0*true_positives/(true_positives+false_negatives)
                    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
                    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

                    print "Evaluation metrics ok!"

                    if j <> 0:
                        max_accuracy = max(accuracy_tracker)
                        max_precision = max(precision_tracker)
                        max_recall = max(recall_tracker)

                    # MAXIMUN ACCURACY AND EVALUATION FOCUS
                    # max_eval_foc
                    if j == 0:
                        pass
                    elif accuracy <= max_accuracy:
                        testing_features.remove(testing_features[use_idx])
                        remove_idx -= 1
                    elif accuracy > max_accuracy:
                        if max_eval_foc.lower() == 'both':
                            if precision < max_precision or recall < max_recall:
                                testing_features.remove(testing_features[use_idx])
                                remove_idx -= 1
                        elif max_eval_foc.lower() == 'prec':
                            if precision < max_precision or recall < .30:
                                testing_features.remove(testing_features[use_idx])
                                remove_idx -= 1
                        elif max_eval_foc.lower() == 'reca':
                            if precision < .30 or recall < max_recall:
                                testing_features.remove(testing_features[use_idx])
                                remove_idx -= 1

                    # tracking progress
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

            # add next feature
            if j <> len(aditional_features):
                testing_features.append(aditional_features[j])
            elif len(testing_features) == 1:
                return_initial = True
                testing_features = initial_features

        print "---> Final test gathering and collecting test {} evaluation metrics:".format(i+1)
        accuracy, precision, recall = test_classifier(clf, dataset, testing_features, returns='eval')
        test = "test_num{}".format(i+1)
        feature_test_dict.update({test: {'accuracy': accuracy,
                                         'precision': precision,
                                         'recall': recall,
                                         'features': testing_features}})
        print "\tFeatures Used in Final Test:\n " \
              "\t{}" \
              "\n\tProcessed in {:0.0f} minute(s) and {:0.0f} second(s)"\
            .format(testing_features, min_sec(time(), t)[0], min_sec(time(), t)[1])

        print "\tAccuracy Tracker {}" \
              "\n\tPrecision Tracker: {}" \
              "\n\tMax Precision: {}" \
              "\n\tRecall Tracker: {}" \
              "\n\tMax Recall: {}".format(accuracy_tracker, precision_tracker, max(precision_tracker),
                                          recall_tracker, max(recall_tracker))

    test_accuracy_list = [feature_test_dict[name]['accuracy'] for name in feature_test_dict.keys()]

    max_test_accuracy = max(test_accuracy_list)


    best_features = [feature_test_dict[name]['features'] for name in feature_test_dict.keys()
                     if feature_test_dict[name]['accuracy'] == max_test_accuracy
                     and feature_test_dict[name]['precision'] >= .30
                     and feature_test_dict[name]['precision'] <> 'NaN'
                     and feature_test_dict[name]['recall'] <> 'NaN'
                     and feature_test_dict[name]['recall'] >= .30]

    if best_features <> []: best_features = best_features[0]

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
    test_classifier(clf, dataset, optimal_features)
    AVERAGE_MESSAGE = "{}\n\tSampling Size: {:0.0f}" \
                      "\tSampling Accuracy Average: {:0.{dec}f}" \
                      "\tSampling Precision Average: {:0.{dec}f}" \
                      "\tSampling Recall Average: {:0.{dec}f}"

    print AVERAGE_MESSAGE.format("*"*60,sampling_size, samp_accuracy_avg,
                                  samp_precision_avg, samp_recall_avg, dec=2)


