from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import test_classifier

# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}

SIZE_AND_TIME_STRING = "\tAccuracy: {:0.2%}\tDatatest Size: {:>0.{dec_points}f}\t\n\
\tTime to fit: {:0.0f} minute(s) and {:0.0f} second(s)\t\
Time to score: {:0.0f} minute(s) and {:0.0f} second(s)\nPredictions:"

def classify_tuner(clf, dataset, optimal_features, parameters={}, tune_size=1):

    # testing classifier before tuning
    print "Test before tuning:"
    features_train, features_test, labels_train, labels_test = \
        test_classifier(clf, dataset, optimal_features, returns='feat')

    clf = GridSearchCV(clf, parameters)

    # Tune Size of Training
    features_train = features_train[:int(len(features_train)*tune_size)]
    labels_train = labels_train[:int(len(labels_train)*tune_size)]

    print "Finding best parameters..."
    t = time()
    clf.fit(features_train, labels_train)
    train_time = min_sec(time(), t)

    print "Best parameters are:"
    pprint(clf.best_params_)
    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])

    print "\nTest after tuning..."
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

def auto_feature(clf, dataset, aditional_features, initial_features, folds=1000, iterate=1):

    feature_test_dict = {}

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
        print "Starting {} out of {}, {} folds...".format(i+1, iterate, folds)
        t = time()
        for j in range(len(aditional_features)):

            # extracting data
            data = featureFormat(dataset, testing_features, sort_keys=True)
            # -> feature scaling here
            labels, features = targetFeatureSplit(data)
            cv = StratifiedShuffleSplit(labels, folds, random_state=42)

            # initiating variables
            true_positives  = 0
            true_negatives  = 0
            false_positives = 0
            false_negatives = 0

            # Creating Stratified Features and Labels

            for train_strat_idxs, test_strat_idxs in cv:
                # resetting lists for next stratified index number
                features_train = []
                features_test  = []
                labels_train   = []
                labels_test    = []

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
                accuracy = 1.0*(true_positives + true_negatives)/total_predictions
                precision = 1.0*true_positives/(true_positives+false_positives)
                recall = 1.0*true_positives/(true_positives+false_negatives)
                f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
                f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

                if j <> 0:
                    max_accuracy = max(accuracy_tracker)
                    max_precision = max(precision_tracker)
                    max_recall = max(recall_tracker)


                if j == 0:
                    pass
                elif accuracy <= max_accuracy:
                    testing_features.remove(testing_features[j+remove_idx])
                    remove_idx -= 1
                elif accuracy > max_accuracy:
                    if precision < max_precision or recall < max_recall:
                        testing_features.remove(testing_features[j+remove_idx])
                        remove_idx -= 1

                testing_features.append(aditional_features[j])

                # tracking progress
                accuracy_tracker.append(accuracy)
                precision_tracker.append(precision)
                recall_tracker.append(recall)



            except:
                print "Got a divide by zero when trying out:", clf
                print "Precision or recall may be undefined due to a lack of true positive predicitons."
                accuracy_tracker.append(accuracy)
                precision_tracker.append(0.)
                recall_tracker.append(0.)
                # automatic removal
                testing_features.remove(testing_features[j+remove_idx])
                remove_idx - 1

        accuracy, precision, recall = test_classifier(clf, dataset, testing_features, returns='eval')
        test = "test_num{}".format(i+1)
        feature_test_dict.update({test: {'accuracy': accuracy,
                                         'precision': precision,
                                         'recall': recall,
                                         'features': testing_features}})
        print "\tFeatures Used:\n " \
              "\t{}" \
              "\n\tProcessed in {:0.0f} minute(s) and {:0.0f} second(s)"\
            .format(testing_features, min_sec(time(), t)[0], min_sec(time(), t)[1])

    test_accuracy_list = [feature_test_dict[name]['accuracy'] for name in feature_test_dict.keys()]

    max_test_accuracy = max(test_accuracy_list)
    #max_test_precision = max([feature_test_dict[name]['precision'] for name in feature_test_dict.keys()])
    #max_test recall = max([feature_test_dict[name]['recall'] for name in feature_test_dict.keys()])

    best_features = [feature_test_dict[name]['features'] for name in feature_test_dict.keys()
                     if feature_test_dict[name]['accuracy'] == max_test_accuracy
                     and feature_test_dict[name]['precision'] >= .30
                     and feature_test_dict[name]['recall'] >= .30]

    if best_features <> []: best_features = best_features[0]

    if best_features <> []:
        # best features
        print "\nOptimal Features Returned"
        print best_features
        return best_features
    else:
        # last features used
        print "\nLast Features Used Returned"
        print testing_features
        return testing_features

def avg_eval_metrics():
    pass








