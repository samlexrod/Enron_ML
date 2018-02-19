from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV

# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
SIZE_AND_TIME_STRING = "\tDataset Size: {:>0.{dec_points}f}\t Reduced Dataset: {:>0.{dec_points}f}\n\t\
Time to fit: {:0.0f} minute(s) and {:0.0f} second(s)\t\
Time to score: {:0.0f} minute(s) and {:0.0f} second(s)\nPredictions:"


def classify(classifier, my_dataset, final_features, fine_tune=False, parameters={}, tune_size=1):

    # Extracting Data
    features_train, features_test, labels_train, labels_test = \
        data_extract(classifier, my_dataset, final_features)

    # Assigning Classifier
    clf = classifier

    # Tune Size of Training
    features_train = features_train[:int(len(features_train)*tune_size)]
    labels_train = labels_train[:int(len(labels_train)*tune_size)]

    # Fitting Training Data
    t = float(time())
    clf = clf.fit(features_train, labels_train)
    fit_time = min_sec(time(), t)

    # Getting Predictions
    pred = clf.predict(features_test)

    # Getting Training Score
    t = time()
    accuracy = clf.score(features_test, labels_test)
    score_time = min_sec(time(), t)

    true_positives = sum([1 for predicted, actual in zip(pred, labels_test)
                          if predicted == actual and predicted == 1])
    true_negatives = sum([1 for predicted, actual in zip(pred, labels_test)
                          if predicted == actual and predicted == 0])
    false_positives = sum([1 for predicted, actual in zip(pred, labels_test)
                           if predicted <> actual and predicted == 1])
    false_negatives = sum([1 for predicted, actual in zip(pred, labels_test)
                           if predicted <> actual and predicted == 0])


    # My version of test_classifier without the Stratified ShuffleSplit
    total_predictions = true_positives + true_negatives + false_positives + false_negatives
    accuracy_manual = 1.0*(true_positives + true_negatives)/total_predictions
    try: precision = 1.0*true_positives/(true_positives+false_positives)
    except: precision = 0
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    try: f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    except: f2 = 0
    # Statistics
    print "\nSelected classifier:", str(classifier)[0:str(classifier).find('(')]
    print "**No StratifiedShuffleSplit applied here <-"
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                       true_negatives)
    print SIZE_AND_TIME_STRING.format(int(len(features_train)),
                                                   int(len(features_train)*tune_size),
                                                   fit_time[0], fit_time[1],
                                                   score_time[0], score_time[1], dec_points=0)
    print pred
    print '-'*50

    # Find Best Parameters
    if fine_tune:
        try:
            grid_search(classifier, [features_train, labels_train], parameters)
        except:
            print "Error: Wrong dictionary of parameters or GridSearchCV " \
                  "does not apply to this classifier"


# this function converts the time to minutes and seconds and returns tuple
def min_sec(from_time, to_time):
    return floor((from_time - to_time) / 60), (from_time - to_time) % 60

# this function finds the best parameters
# it is called within the classifier function when fine_tune is True
def grid_search(classifier, train_data, parameters):

    clf = GridSearchCV(classifier, parameters)

    print "\nFinding best parameters..."
    t = time()
    clf.fit(train_data[0], train_data[1])
    train_time = min_sec(time(), t)

    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])

    print "Best parameters are:"
    pprint(clf.best_params_)

    print "-" * 50

def feature_scaling(classifier, data):
    classifier_name = str(classifier)[0:str(classifier).find('(')]
    if classifier_name == 'SVC' or classifier_name == 'KMean':
        from sklearn.preprocessing import MinMaxScaler
        print "Applying Feature Scaling..."
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data
    return data

# the feature will extract data in labels and features
def data_extract(classifier, my_dataset, testing_features):

    data = featureFormat(my_dataset, testing_features, sort_keys=True)
    data = feature_scaling(classifier, data)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    return features_train, features_test, labels_train, labels_test

# features_for_testing and initial_features should be in the following format: [a, b, c...]
def auto_feature(classifier, my_dataset, features_for_testing, initial_features, show=False):
    print "Finding automatic features..."
    testing_features = [feature for feature in initial_features]
    removed_features = []
    score_tracker = []

    features_train, features_test, labels_train, labels_test = \
        data_extract(classifier, my_dataset, testing_features)

    # conducting first test of initial features before iteration
    clf = classifier
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    pred = clf.predict(features_train)

    # First Test Stats
    if show:
        print "\nType of classifier:", str(classifier)[0:str(classifier).find('(')]
        print 'First score is {:0.2%}.'.format(score)
        print pred

    # Appending scores to compare maximum to new score
    score_tracker.append(score)

    # testing prediction on one feature at a time
    # appending only if greater than max
    for feature in features_for_testing:
        # adding one feature at a time
        testing_features.append(feature)

        # fitting the model
        features_train, features_test, labels_train, labels_test = \
            data_extract(classifier, my_dataset, testing_features)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        score = clf.score(features_test, labels_test)

        # user info
        if score <= max(score_tracker):
            if show: print "The score is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                           "\n{1} was removed."\
                .format(score, feature, max(score_tracker))
            testing_features.remove(feature)
            removed_features.append(feature)
        elif score > max(score_tracker):
            if show: print "**The score is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                           "\n{1} was added." \
                .format(score, feature, max(score_tracker))

        if show: print zip(pred, labels_test)

        # keeping track of the score
        score_tracker.append(score)

    # final user info
    print "\nType of classifier:", str(classifier)[0:str(classifier).find('(')]
    print "Copy: features_kept = {}".format(testing_features)
    print "Final score: {:0.2%}".format(max(score_tracker))
    print "Pred:", pred
    print "-"*50

    final_features = testing_features
    return final_features





