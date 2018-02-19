from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}

SIZE_AND_TIME_STRING = "\tAccuracy: {:0.2%}\tDatatest Size: {:>0.{dec_points}f}\t\n\
\tTime to fit: {:0.0f} minute(s) and {:0.0f} second(s)\t\
Time to score: {:0.0f} minute(s) and {:0.0f} second(s)\nPredictions:"

def classify(classifier, my_dataset, final_features, fine_tune=False, parameters={}, tune_size=1):

    # Extracting Data
    features_train, features_test, labels_train, labels_test = \
        stratified_data_extract(my_dataset, final_features)

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
    preds = clf.predict(features_test)

    # Getting Training Score
    t = time()
    accuracy = clf.score(features_test, labels_test)
    score_time = min_sec(time(), t)


    print final_features
    print clf
    print SIZE_AND_TIME_STRING.format(accuracy,
                                       int(len(features_test)),
                                       int(len(features_test)*tune_size),
                                       fit_time[0], fit_time[1],
                                       score_time[0], score_time[1], dec_points=0)
    print preds
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

    # Extracting Data
    data = featureFormat(my_dataset, testing_features, sort_keys=True)
    data = feature_scaling(classifier, data)
    labels, features = targetFeatureSplit(data)

    # Spliting Features and Labels
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    return features_train, features_test, labels_train, labels_test


def stratified_data_extract(my_dataset, testing_features, folds=1000):

    # Extracting and Stratifying Data
    data = featureFormat(my_dataset, testing_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state=42, test_size=.1)

    # Creating Stratified Features and Labels
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for i in train_idx:
            features_train.append(features[i])
            labels_train.append(labels[i])
        for j in test_idx:
            features_test.append(features[j])
            labels_test.append(labels[j])

    return features_train, features_test, labels_train, labels_test

# features_for_testing and initial_features should be in the following format: [a, b, c...]

def eval_metrics(preds, labels_test):
    true_positives = sum([1 for pred, actual in zip(preds, labels_test)
                          if pred == actual and pred == 1])
    false_positives = sum([1 for pred, actual in zip(preds, labels_test)
                           if pred <> actual and pred == 1])
    false_negatives = sum([1 for pred, actual in zip(preds, labels_test)
                           if pred <> actual and pred == 0])

    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)

    return precision, recall

def auto_feature(classifier, my_dataset, features_for_testing, initial_features, show=False):

    classifier_name = str(classifier)[0:str(classifier).find('(')]
    print "Finding automatic features for {}".format(classifier_name)
    testing_features = [feature for feature in initial_features]
    removed_features = []
    acc_score_tracker = []
    prec_score_tracker = []
    reca_score_tracker = []

    features_train, features_test, labels_train, labels_test = \
        stratified_data_extract(my_dataset, testing_features)

    # conducting first test of initial features before iteration
    clf = classifier
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    preds = clf.predict(features_train)

    # First Test Stats
    if show:
        print "\nType of classifier:", classifier_name
        print 'First score is {:0.2%}.'.format(score)
        print preds

    # Initial Score Trackers
    acc_score_tracker.append(score)
    try:
        precision, recall = eval_metrics(preds, labels_test)
        prec_score_tracker.append(precision)
        reca_score_tracker.append(recall)
        if show: print "++{} addition leads to valid evaluation metrics".format(feature)
    except:
        if show: print "--{} addition does not lead to valid evaluation metrics".format(feature)
        prec_score_tracker.append(0)
        reca_score_tracker.append(0)

    if show: print "|\nNext feature..."

    # testing prediction on one feature at a time
    # appending only if greater than max
    for feature in features_for_testing:
        # adding one feature at a time
        testing_features.append(feature)

        # fitting the model
        features_train, features_test, labels_train, labels_test = \
            stratified_data_extract(my_dataset, testing_features)
        clf.fit(features_train, labels_train)
        preds = clf.predict(features_test)

        # To debug SVC
        if classifier_name <> "SVC":
            pass

        # EVALUATION METRICS
        try:
            precision, recall = eval_metrics(preds, labels_test)
            prec_score_tracker.append(precision)
            reca_score_tracker.append(recall)
            if show: print "++{} addition leads to valid evaluation metrics".format(feature)
        except:
            if show: print "--{} addition does not lead to valid evaluation metrics".format(feature)
            prec_score_tracker.append(0)
            reca_score_tracker.append(0)

        # SCORE TRACKERS
        acc_score = clf.score(features_test, labels_test)

        # USER STATS WHEN CONDITION MEET OR NOT
        if acc_score <= max(acc_score_tracker):
            if show: print "The score is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                           "\n{1} was removed."\
                .format(acc_score, feature, max(acc_score_tracker))
            testing_features.remove(feature)
            removed_features.append(feature)
        elif acc_score > max(acc_score_tracker):
            if precision < .30 or recall < .30:
                print "\tWARNING!"
                if show:
                    print "The precision is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                               "\n{1} was removed." \
                                .format(precision, feature, max(prec_score_tracker))
                    print "The recall is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                               "\n{1} was removed." \
                                .format(recall, feature, max(reca_score_tracker))
                testing_features.remove(feature)
                removed_features.append(feature)
            else:
                if show: print "**The score is {0:0.2%} when testing with {1}. The maximum is {2:0.2%}. " \
                               "\n{1} was added." \
                    .format(acc_score, feature, max(acc_score_tracker))

        if show: print zip(preds, labels_test)

        # keeping track of the score
        acc_score_tracker.append(acc_score)

        if feature == features_for_testing[len(features_for_testing)-1] and show:
            print ""
        elif show:
            print "|\nNext feature..."

    # final user info
    print "\nType of classifier:", str(classifier)[0:str(classifier).find('(')]
    print "Copy: features_kept = {}".format(testing_features)
    print "Final score: {:0.2%}".format(max(acc_score_tracker))
    print "Pred:", preds
    print "-"*50

    final_features = testing_features
    return final_features





