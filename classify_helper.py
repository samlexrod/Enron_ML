from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from time import time
from math import floor
from pprint import pprint
from sklearn.model_selection import GridSearchCV

# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}
def classify(classifier, my_dataset, final_features, fine_tune=False, parameters={}, tune_size=1):

    # Extracting Data
    data = featureFormat(my_dataset, final_features, sort_keys=True)
    data = feature_scaling(classifier, data)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=0)

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

    # Statistics
    print "\nSelected classifier:", str(classifier)[0:str(classifier).find('(')]
    print clf
    print "Size of dataset: {} data points".format(int(len(features_train)))
    print "Size of reduced dataset: {} data points".format(int(len(features_train)*tune_size))
    print "Accuracy Score:", "{:0.2%}".format(accuracy)
    print "Time to fit the model: {:0.0f} minute(s) and {:0.0f} second(s)"\
        .format(fit_time[0], fit_time[1])
    print "Time to score the training: {:0.0f} minute(s) and {:0.0f} second(s)"\
        .format(score_time[0], score_time[1])
    print "Predictions:"
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
    # pprint(clf.cv_results_)
    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])
    print "-"*50

    print "Best parameters are:"
    pprint(clf.best_params_)
    # pprint(clf.best_estimator_)
    # pprint(clf.best_score_)
    # pprint(clf.best_index_)
    # pprint(clf.scorer_)
    # pprint(clf.n_splits_)

def feature_scaling(classifier, data):
    classifier_name = str(classifier)[0:str(classifier).find('(')]
    if classifier_name == 'SVC' or classifier_name == 'KMean':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data
    return data


# features_for_testing and initial_features should be in the following format: [a, b, c...]
def auto_feature(classifier, my_dataset, features_for_testing, initial_features, show=False):

    testing_features = [feature for feature in initial_features]
    removed_features = []
    score_tracker = []

    def data_extract():
        data = featureFormat(my_dataset, testing_features, sort_keys=True)
        if show: print '\nFeature Scaling...'
        data = feature_scaling(classifier, data)
        labels, features = targetFeatureSplit(data)

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=0)

        return features_train, features_test, labels_train, labels_test

    features_train, features_test, labels_train, labels_test = data_extract()

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
        features_train, features_test, labels_train, labels_test = data_extract()
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
    print "Features kept:"
    pprint(testing_features)
    print "Final score: {:0.2%}".format(max(score_tracker))
    print "-"*50

    final_features = testing_features
    return final_features





