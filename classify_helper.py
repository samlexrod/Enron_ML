
# this function trains the data and find best parameters when fine_tune is True
# train_data and test_data should be in this format: [features, labels]
# parameters should be a dictionary of parameters: {'kernel': ('linear', 'rgf') 'C': [1, 10]}
def classify(classifier, train_data, test_data, fine_tune=False, parameters = {}):
    from time import time

    # assigning classifier
    clf = classifier

    # fitting training data
    t = float(time())
    clf = clf.fit(train_data[0], train_data[1])
    fit_time = min_sec(time(), t)

    # getting training score
    t = time()
    accuracy = clf.score(test_data[0], test_data[1])
    score_time = min_sec(time(), t)

    # statistics
    print "\nType of classifier:", str(classifier)[0:str(classifier).find('(')]
    print clf
    print "Accuracy Score:", "{:0.2%}".format(accuracy)
    print "Time to fit the model: {:0.0f} minute(s) and {:0.0f} second(s)"\
        .format(fit_time[0], fit_time[1])
    print "Time to score the training: {:0.0f} minute(s) and {:0.0f} second(s)"\
        .format(score_time[0], score_time[1])
    print '-'*50

    # find best parameters
    if fine_tune:
        try:
            grid_search(classifier, train_data, parameters)
        except:
            print "Error: Wrong dictionary of parameters or GridSearchCV " \
                  "does not apply to this classifier"

# this function converts the time to minutes and seconds and returns tuple
def min_sec(from_time, to_time):
    from math import floor
    return floor((from_time - to_time) / 60), (from_time - to_time) % 60

# this function finds the best parameters
# it is called within the classifier function when fine_tune is True
def grid_search(classifier, train_data, parameters):
    from pprint import pprint
    from sklearn.model_selection import GridSearchCV
    from time import time
    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    clf = GridSearchCV(classifier, parameters)

    print "\nFinding best parameters..."
    t = time()
    clf.fit(train_data[0], train_data[1])
    train_time = min_sec(time(), t)
    pprint(clf.cv_results_)
    print "Time to find best parameters: " \
        "{:.0f} minute(s) and {:0.0f} second(s)".format(train_time[0], train_time[1])
    print "-"*50

    pprint(clf.best_params_)
    pprint(clf.best_estimator_)
    pprint(clf.best_score_)
    pprint(clf.best_index_)
    pprint(clf.scorer_)
    pprint(clf.n_splits_)





