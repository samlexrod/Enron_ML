# Extracting Data
    features_train, features_test, labels_train, labels_test = \
        stratified_data_extract(my_dataset, final_features)

    # Assigning Classifier
    clf = classifier



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