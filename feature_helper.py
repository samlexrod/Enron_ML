
# the function wil calculate the sum of all stated features
# the features to sum should go under the feat_sum_list with this format: [a, b, c, ...]
# the name of the new feature should go in the new_feature_str as a string
# having absol set to true will convert all features to positive numbers and then add to avoid subtraction
def feat_sum(data, new_feature_str, feat_sum_list, absol = True):
    # sum all values from selected features
    # add a new key to the dictionary and returns it
    # to create a new feature

    for name in data.keys():
        if absol:
            total = sum([abs(data[name][feature]) for feature in feat_sum_list if data[name][feature] <> 'NaN'])
            new_feature_str = new_feature_str + "_abs"
        else:
            total = sum([data[name][feature] for feature in feat_sum_list if data[name][feature] <> 'NaN'])

        data[name][new_feature_str] = total

        # validating aggregation
        # print "Compare to: ", [(feature, data[name][feature]) for feature in feat_sum_list]
        # print "Analyze negative: ", [(feature, data[name][feature]) for feature in feat_sum_list \
                                     # if data[name][feature] < 0]
        # print "Returning:", data[name][new_feature_str]

    return data

# the function will calculate the ratio of given combination of values, nominator and denominator
# the feature_list should be entered in this format [[a, b], [c, d]]
# the name of the new feature should be entered in this format: [a, b]
def feat_ratio(data, features_list, feature_names):
    for name in data.keys():
        # validating ratio
        # print [(nominator, denominator) for (nominator, denominator) in features_list]
        # print name, [(data[name][nominator], data[name][denominator]) for (nominator, denominator) in features_list]

        ratio = [float(data[name][nominator])/float(data[name][denominator])
                 for (nominator, denominator) in features_list
                     if data[name][nominator] <> 'NaN' or data[name][denominator] <> 'NaN']
        if ratio == []: ratio = [0, 0]

        # print 'ratio', ratio # testing ratio

        for i in range(len(feature_names)):
            # print feature_names[i], ratio[i]
            data[name][feature_names[i]] = ratio[i]

        # print data[name][feature_names[0]] # testing ratio
        # print data[name][feature_names[1]] # testing ratio

    return data

def data_explore(data, features):
    print "\nData Exploration:"
    import pandas as pd
    nan_dict = {}

    # regular statistics
    print "Total Number of Data Points:", len(data.keys())
    print "Allocation of POI Accross Dataset:", \
        "{:0.2f}%".format(sum([1.0 for value in data.values()
                                  if value['poi'] == True]) / len(data.keys())*100)
    print "Total Features:", len(data[data.keys()[0]])

    # missing value calculations
    for feature in features:
        nan_dict[feature] = len([feat_val[feature] for feat_val in data.values()
                                 if feat_val[feature] == 'NaN'])

    #
    many_missing = [1 for value in nan_dict.values() if value > len(data.keys())*.20]

    print "There are", sum(many_missing), "features with at least 20% of missing values."

    nan_dict_frame = {}
    nan_dict_frame['features'] = nan_dict.keys()
    nan_dict_frame['values'] = nan_dict.values()
    nan_dict_frame['percentage'] = ["{:0.2f}".format(float(value)/len(data.keys())*100)
                                    for value in nan_dict.values()]

    nan_dict_frame = pd.DataFrame(nan_dict_frame)

    print nan_dict_frame.sort_values(['values'], ascending=[False])




