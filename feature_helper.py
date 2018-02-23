import pandas as pd

# the function wil calculate the sum of all stated features
# the features to sum should go under the feat_sum_list with this format: [a, b, c, ...]
# the name of the new feature should go in the new_feature_str as a string
# having absol set to true will convert all features to positive numbers and then add to avoid subtraction
def feat_sum(data, new_feature_str, feat_sum_list, absol = True):
    # sum all values from selected features
    # add a new key to the dictionary and returns it
    # to create a new feature

    absol_name = new_feature_str + '_abs'
    for name in data.keys():

        if absol:
            total = sum([abs(data[name][feature]) for feature in feat_sum_list if data[name][feature] <> 'NaN'])
            new_feature_str = absol_name
        else:
            total = sum([data[name][feature] for feature in feat_sum_list if data[name][feature] <> 'NaN'])

        data[name][new_feature_str] = total

        # validating aggregation
        '''
        print "Compare to: ", [(feature, data[name][feature]) for feature in feat_sum_list]
        print "Analyze negative: ", [(feature, data[name][feature]) for feature in feat_sum_list \
                                     # if data[name][feature] < 0]
        print "Returning:", data[name][new_feature_str]
        '''

    return data

# the function will calculate the ratio of given combination of values, nominator and denominator
# the feature_list should be entered in this format [[a, b], [c, d]]
# the name of the new feature should be entered in this format: [a, b]
def feat_ratio(data, features_lists, feature_names):
    for name in data.keys():

        # Calculating the ratio
        feat_name_index = 0
        for features_list in features_lists:
            nominator = features_list[0]
            denominator = features_list[1]

            if data[name][nominator] == 'NaN' or data[name][denominator] == 'NaN':
                data[name][feature_names[feat_name_index]] = 0.
            elif data[name][nominator] == 0 or data[name][denominator] == 0:
                data[name][feature_names[feat_name_index]] = 0.
            else:
                data[name][feature_names[feat_name_index]] = float(data[name][nominator])/float(data[name][denominator])

            # Validating the calculations
            '''
            print name, feature_names[feat_name_index], \
                data[name][feature_names[feat_name_index]], "<-",\
                (data[name][nominator], data[name][denominator])
            '''

            feat_name_index += 1

    return data

# the function will explore total number of data points, allocation of pois, total features,
# and missing values of given features
# features should be entered in this format: [a, b, c, ...]
def data_explore(dataset):

    print "\nData Exploration:"
    nan_dict = {}
    features = dataset.values()[0].keys()

    data_points = len(dataset.keys())
    poi_number = sum([1 for name in dataset.keys() if dataset[name]['poi']])
    poi_alloca = sum([1.0 for value in dataset.values() if value['poi']]) / len(dataset.keys())
    total_feat = len(dataset[dataset.keys()[0]])

    # regular statistics
    print "Total Number of Data Points: {}" \
          "\nTotal Number of POI: {}" \
          "\nAllocation of POI Across Dataset: {:0.2%}" \
          "\nTotal Features: {}"\
        .format(data_points, poi_number, poi_alloca, total_feat)

    # missing value calculations
    for feature in features:
        nan_dict[feature] = len([feat_val[feature] for feat_val in dataset.values()
                                 if feat_val[feature] == 'NaN'])

    #
    many_missing = [1 for value in nan_dict.values() if value > len(dataset.keys())*.20]

    print "There are", sum(many_missing), "features with at least 20% of missing values."

    nan_dict_frame = {}
    nan_dict_frame['features'] = nan_dict.keys()
    nan_dict_frame['miss_values'] = nan_dict.values()
    nan_dict_frame['percentage'] = ["{:0.2f}".format(float(value)/len(dataset.keys())*100)
                                    for value in nan_dict.values()]

    nan_dict_frame = pd.DataFrame(nan_dict_frame)

    print nan_dict_frame.sort_values(['miss_values'], ascending=[False])
    print '-'*50

def nan_handler(my_dataset):

    features = my_dataset.values()[0].keys()
    names = my_dataset.keys()

    # replacing null values to zero float values
    for name in names:
        for feature in features:
            if my_dataset[name][feature] == 'NaN':
                my_dataset[name][feature] = 0.
    return my_dataset

# the feature prints the data as it was before or after new features
# it is just to have a cleaner code in the machine_learning_udacity script
def data_print(data, after_feat=True):
    from pprint import pprint

    if after_feat:
        print '\nData Structure After Feature Addition:'
        pprint(data.values()[0])
        print '-'*50
    else:
        print '\nData Structure Before Feature Addition:'
        pprint(data.values()[0])
        print '-'*50