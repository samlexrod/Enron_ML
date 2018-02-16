

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

def feat_ratio():
    pass
