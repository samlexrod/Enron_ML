#!/usr/bin/python

# imports
import sys
import pickle
import numpy as np
from outlier_manage import outlier_dict, pop_selected
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from pprint import pprint
from feature_helper import feat_sum, feat_ratio, data_explore

# sys.path.append("../tools/")

path = "D:\GitHub\Enron_ML"

path = "/".join(path.split("\\")) + "/"
sys.path.append(path)



# Task 1: Select what features you'll use
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# from outlier_show import outlier_hist

# determining outliers
# quantitative features with outlier values
# analyze the outliers
features_available = data_dict.items()[0][1].keys()
outlier_dict(features_available, data_dict)

# delete outliers
pop_list = ['TOTAL']

pop_selected(pop_list, features_available, data_dict)

# Task 3: Create new feature(s)
print '\nData Structure Before Feature Addition:'
pprint(data_dict.values()[0])

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                      'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

ratio_email_features = [['from_this_person_to_poi', 'from_messages'],
                        ['from_poi_to_this_person', 'to_messages']]

feat_ratio(data_dict, ratio_email_features, ['from_message_impact', 'to_messages_impact'])

feat_sum(data_dict, 'total_compensation', financial_features)

# Store to my_dataset for easy export below.
my_dataset = data_dict

# validating additional feature(s)
print '\nData Structure After Addition:'
pprint(my_dataset.values()[0])

data_explore(my_dataset, financial_features)

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = GaussianNB()

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)