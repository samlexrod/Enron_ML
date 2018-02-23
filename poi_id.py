#!/usr/bin/python

# imports
import sys
import pickle
from outlier_manage import outlier_dict, pop_selected, find_non_nan
from my_tester import my_dump_classifier_and_data, my_test_classifier
from sklearn.naive_bayes import GaussianNB
from feature_helper import feat_sum, feat_ratio, data_explore, data_print, nan_handler
from sklearn.svm import SVC
from sklearn.decomposition import pca
from sklearn.tree import DecisionTreeClassifier
from classify_helper import classify_tuner, auto_feature, avg_eval_metrics
from sklearn.ensemble import RandomForestClassifier

# sys.path.append("../tools/")

path = "D:\GitHub\Enron_ML"

path = "/".join(path.split("\\")) + "/"
sys.path.append(path)

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# REMOVING OUTLIERS
# *****************
# determining outliers
# quantitative features with outlier values
# analyze the outliers
outlier_dict(data_dict)

# data exploration before outlier removal
data_explore(data_dict)

# delete outliers
pop_list = ['TOTAL']
pop_selected(data_dict, pop_list)

find_non_nan(data_dict, pop=True)

# CREATING NEW FEATURES
# *********************
# it prints a dictionary of features
# and its value
data_print(data_dict, after_feat=False)

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                      'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

ratio_email_features = [['from_this_person_to_poi', 'from_messages'],
                        ['from_poi_to_this_person', 'to_messages']]
feat_ratio(data_dict, ratio_email_features, ['from_message_impact', 'to_messages_impact'])

feat_sum(data_dict, 'total_compensation', financial_features)

ratio_financial_features = [['salary', 'total_compensation_abs']]
feat_ratio(data_dict, ratio_financial_features, ['salary_impact'])

# it prints the dictionary of features and its values
# after addition of custom features to validate
data_print(data_dict, after_feat=True)


# EXPLORING, HANDLING NULLS, AND STORING DATA
# *******************************************
#  Store to my_dataset for easy export below.
my_dataset = data_dict

# data exploration after outlier removal and
# addition of custom features
data_explore(my_dataset)

#  Converting null to zero value
#nan_handler(my_dataset)
#data_explore(my_dataset)


# AUTOMATICALLY LOOK FOR FEATURES AND SELECT CLASSIFIER
# *****************************************************
#  preparing the features to add to test for better score
additional_features = my_dataset.values()[0].keys()
additional_features.remove('salary')
additional_features.remove('poi')
additional_features.remove('email_address')
initial_features = ['poi', 'salary']

print "Additional Features:", len(additional_features)

#  initiating automatic feature search
'''
final_features_SVC = auto_feature(SVC(),
                                 my_dataset, additional_features, initial_features, iterate=2)
final_features_NB = auto_feature(GaussianNB(),
                                 my_dataset, additional_features, initial_features, iterate=2)
final_features_FR = auto_feature(RandomForestClassifier(),
                                 my_dataset, additional_features, initial_features, iterate=1)
                                 '''


# OPTIMIZING SELECTED CLASSIFIER
# ******************************
#   optimizing features in classifier using default parameters

clf_def = DecisionTreeClassifier()
optimal_features = auto_feature(clf_def, my_dataset, additional_features, initial_features, iterate=5)


# TUNE SELECTED CLASSIFIER
# *************************
#   tuning the classifier using optimal features

parameters_DT = {'criterion': ['gini', 'entropy'], 'min_samples_split': range(2, 50, 2),
                 'splitter': ['best', 'random'], 'max_depth': [None, 1, 100],
                 'min_samples_leaf': range(1, 60, 2)}
clf_best_estimator = classify_tuner(clf_def, my_dataset, optimal_features,
                                            parameters=parameters_DT, tune_size=.5)


# LAST OPTIMAL TEST
# *****************
#   testing classifier with suggested parameters
print "\n-> Testing Classifier with New Parameters..."
my_test_classifier(clf_best_estimator, my_dataset, optimal_features)

#   optimizing features of tuned classifier
print "\tContinue folding and optimizing features with new parameters focusing on maximizing recall:"
optimal_features_tune = auto_feature(clf_best_estimator, my_dataset, additional_features, initial_features,
                                iterate=1, max_eval_foc='reca')

#   testing classifier with suggested parameters and newly optimal features
print "\n-> Testing Classifier with New Parameters..."
my_test_classifier(clf_best_estimator, my_dataset, optimal_features_tune) # parameter fails the test

# DUMPING
# *******
clf_dump = DecisionTreeClassifier() # default parameters are selected

# getting optimal averages
avg_eval_metrics(clf_dump, my_dataset, optimal_features, sampling_size=30)

my_dump_classifier_and_data(clf_dump, my_dataset, optimal_features)


