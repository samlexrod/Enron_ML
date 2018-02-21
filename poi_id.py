#!/usr/bin/python

# imports
import sys
import pickle
from outlier_manage import outlier_dict, pop_selected
from tester import dump_classifier_and_data, test_classifier
from sklearn.naive_bayes import GaussianNB
from feature_helper import feat_sum, feat_ratio, data_explore, data_print, nan_handler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from classify_helper import classify_tuner, auto_feature, avg_eval_metrics
from sklearn.ensemble import RandomForestClassifier

# sys.path.append("../tools/")

path = "D:\GitHub\Enron_ML"

path = "/".join(path.split("\\")) + "/"
sys.path.append(path)



# Task 1: Select what features you'll use
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list1 = ['poi','salary', 'from_message_impact', 'to_messages_impact'] # You will need to use more features

features_list2 = ['poi', 'total_compensation_abs', 'from_message_impact', 'to_messages_impact']

features_list3 = ['poi', 'salary']

features_lists = [features_list1, features_list2, features_list3]

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

# CREATING NEW FEATURES
data_print(data_dict, after=False)

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

# validating additional feature(s)
data_print(data_dict, after=True)


# EXPLORING, HANDLING NULLS, AND STORING DATA

#  Store to my_dataset for easy export below.
my_dataset = data_dict

all_features = my_dataset.values()[0].keys()

data_explore(my_dataset, all_features)

#  Converting null to zero value
nan_handler(my_dataset)

data_explore(my_dataset, all_features)


# AUTOMATICALLY LOOK FOR FEATURES AND SELECT CLASSIFIER

#  preparing the features to add to test for better score
additional_features = my_dataset.values()[0].keys()
additional_features.remove('salary')
additional_features.remove('poi')
additional_features.remove('email_address')
initial_features = ['poi', 'salary']

#  initiating automatic search
'''
final_features_SVC = auto_feature(SVC(),
                                 my_dataset, additional_features, initial_features, iterate=1)
final_features_NB = auto_feature(GaussianNB(),
                                 my_dataset, additional_features, initial_features)
final_features_FR = auto_feature(RandomForestClassifier(),
                                 my_dataset, additional_features, initial_features)
                                 '''


# OPTIMIZING SELECTED CLASSIFIER

#   optimizing features in classifier using default parameters
clf_def = DecisionTreeClassifier()
optimal_features = auto_feature(clf_def, my_dataset, additional_features, initial_features, iterate=5)

# DEBUG
optimal_features = ['poi', 'salary', 'shared_receipt_with_poi', 'loan_advances',
                    'director_fees', 'exercised_stock_options', 'total_compensation_abs']

# TUNE SELECTED CLASSIFIER

#   tuning the classifier using optimal features

parameters_DT = {'criterion': ['gini', 'entropy'], 'min_samples_split': range(2, 50, 2),
                 'splitter': ['best', 'random'], 'max_depth': [None, 1, 100],
                 'min_samples_leaf': range(1, 60, 2)}
clf_best_estimator = classify_tuner(clf_def, my_dataset, optimal_features,
                                            parameters=parameters_DT, tune_size=.5)
'''
clf_best_estimator = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                            max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=38,
                                            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                            splitter='random')
                                            '''

# LAST OPTIMAL TEST

#   testing classifier with suggested parameters
print "\n-> Testing Classifier with New Parameters..."
test_classifier(clf_best_estimator, my_dataset, optimal_features)

#   optimizing features of tuned classifier
print "\tContinue folding and optimizing features with new parameters focusing on maximizing recall:"
optimal_features_tune = auto_feature(clf_best_estimator, my_dataset, additional_features, initial_features,
                                iterate=5, max_eval_foc='reca')

#   testing classifier with suggested parameters and newly optimal features
print "\n-> Testing Classifier with New Parameters..."
test_classifier(clf_best_estimator, my_dataset, optimal_features_tune) # parameter fails the test

# SELECTING CLASSIFIER TO DUMP
clf_dump = DecisionTreeClassifier() # default parameters are selected

# GETTING OPTIMAL AVERAGES

avg_eval_metrics(clf_dump, my_dataset, optimal_features, sampling_size=30)

dump_classifier_and_data(clf_dump, my_dataset, optimal_features)


