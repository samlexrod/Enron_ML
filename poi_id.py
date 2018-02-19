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
from classify_helper import classify, auto_feature
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
features_for_testing = my_dataset.values()[0].keys()
features_for_testing.remove('salary')
features_for_testing.remove('poi')
features_for_testing.remove('email_address')
initial_features = ['poi', 'salary']

#  initiating automatic search
# final_features_SVC = auto_feature(SVC(),
                                 # my_dataset, features_for_testing, initial_features, show=False)
# final_features_NB = auto_feature(GaussianNB(),
                                 # my_dataset, features_for_testing, initial_features, show=False)
# final_features_FR = auto_feature(RandomForestClassifier(),
                                 # my_dataset, features_for_testing, initial_features, show=False)

#  selected classifier
final_features_DT = auto_feature(DecisionTreeClassifier(),
                                 my_dataset, features_for_testing, initial_features, show=False)


# TUNE IN THE SELECTED CLASSIFIER
parameters_DT = {'criterion': ['gini', 'entropy'], 'min_samples_split': range(2, 50, 2),
                 'splitter': ['best', 'random'], 'max_depth': [None, 1, 100],
                 'min_samples_leaf': range(1, 60, 2), 'max_features': [None, 'auto', 'sqrt', 'log2']}
classify(DecisionTreeClassifier(), my_dataset, final_features_DT, fine_tune=False, parameters=parameters_DT)


# TEST FINAL STEPS
clf_DT_clean = DecisionTreeClassifier()

clf_DT_grid = DecisionTreeClassifier(criterion='gini',
                                max_depth=None, max_features='log2',
                                min_samples_leaf=1, min_samples_split=44,
                                splitter='random')
clf_DT_manual = DecisionTreeClassifier(criterion='gini',
                                max_depth=None, max_features='log2',
                                min_samples_leaf=1, min_samples_split=2,
                                splitter='random')


#  find optimal Parameters & Features Test
#  running with GridSearchCV suggested parameters
print "GridSearchCV suggestions"
print final_features_DT
for i in range(0, 2):
    test_classifier(clf_DT_grid, my_dataset, final_features_DT)
print "-"*50
#  running clean
print "Default Parameters"
print final_features_DT
for i in range(0, 2):
    test_classifier(clf_DT_clean, my_dataset, final_features_DT)
print "-"*50
#  running manual
print "Manual Parameters"
print final_features_DT
for i in range(0, 2):
    test_classifier(clf_DT_manual, my_dataset, final_features_DT)
print "-"*50


#  after running the test set the optimal features here:
optimal_features = ['poi', 'salary', 'to_messages', 'to_messages_impact', 'restricted_stock', 'exercised_stock_options']
second_optimal_features = ['poi', 'salary', 'deferred_income', 'exercised_stock_options']

print "Final Selection"
print optimal_features
for i in range(0, 2):
    test_classifier(clf_DT_clean, my_dataset, optimal_features)
print "-"*50

clf = clf_DT_clean
dump_classifier_and_data(clf, my_dataset, optimal_features)
