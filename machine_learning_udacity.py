#!/usr/bin/python

# imports
import sys
import pickle
from outlier_manage import outlier_dict, pop_selected
from tester import dump_classifier_and_data, test_classifier
from sklearn.naive_bayes import GaussianNB
from feature_helper import feat_sum, feat_ratio, data_explore, data_print
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from classify_helper import classify, auto_feature

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

# Task 3: Create new feature(s)
data_print(data_dict, after=False)

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                      'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

ratio_email_features = [['from_this_person_to_poi', 'from_messages'],
                        ['from_poi_to_this_person', 'to_messages']]

feat_ratio(data_dict, ratio_email_features, ['from_message_impact', 'to_messages_impact'])

feat_sum(data_dict, 'total_compensation', financial_features)

# validating additional feature(s)
data_print(data_dict, after=True)

# Store to my_dataset for easy export below.
my_dataset = data_dict

data_explore(my_dataset, my_dataset.values()[0].keys())

# Extract features and labels from dataset for local testing
# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# preparing the features to add to test for better score
features_for_testing = my_dataset.values()[0].keys()
features_for_testing.remove('salary')
features_for_testing.remove('poi')
features_for_testing.remove('email_address')

initial_features = ['poi', 'salary']

final_features_SVC = auto_feature(SVC(kernel='rbf'),
                                  my_dataset, features_for_testing, initial_features)
final_features_NB = auto_feature(GaussianNB(),
                                 my_dataset, features_for_testing, initial_features)
final_features_DT = auto_feature(DecisionTreeClassifier(),
                                 my_dataset, features_for_testing, initial_features, show=False)
from sklearn.ensemble import RandomForestClassifier
final_features_FR = auto_feature(RandomForestClassifier(),
                                 my_dataset, features_for_testing, initial_features, show=False)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': [0, 0.1]}
classify(SVC(), my_dataset, final_features_SVC, fine_tune=False, parameters=parameters, tune_size= .40)
classify(SVC(kernel='rbf',
             C=1,
             gamma=0.01),
         my_dataset, final_features_SVC, fine_tune=False, parameters=parameters, tune_size= .40)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

# selected classifier

print "\nStarting test..."
clf_DT = DecisionTreeClassifier()
test_classifier(clf_DT, my_dataset, final_features_DT)

clf_RF = RandomForestClassifier()
test_classifier(clf_RF, my_dataset, final_features_FR)

clf_NB = GaussianNB()
test_classifier(clf_NB, my_dataset, final_features_NB)

clf_SVC = SVC()
test_classifier(clf_SVC, my_dataset, final_features_SVC)


#features_final = final_features_SVC
#dump_classifier_and_data(clf, my_dataset, features_final)

