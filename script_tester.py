feature_dict = {}
accuracy = .89
precision = .30
recall = .30
features = ['poi', 'salary', 'shared_receipt_with_poi', 'loan_advances', 'director_fees', 'exercised_stock_options', 'total_compensation_abs']
test = "test_number{}".format(1)
feature_dict.update({test: {'accuracy': accuracy,
                       'precision': precision,
                       'recall': recall,
                       'features': features}})

accuracy = .90
precision = .40
recall = .40
features = ['poi', 'salary', 'shared_receipt_with_poi', 'loan_advances', 'director_fees', 'exercised_stock_options', 'total_compensation_abs']
test = "test_number{}".format(2)
feature_dict.update({test: {'accuracy': accuracy,
                       'precision': precision,
                       'recall': recall,
                       'features': features}})


print feature_dict

