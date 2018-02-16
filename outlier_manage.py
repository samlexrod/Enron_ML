import numpy as np
import pprint as pp

def outlier_dict(features, my_dataset):
    outlier_dict = {}
    
    for feature in features:        
         
        # eliminating null values
        data_feat_zip = [[key, value[feature]] for key, value in 
                         zip(my_dataset.keys(), my_dataset.values()) 
                         if value[feature] <> 'NaN']        
        
        # split data
        data_feat = [i[1] for i in data_feat_zip]
        
        # include only features with int or float data values
        if type(data_feat[0]) == int or type(data_feat[0]) == float:
            
            # measures of central tendency
            mu = np.mean(data_feat)
            sd = np.std(data_feat)
            
            # zipping only keys and values with outlier values
            outlier_zip = [[key, value] for key, value in data_feat_zip 
                           if value > mu+3*sd or value < mu-3*sd]
            
            # separating keys and outlier values
            outlier_names = [key[0] for key in outlier_zip]            
            outlier_values = [value[1] for value in outlier_zip]
            
            if len(outlier_values) > 0:
                outlier_dict[feature] = outlier_names

    print "\nFinding Outliers:"
    pp.pprint(outlier_dict)
    print "-"*50

def pop_selected(pop_list, features, data):
    
    if len(pop_list) <= len(features):
        pop_array = [feature.upper() for feature in pop_list]

        try:
            [data.pop(feature) for feature in pop_list]
            print "\nRemoving Outliers:"
            print '{} key(s) removed from dataset'.format(len(pop_list))
            print "-"*50
            outlier_dict(features, data)            
        except:
            print 'Error: At least one key was not found. Check spelling and try again.'
    else:
        print 'Error: Wrong arguments... try: (pop_list, features, data).'
    
  
        
