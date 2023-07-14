import pandas as pd
import numpy as np
import os
import re

class FeatureImportance():
    
    
    def __init__(self, name):
        self.name = name
        self.directory_source = "source"
        self.directory_explanations = "feature_importance"
        self.directory_quality = "quality_measures"
        
    
    def get_measures_for_explanations(self, methods, selected_data=None):
        explanation_methods = []
        directory = './'+ self.directory_source + '/'+ self.name + '/' + self.directory_quality + '/'
        for file_name in os.listdir(directory):
            if file_name.endswith('.csv'):
                method_name = re.search('(.+?).csv', file_name)
                if method_name.group(1) in methods:
                    file_path = os.path.join(directory, file_name)
                    df = pd.read_csv(file_path)
                    explanation_methods.append(method_name.group(1))
                
        scores_infidelity = []
        scores_sensitivity = []
        for method in explanation_methods:
            path = directory + method +'.csv'
            df = pd.read_csv(path)
            if selected_data is not None:
                df = df.iloc[selected_data]
            mean = df.mean(axis=0)
            min_value = df.min(axis=0)
            max_value = df.max(axis=0)
            median_value = df.median(axis=0)
            scores_infidelity.append([self.name, self.name, method, min_value[0], median_value[0], max_value[1], mean[0]])
            scores_sensitivity.append([self.name, self.name, method, min_value[1], median_value[1], max_value[1], mean[1]])
        
        cols = ['model', 'dataset', 'method', 'min', 'median', 'max', 'mean']
        scores_infidelity_df = pd.DataFrame(np.array(scores_infidelity), columns=cols)
        scores_sensitivity_df = pd.DataFrame(np.array(scores_sensitivity), columns=cols)
        return scores_infidelity_df, scores_sensitivity_df   