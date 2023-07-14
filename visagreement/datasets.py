import pandas as pd
import numpy as np

import os


class Datasets():
    
    def __init__(self, name):
        self.name = name
        self.directory_source = "source"

        directory = './' + self.directory_source + '/'
        
        subdirectories = []
        for name_subdirectory in os.listdir(directory):
            full_path = os.path.join(directory, name_subdirectory)
            if os.path.isdir(full_path):
                subdirectories.append(name_subdirectory)
                
        if name not in subdirectories:
            raise FileNotFoundError('Data not found.')
            
    def get_feature_names(self):
        file_path = './' + self.directory_source + '/' + self.name + '/' + 'X_train.csv'
        X_train = pd.read_csv(file_path)
        feature_names = X_train.columns.to_list()
        return feature_names
    
    def get_train_dataset(self):
        file_path = './' + self.directory_source + '/' + self.name + '/' + 'X_train.csv'
        X_train = pd.read_csv(file_path)
        return X_train
    
    def get_test_dataset(self):
        file_path = './' + self.directory_source + '/' + self.name + '/' + 'X_test.csv'
        X_test = pd.read_csv(file_path)
        return X_test
    
    def get_predictions(self):
        file_path = './' + self.directory_source + '/' + self.name + '/' + 'test_prediction_result.csv'
        predictions = pd.read_csv(file_path)
        return predictions