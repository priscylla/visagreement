import pandas as pd
import numpy as np
import os
import re

import torch
torch.manual_seed(42)
import torch.nn as nn
from torch.autograd import Variable

from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    FeatureAblation,
    FeaturePermutation,
    ShapleyValueSampling,
    Lime,
    KernelShap,
    LRP,   
)

from captum.metrics import (
    infidelity,
    sensitivity_max,
    infidelity_perturb_func_decorator,
)


class Visagreement():
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.X_test = None
        self.feature_names = None
        self.directory_source = "source"
        self.directory_explanations = "feature_importance"
        self.directory_quality = "quality_measures"

    def load_model(self, model, X_train, X_test, y_test):
        self.__create_directory()
        self.model = model
        
        try:
            X_train.to_csv('./'+ self.directory_source + '/'+ self.name +'/X_train.csv', index=False)
            X_test.to_csv('./'+ self.directory_source + '/'+ self.name +'/X_test.csv', index=False)
            y_test.to_csv('./'+ self.directory_source + '/'+ self.name +'/y_test.csv', index=False)
            
            self.X_test = X_test
            self.feature_names = X_train.columns.to_list()
        except AttributeError:
            raise AttributeError(
                f"X_train, X_test and y_test must be a Pandas Dataframe."
            )
            
        test_var = Variable(torch.FloatTensor(X_test.to_numpy()), requires_grad=True)
        with torch.no_grad():
            result = model.predict(X_test)
        
        if result.shape[1] > 2 or result.shape[1] == 1:
            raise NotImplementedError('NotImplementedError.')
        if result.shape[1] == 2:
            test_prediction = pd.DataFrame(result, columns = ['prob_0','prob_1'])
            result = torch.from_numpy(result.astype(np.float32))    
            values, labels = torch.max(result, 1)
            y_pred = labels.detach().numpy()
            test_prediction['prediction'] = y_pred
            test_prediction['real_class'] = y_test.to_numpy()
            test_prediction.to_csv('./'+ self.directory_source + '/'+ self.name +'/test_prediction_result.csv', index=False)
        
    def load_explanation(self, name, explanation, measures):
        self.__create_directory_explanations()
        self.__create_directory_quality()
        try:
            explanation.to_csv('./'+ self.directory_source + '/'+ self.name + '/' + self.directory_explanations + '/'+ name +'.csv', index=False)
            measures.to_csv('./'+ self.directory_source + '/'+ self.name + '/' + self.directory_quality + '/'+ name +'.csv', index=False)
        except AttributeError:
            raise AttributeError(
                f"explanation and measures must be a Pandas Dataframe."
            )

        
    def create_explanations(self, name, func_explanation_method):
        self.__create_directory_explanations()
        attributions_feature_importance = []
        for input_ in self.X_test.to_numpy():
            input_ = torch.from_numpy(input_.astype(np.float32))
            input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
            input_ = torch.reshape(input_, (1, len(self.feature_names)))
            input_.requires_grad_()
            attr_method = func_explanation_method(self.model) #ig
            attributions = attr_method.attribute(input_, target=1)
            attributions = attributions.detach().numpy()
            attributions_feature_importance.append(attributions[0])

        feature_importance = pd.DataFrame(attributions_feature_importance, columns=self.feature_names)
        feature_importance.to_csv('./'+ self.directory_source + '/'+ self.name + '/' + self.directory_explanations + '/'+ name +'.csv', index=False)
        
        self.__create_quality_measures(name, attr_method)

        
    def __create_quality_measures(self, method_name, attr_method):
        
        multipy_by_inputs = False
        @infidelity_perturb_func_decorator(multipy_by_inputs=multipy_by_inputs)
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
            return inputs - noise

        self.__create_directory_quality()
        
        scores_inf = []
        scores_sens = []
        
        file_path = './'+ self.directory_source + '/'+ self.name + '/' + self.directory_explanations + '/' + method_name + '.csv'
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            attributions = torch.from_numpy(row.to_numpy())
            attributions = torch.reshape(attributions, (1, len(df.columns)))
                    
            input_ = self.X_test.iloc[[index]].to_numpy()
            input_ = torch.from_numpy(input_.astype(np.float32))
            input_ = Variable(torch.FloatTensor(input_), requires_grad=True)
            input_ = torch.reshape(input_, (1, len(df.columns)))
            input_.requires_grad_()
                
            infid = infidelity(self.model, perturb_fn, input_, attributions, target=1)
            inf_value = infid.detach().numpy()[0]
            
            sens = sensitivity_max(attr_method.attribute,input_,target=1)
            sens_value = sens.detach().numpy()[0]
                    
            scores_inf.append(inf_value)
            scores_sens.append(sens_value)
                
        scores = pd.DataFrame(scores_inf, columns=['infidelity'])
        scores['sensitivity'] = scores_sens
        scores.to_csv('./'+ self.directory_source + '/'+ self.name + '/' + self.directory_quality + '/'+ method_name +'.csv', index=False)
        
        
        
    def __create_directory(self):

        if not os.path.exists(self.directory_source):
            os.mkdir(self.directory_source)

        directory_subdirectory = os.path.join(self.directory_source, self.name)

        if not os.path.exists(directory_subdirectory):
            os.mkdir(directory_subdirectory)
            
    def __create_directory_explanations(self):
        
        if not os.path.exists(self.directory_source):
            raise FileNotFoundError('load_model function must be executed before loading explanations.')
        
        directory_model = os.path.join(self.directory_source, self.name)
        directory_subdirectory = os.path.join(directory_model, self.directory_explanations)
        
        if not os.path.exists(directory_subdirectory):
            os.mkdir(directory_subdirectory)
            
    def __create_directory_quality(self):
        
        if not os.path.exists(self.directory_source):
            raise FileNotFoundError('load_model function must be executed before loading explanations.')
        
        directory_model = os.path.join(self.directory_source, self.name)
        directory_subdirectory = os.path.join(directory_model, self.directory_quality)
        
        if not os.path.exists(directory_subdirectory):
            os.mkdir(directory_subdirectory)
        