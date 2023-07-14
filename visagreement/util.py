import numpy as np
import pandas as pd
import os
import re

import visagreement.metrics as mt
from visagreement.feature_importance import FeatureImportance

from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


class Util():
    
    def __init__(self, dataset):
        feature_importance = FeatureImportance(dataset)
        self.name = dataset
        self.directory_source = "source"
        self.directory_explanations = "feature_importance"
        
        
    def get_dict_topk(self, k=5):
        dict_topk = {}
        directory = './'+ self.directory_source + '/'+ self.name + '/' + self.directory_explanations + '/'
        for file_name in os.listdir(directory):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory, file_name)
                df = pd.read_csv(file_path)
                method_name = re.search('(.+?).csv', file_name)
                topk = mt.get_top_k(df, k)
                dict_topk[method_name.group(1)] = topk
                
        return dict_topk
    
    
    def control_points_position(self, samples):
        positions = []

        line1 = np.arange(0.0, 0.34, (1/3)/5)
        line2 = np.arange(0.0, 0.67, (2/3)/14)
        line3 = np.arange(0.0, 1.01, 1/19)
        line4 = np.arange(0.0, 0.67, (2/3)/14)
        line5 = np.arange(0.0, 0.34, (1/3)/5)

        for point in samples:
            if np.sum(point)==0:
                positions.append(np.array([0,0]))
            elif np.sum(point)==6:
                positions.append(np.array([1,1]))
            elif np.sum(point)==1:
                x=line1[0]
                y=(1/3)-line1[0]
                positions.append(np.array([x,y]))
                line1 = np.delete(line1, 0)
            elif np.sum(point)==2:
                x=line2[0]
                y=(2/3)-line2[0]
                positions.append(np.array([x,y]))
                line2 = np.delete(line2, 0)
            elif np.sum(point)==3:
                x=line3[0]
                y=1-line3[0]
                positions.append(np.array([x,y]))
                line3 = np.delete(line3, 0)
            elif np.sum(point)==4:
                temp_x = line4+(1/3)
                x = temp_x[0]
                temp_y = 1-line4
                y = temp_y[0]
                positions.append(np.array([x,y]))
                line4 = np.delete(line4, 0)
            elif np.sum(point)==5:
                temp_x = line5+(2/3)
                x = temp_x[0]
                temp_y = 1-line5
                y = temp_y[0]
                positions.append(np.array([x,y]))
                line5 = np.delete(line5, 0)
            else:
                positions.append(np.array([-1,-1]))
                
        return positions
    
    
    def get_areas_by_distance(self, data_proj, num_instances, point_disagreement=1, radius=0.15):
        '''
        Funciona apenas para 4 métodos, ou seja, 6 combinações
        '''
        dist_class = []
        for i in np.arange(0,num_instances,1):
            x = data_proj[:num_instances,0][i]
            y = data_proj[:num_instances,1][i]
            dst_agree = distance.euclidean((0,0), (x,y))
            dst_disagree = distance.euclidean((point_disagreement,point_disagreement), (x,y))
            
            if dst_agree < radius:
                dist_class.append('Disagreement Area')
            elif dst_disagree < radius:
                dist_class.append('Agreement Area')
            else:
                dist_class.append('Neutral Area')
        return np.asarray(dist_class)