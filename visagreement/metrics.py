import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import combinations


def get_top_k(data, k):
    '''
    Return dict with a sorted series where the name of feature is the index and value with sign
    '''
    dict_topk = {}
    for index, row in data.iterrows():
        features = row.abs().nlargest(k, keep='first')
        dict_topk[index] = features
        for feature_name, value in features.items():
            features[feature_name] = row[feature_name]
    return dict_topk

def feature_agreement(topk_1, topk_2):
    '''
    Return a value indicating the feature agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric feature agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        #calcula a intersecção e divite por k
        return len(set(topk_1.index) & set(topk_2.index)) / len(topk_1)
    
def rank_agreement(topk_1, topk_2):
    '''
    Return a value indicating the rank agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric rank agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        #calcula a quantidade de features na mesma posição em ambos os topk e divite por k
        list1 = topk_1.index.to_list()
        list2 = topk_2.index.to_list()
        return sum(first == second for (first, second) in zip(list1, list2)) / len(topk_1)
    
def sign_agreement(topk_1, topk_2):
    '''
    Return a value indicating the sign agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric sign agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        count_same_sign = (topk_1 * topk_2 >= 0).sum()
        return count_same_sign / len(topk_1)              
    
    
def sign_rank_agreement(topk_1, topk_2):
    '''
    Return a value indicating the signed rank agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric signed rank agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        sameSign = topk_1 * topk_2 >= 0
        sameSign = sameSign[sameSign == True]
        count = 0
        for feature_name, value in sameSign.items():
            pos1 = topk_1.index.to_list().index(feature_name)
            pos2 = topk_2.index.to_list().index(feature_name)
            if pos1 == pos2:
                count+=1
        return count / len(topk_1)
    

def create_matrix_combination_methdos_by_metric(dict_topk, func_metric, num_instancias, methods):
    src = methods
    combinations_methods = []
    for s in combinations(src, 2):
        combinations_methods.append(s)
    list_combinations_methods = [str(t) for t in combinations_methods]
    
    N = len(combinations_methods)
    matrix_points = {}
    matrix_points = np.zeros((num_instancias,N))
    for instance in range(0,num_instancias):
        num_combination = 0
        for method_names in combinations_methods:
            method1 = dict_topk[method_names[0]][instance]
            method2 = dict_topk[method_names[1]][instance]
            metric = func_metric(method1, method2)
            matrix_points[instance][num_combination] = metric
            num_combination += 1
        
    return matrix_points, list_combinations_methods
