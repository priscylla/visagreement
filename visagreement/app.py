import os
import re

from dash import Dash, dcc, html, dash_table, Input, Output, callback, ctx, State
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import umap
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
import json
from itertools import combinations


import visagreement.metrics as mt

from visagreement import (
    FeatureImportance, Util, Datasets, Lamp
)

directory = './source/'
dataset_names = []
for name_subdirectory in os.listdir(directory):
    full_path = os.path.join(directory, name_subdirectory)
    if os.path.isdir(full_path):
        dataset_names.append(name_subdirectory)

datasets_list = {}
def get_umap(dataset_names):
    umap_values = {}
    size_datasets = {} 
    for name in dataset_names:
        dataset = Datasets(name)
        datasets_list[name] = dataset
        X_test = dataset.get_test_dataset()
        reducer = umap.UMAP(random_state=42)
        X_test_umap = reducer.fit_transform(X_test)
        umap_values[name] = X_test_umap
        size = X_test_umap.shape[0]
        size_datasets[name] = size
    return umap_values, size_datasets

def get_tsne(dataset_names):
    tsne_values = {}
    for name in dataset_names:
        dataset = Datasets(name)
        datasets_list[name] = dataset
        X_test = dataset.get_test_dataset()
        reducer = TSNE(n_components=2, learning_rate='auto', random_state=42)
        X_test_tsne = reducer.fit_transform(X_test)
        tsne_values[name] = X_test_tsne
    return tsne_values
    
    
umap_datasets, size_datasets = get_umap(dataset_names)
tsne_datasets = get_tsne(dataset_names)
default_dataset = dataset_names[0]
explanation_methods_names = []
for file_name in os.listdir('./source/'+ default_dataset + '/feature_importance/'):
    if file_name.endswith('.csv'):
        method_name = re.search('(.+?).csv', file_name)
        explanation_methods_names.append(method_name.group(1))
        
default_methods = explanation_methods_names[:4]
dataset_names = [dataset_names[0]]

#################################################
# App
#################################################
# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
app.title = "Tool v1.0.0"

#################################################
# Data
#################################################

options = {
    "feature agreement": mt.feature_agreement,
    "rank agreement": mt.rank_agreement,
    "sign agreement": mt.sign_agreement,
    "signed rank agreement": mt.sign_rank_agreement,
}

metrics = ['feature agreement','rank agreement','sign agreement','signed rank agreement']

#################################################
# Layout
#################################################

################ Banner ############################
header = html.H4(
    "Visagreement", className="bg-primary text-white p-2 mb-2 text-center"
)

################ Controls ############################

dropdown_dataset = html.Div(
    [
        dbc.Label("Select dataset and model"),
        dcc.Dropdown(
            id='dataset_dropdown',
            value=default_dataset,
            clearable=False,
            options=[{'label': x, 'value': x} for x in dataset_names]
        ),
    ],
    className="mb-4",
)

dropdown_metric = html.Div(
    [
        dbc.Label("Select metric"),
        dcc.Dropdown(
            id='metric_dropdown',
            value='rank agreement',
            clearable=False,
            options=[{'label': x, 'value': x} for x in sorted(metrics)]
        ),
    ],
    className="mb-4",
)

dropdown_ranking_size = html.Div(
    [
        dbc.Label("Top features size"),
        dcc.Dropdown(
            id='ranking_size_dropdown',
            value=5,
            clearable=False,
            options=[{'label': x, 'value': x} for x in np.arange(1,61,1)]
        ),
    ],
    className="mb-4",
)

dropdown_explanation_methods = html.Div(
    [
        dbc.Label("Explanation methods"),
        dcc.Dropdown(
            id="explanation_methods_dropdown",
            options=[{'label': x, 'value': x} for x in sorted(explanation_methods_names)],
            value=default_methods,
            multi=True,
        ),
        html.Div(id="methods_warning"),
    ],
    className="mb-4",
)

dropdown_calc_areas = html.Div(
    [
        dbc.Label("Areas definition"),
        dcc.Dropdown(
            id="calc_areas_dropdown",
            value='Automatic',
            options=[{'label': x, 'value': x} for x in ['Automatic', 'Manual']],
        ),
    ],
    #className="mb-4",
)

input_radio = html.Div(
    [
        dbc.Label("Threshold"),
        dcc.Input(
            id="input_threshold_radio", type="number", value=0.35,
            min=0.05, max=0.7, step=0.05,
        ),
    ],
    #className="mb-4",
)

button_reload = html.Div(
    [
        dbc.Button('Reload', id='button-explanations', color="primary", className="me-1"),
    ]
)

controls = dbc.Card(
    [dropdown_dataset, dropdown_metric, dropdown_ranking_size, dropdown_explanation_methods, dropdown_calc_areas, input_radio, button_reload],
    body=True,
    #style={'height':'60vh'},
)

dropdown_projector = html.Div(
    [
        dbc.Label("Projector: "),
        dcc.Dropdown(
            id='projector_dropdown',
            value='UMAP',
            clearable=False,
            options=[{'label': x, 'value': x} for x in ['UMAP', 't-SNE']]
        ),
    ],
    #className="mb-4",
)


################ Charts ############################

lamp_graph = dcc.Graph(figure={})
umap_graph = dcc.Graph(figure={})



graph_measures_infidelity = dcc.Graph(figure={})
graph_measures_sensitivity = dcc.Graph(figure={})

heatmap_agreement = dcc.Graph(figure={})
heatmap_fuzzy = dcc.Graph(figure={})
heatmap_disagreement = dcc.Graph(figure={})

confusion_matrix_agree = dcc.Graph(figure={})
confusion_matrix_fuzzy = dcc.Graph(figure={})
confusion_matrix_disagree = dcc.Graph(figure={})

model_metrics_agree = dcc.Graph(figure={})
model_metrics_fuzzy = dcc.Graph(figure={})
model_metrics_disagree = dcc.Graph(figure={})


features_graph_agree = dcc.Graph(figure={})
features_graph_fuzzy = dcc.Graph(figure={})
features_graph_disagree = dcc.Graph(figure={})


################ Tabs ############################
tab1 = dbc.Tab(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card([dbc.Label("Sensitivity"), graph_measures_sensitivity], body=True),width=6),
                dbc.Col(dbc.Card([dbc.Label("Infidelity"), graph_measures_infidelity], body=True), width=6),   
            ],
        ),
    ],
    label='Explanations Quality'
)

tab2 = dbc.Tab(
    [
        dbc.Row(
            dbc.CardGroup(
                [
                    dbc.Card([dbc.Label("Agreement Area"), heatmap_agreement], body=True),
                    dbc.Card([dbc.Label("Disagreement Area"),heatmap_disagreement], body=True),
                    dbc.Card([dbc.Label("Neutral Area"),heatmap_fuzzy], body=True),
                ],
            ),
        ),
    ],
    label='Explanation (Dis)Agreement'
)

tab3 = dbc.Tab(
    [
        dbc.Row(
            dbc.CardGroup(
                [
                    dbc.Card([dbc.Label("Confusion Matrix - Agreement Area"), confusion_matrix_agree], body=True),
                    dbc.Card([dbc.Label("Confusion Matrix - Disagreement Area"), confusion_matrix_disagree], body=True),
                    dbc.Card([dbc.Label("Confusion Matrix - Neutral Area"), confusion_matrix_fuzzy], body=True),
                ],
            
            ),
        ),
        dbc.Row(
            dbc.CardGroup(
                [
                    dbc.Card([dbc.Label("Classification Metrics - Agreement Area"), model_metrics_agree], body=True),
                    dbc.Card([dbc.Label("Classification Metrics - Disagreement Area"), model_metrics_disagree], body=True),
                    dbc.Card([dbc.Label("Classification Metrics - Neutral Area"), model_metrics_fuzzy], body=True),
                ],
            
            ),
        ),
    ],
    label='Model Accuracy'
)

tab4 = dbc.Tab(
    [
        dbc.Row(
            dbc.CardGroup(
                [
                    dbc.Card([dbc.Label("Agreement Area"), features_graph_agree], body=True),
                    dbc.Card([dbc.Label("Disagreement Area"), features_graph_disagree], body=True),
                    dbc.Card([dbc.Label("Neutral Area"), features_graph_fuzzy], body=True), 
                ],
            )
        ),
    ],
    label='Features Disagreement'
)

tabs_explanations = dbc.Card(dbc.Tabs([tab1, tab2, tab3, tab4]))

################ APP layout ############################


app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls,
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.CardGroup(
                                [
                                    dbc.Card(
                                        [
                                            html.H4("(Dis)Agreement Space"),
                                            html.Button('Disagreement', id='btn-disagrement', disabled=False),
                                            html.Button('Agreement', id='btn-agreement', disabled=False),
                                            html.Button('Clear', id='btn-clear', disabled=False),
                                            dcc.Loading(id = "loading-icon-lamp", children=[lamp_graph], type="default"),
                                        ],
                                        body=True,
                                    ),
                                    dbc.Card(
                                        [
                                             html.H4("Feature Space"),
                                            dropdown_projector, 
                                            dcc.Loading(id = "loading-icon-embbeding", children=[umap_graph], type="default"),
                                            #umap_graph
                                        ],
                                        body=True,
                                    ),
                                ],
                            )
                        ),
                        html.Br(),
                        dbc.Row(
                            dcc.Loading(id = "loading-icon-tabs", children=[tabs_explanations], type="default"),
                        ),
                    ],
                    width=10,
                ),
            ],
        ),
        dcc.Store(id='lamp-value'),
        dcc.Store(id='matrix-points-value'),
        dcc.Store(id='list-methods-value'),
    ],
    fluid=True,
    className="dbc",
)

#############################################################################
###################### methods to draw the charts
#############################################################################

def calc_metrics(ranking_size, methods, metric, dict_topk):
    

    dict_topk = {k: v for k, v in dict_topk.items() if k in methods}
    

    first_element = next(iter(dict_topk.items()))
    num_instancias = len(first_element[1])
    

    matrix_points, list_combinations_methods = mt.create_matrix_combination_methdos_by_metric(dict_topk, metric, num_instancias, methods)
    
    
    return matrix_points, list_combinations_methods


def draw_lamp(ranking_size, metric, methods, threshold_radius, selected_dataset,
              area_marking_automatic=True, sel_indices_manual=None, sel_area_manual=None, df_lamp_prev=None):
    
    util = Util(selected_dataset)
    dict_topk = util.get_dict_topk(ranking_size)
    
    matrix_points, list_combinations_methods = calc_metrics(ranking_size, methods, metric, dict_topk)
    

    df_ds = pd.DataFrame(matrix_points)
    n_ds = df_ds.shape[0]
    
    
    ################### LAMP calc
    # creating the control points
    sample_size = matrix_points.shape[1] 
    table = list(product([0, 1], repeat=sample_size))
    ctp_samples = []
    for i in table:
        ctp_samples.append(list(i))
        
    #
    ctp_samples = np.asarray(ctp_samples)
    
    #
    control_points_samples = np.array(ctp_samples) #
    df_ctp_5d = pd.DataFrame(control_points_samples)
    n_5d = df_ctp_5d.shape[0]
    cp_positions__ = util.control_points_position(control_points_samples)
    df_ctp_2d = pd.DataFrame(cp_positions__)
    df_ds_cct = pd.concat([df_ds, df_ctp_5d], ignore_index=True)
    ids = np.arange(n_ds,n_ds+n_5d)
    df_ctp_2d[2] = ids
    ctp_2d = df_ctp_2d.values
    data = df_ds_cct.values
    lamp_proj = Lamp(Xdata = data, control_points = ctp_2d, label=False)
    data_proj = lamp_proj.fit()
    cp_positions = np.asarray(cp_positions__)
    #                                                                                          
    
    #build lamp plot
    size_dataset = len(matrix_points)
    df = pd.DataFrame(data_proj[:size_dataset,:], columns=['Comp_1','Comp_2'])
    df['Class'] = 'Original'
    
    if area_marking_automatic:
        df['Area'] = util.get_areas_by_distance(data_proj, size_dataset, point_disagreement=1, radius=threshold_radius)
    else:
        if df_lamp_prev is None:
            df['Area'] = "Neutral Area"
        else:
            df_lamp_prev = df_lamp_prev.iloc[:len(df),:]
            df['Area'] = df_lamp_prev['Area'].to_numpy()
            
        if sel_indices_manual is not None and sel_area_manual is not None:
            df.loc[sel_indices_manual,'Area'] = sel_area_manual          
            
            
    
    df2 = pd.DataFrame(cp_positions__, columns=['Comp_1','Comp_2'])
    df2['Class'] = 'Control Point'
    df2['Area'] = 'Control Point'
    
    #
    #
    df2 = df2.iloc[[0, len(df2)-1]] 
    
    df = pd.concat([df, df2])
    df = df.reset_index()
    df['Index'] = df.index    
    
    #
    c = Counter(zip(df.Comp_1,df.Comp_2))
    #
    s = [10*c[(xx,yy)] if 10*c[(xx,yy)]<=50 else 80+0.001*c[(xx,yy)] for xx,yy in zip(df.Comp_1,df.Comp_2)]
    #
    df['Size'] = s
    symbols = ['circle', 'circle', 'circle', 'x']
    
    
    fig = px.scatter(df, x="Comp_1", y="Comp_2", color="Area",
                     opacity=0.7, size="Size", symbol="Area", symbol_sequence = symbols,
                     hover_data={
                         'Area':True,
                         'Index': False,
                         'Comp_1':False,
                         'Comp_2':False,
                         'Size':False,
                     },
                    color_discrete_sequence=["#cc79a7", "#0072b2", "#009e73", "#f0e442"],
                    category_orders={ # replaces default order by column name
                    "Area": ["Neutral Area", "Disagreement Area", "Agreement Area", "Control Point"]}
                    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_traces(marker=dict(#size=10,
        line=dict(width=0.5, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    
    fig.update_yaxes(title='y', visible=False, showticklabels=False)
    fig.update_xaxes(title='x', visible=False, showticklabels=False)
    fig.update_yaxes(range = [-0.1,1.1])
    fig.update_xaxes(range = [-0.1,1.1])
    
    return fig, df, matrix_points, list_combinations_methods

def draw_umap(df_lamp, projector, selected_dataset, selectedpoints=None):
    if projector=='UMAP':
        X_test_umap = umap_datasets[selected_dataset]
        df_umap = pd.DataFrame(X_test_umap, columns=["Comp_1", "Comp_2"])
    else:
        X_test_tsne = tsne_datasets[selected_dataset]
        df_umap = pd.DataFrame(X_test_tsne, columns=["Comp_1", "Comp_2"])
    
    
    df_lamp = df_lamp.iloc[:len(df_umap),:]

    
    df_umap['Area'] = df_lamp['Area'].to_numpy()
    df_umap['Index'] = df_lamp['Index'].to_numpy()
    
    if selectedpoints is None:
        fig = px.scatter(df_umap, x="Comp_1", y="Comp_2",#width=600, height=400,
                         opacity=0.7,
                         color="Area",
                         hover_data={
                             'Area':True,
                             'Index': True,
                             'Comp_1':False,
                             'Comp_2':False,
                         },
                         color_discrete_sequence=["#cc79a7", "#0072b2", "#009e73", "#f0e442"],
                         category_orders={ # replaces default order by column name
                             "Area": ["Neutral Area", "Disagreement Area", "Agreement Area"]
                         }
                    )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        fig.update_traces(marker=dict(size=8,
                              line=dict(width=0.5,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        
        fig.update_yaxes(title='y', visible=False, showticklabels=False)
        fig.update_xaxes(title='x', visible=False, showticklabels=False)
    else:
        df_selectedpoints = df_umap.iloc[selectedpoints]
        df_umap = df_umap.loc[df_umap.index.difference(selectedpoints), :]
        #
        fig = go.Figure(
            layout = {
                'xaxis': {'title': 'x-label',
                        'visible': False,
                        'showticklabels': False},
                'yaxis': {'title': 'y-label',
                        'visible': False,
                        'showticklabels': False}
              }
        )
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=df_umap['Comp_1'],
                y=df_umap['Comp_2'],
                text = df_umap['Index'],
                hovertemplate = "%{text}",
                opacity=0.5,
                showlegend=False,
                name='Not selected',
                marker=dict(size=8,
                              line=dict(width=0.5,
                                        color='DarkSlateGrey'))
            )
        )
        # 
        # 
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=df_selectedpoints['Comp_1'],
                y=df_selectedpoints['Comp_2'],
                text = df_selectedpoints['Index'],
                hovertemplate = "%{text}",
                opacity=1.0,
                showlegend=False,
                name='Selected',
                marker=dict(size=8,color='#0d2a68',
                              line=dict(width=0.5,
                                        color='DarkSlateGrey')),
            )
        )
        #
        fig.update_layout(
            autosize=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        
    return fig


def draw_boxplot_measures(df_lamp, methods, selected_dataset):
    
    def set_area(row, agree_list, disagree_list, neutral_list):
        if row['index'] in agree_list:
            return 'Agreement Area'
        if row['index'] in disagree_list:
            return 'Disagreement Area'
        return 'Neutral Area'

    size = size_datasets[selected_dataset]
    df_lamp = df_lamp.iloc[:size,:]
    agreement_points = df_lamp[df_lamp['Area'] == 'Agreement Area']['Index'].to_list()
    disagreement_points = df_lamp[df_lamp['Area'] == 'Disagreement Area']['Index'].to_list()
    fuzzy_points = df_lamp[df_lamp['Area'] == 'Neutral Area']['Index'].to_list()
    
    feature_importance = FeatureImportance(selected_dataset)
    
    
    df_sens = feature_importance.get_measures_for_explanations(methods, 'sensitivity')
    df_inf = feature_importance.get_measures_for_explanations(methods, 'infidelity')
    
    df_sens['Area'] = df_sens.apply(lambda x: set_area(x, agreement_points, disagreement_points, fuzzy_points), axis=1)
    df_inf['Area'] = df_inf.apply(lambda x: set_area(x, agreement_points, disagreement_points, fuzzy_points), axis=1)

    fig_sens = px.box(df_sens, x="method", y="sensitivity", color="Area",
                      color_discrete_map={"Neutral Area": "#cc79a7", 
                                          "Disagreement Area": "#0072b2", 
                                          "Agreement Area": "#009e73"},
                      category_orders={ # replaces default order by column name
                             "Area": ["Neutral Area", "Disagreement Area", "Agreement Area"]
                         }
                     )
    fig_sens.update_traces(quartilemethod="exclusive") # 
    
    fig_inf = px.box(df_inf, x="method", y="infidelity", color="Area",
                     color_discrete_map={"Neutral Area": "#cc79a7", 
                                          "Disagreement Area": "#0072b2", 
                                          "Agreement Area": "#009e73"},
                     category_orders={ # replaces default order by column name
                             "Area": ["Neutral Area", "Disagreement Area", "Agreement Area"]
                         }
                    )
    fig_inf.update_traces(quartilemethod="exclusive") #
    
    return fig_inf, fig_sens


def draw_heatmaps(df_lamp, matrix_points, list_combinations_methods, selected_dataset):
    size = size_datasets[selected_dataset]
    df_lamp = df_lamp.iloc[:size,:]
    agreement_points = df_lamp[df_lamp['Area'] == 'Agreement Area']['Index'].to_list()
    disagreement_points = df_lamp[df_lamp['Area'] == 'Disagreement Area']['Index'].to_list()
    fuzzy_points = df_lamp[df_lamp['Area'] == 'Neutral Area']['Index'].to_list()
    
    df = pd.DataFrame(matrix_points, columns=list_combinations_methods)
    
    agreement_fig = go.Figure()
    fuzzy_fig = go.Figure()
    disagreement_fig = go.Figure()
    
    if len(agreement_points)!= 0:
        agreement_fig = px.imshow(df.loc[df.index[agreement_points]],text_auto=False,
                                  color_continuous_scale='Blues',
                                  zmin=0, zmax=1, labels=dict(color="Agreement Level", y="Instances", x="Pairs of explanation methods"),
                   )
        agreement_fig.update_yaxes(showticklabels=False)
    
    if len(fuzzy_points)!= 0:
        fuzzy_fig = px.imshow(df.loc[df.index[fuzzy_points]],text_auto=False,
                              color_continuous_scale='Blues',
                              zmin=0, zmax=1, labels=dict(color="Agreement Level", y="Instances", x="Pairs of explanation methods"),
                   )
        fuzzy_fig.update_yaxes(showticklabels=False)
        
    if len(disagreement_points)!= 0:
        disagreement_fig = px.imshow(df.loc[df.index[disagreement_points]],text_auto=False,
                                     color_continuous_scale='Blues',
                                     zmin=0, zmax=1, labels=dict(color="Agreement Level", y="Instances", x="Pairs of explanation methods"),
                   )
        disagreement_fig.update_yaxes(showticklabels=False)
        
    
    return agreement_fig, fuzzy_fig, disagreement_fig



def build_matrix_ranking(dataset, top_ranking, metric, selected_points=None):
    
    dict_matrix = {}
    for method_key in top_ranking.keys():
        df = pd.DataFrame(columns=dataset.get_feature_names())
        for index_instance, rank_features in top_ranking[method_key].items():
            position = 1
            dict_row = {}
            for k,v in rank_features.items():
                if metric=='rank agreement':
                    dict_row[k] = position
                    position += 1
                else:
                    if metric=='feature agreement':
                        dict_row[k] = position
                    else:
                        if metric=='sign agreement':
                            dict_row[k] = position if v >= 0 else position*-1
                        else:
                            #signed rank agreement
                            dict_row[k] = position if v >= 0 else position*-1
                            position += 1
                            
            new_row = pd.DataFrame(dict_row, index=[0])
            df = pd.concat([df, new_row]).reset_index(drop=True)
        df = df.fillna(0)
        if selected_points is not None:
            df = df.iloc[selected_points]

        dict_matrix[method_key] = df.to_numpy()
    return dict_matrix

def compare_ranking_between_methods(matrices, dict_topk):
    
    def compare_two_methods(matrix_1, matrix_2):
        comparison = (matrix_1 == matrix_2) * 1
        non_zeros = np.count_nonzero(comparison, axis=0)
        resul = comparison.shape[0] - non_zeros
        return resul
    
    vectors = {}
    for elem in list(combinations(list(dict_topk.keys()), 2)):
        vector = compare_two_methods(matrices[elem[0]], matrices[elem[1]])
        vectors[str(elem)] = vector

    return vectors

def draw_features_disagreement(methods, selected_dataset, ranking_size, metric, dis_points=None):
    
    if len(dis_points)==0:
        return go.Figure()
    
    util = Util(selected_dataset)
    dict_topk = util.get_dict_topk(ranking_size)
    dict_topk = {k: v for k, v in dict_topk.items() if k in methods}
    dataset = datasets_list[selected_dataset]
    
    matrices = build_matrix_ranking(dataset, dict_topk, metric, dis_points)
    dict_comparations = compare_ranking_between_methods(matrices, dict_topk)
    
    df = pd.DataFrame(columns=dataset.get_feature_names(), index=list(dict_comparations.keys()))
    for combination_of_methods, v in dict_comparations.items():
        df.loc[combination_of_methods] = v
    
    if dis_points is None:
        number_instances = len(dict_topk[list(dict_topk.keys())[0]])
    else:
        number_instances = len(dis_points)
        
    df = df.div(number_instances)
    
    df['Combinations'] = df.index
    
    df_long=pd.melt(df , id_vars=['Combinations'])
    df_long.columns = ['Combinations', 'Features', '% disagreements']
    
    fig = go.Figure()
    for feature in df_long['Features'].value_counts().index.to_list():
        df_one_feature = df_long[df_long['Features'] == feature]
        fig.add_trace(go.Bar(
            y= df_one_feature['Combinations'].values,
            x= df_one_feature['% disagreements'].values,
            name= str(feature),
            orientation='h',
        ))
    
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        xaxis_title="Proportion of disagreements",
        yaxis_title="Pairs of explanation methods",
        legend_title="Features",
    )
    
    return fig

def create_confucion_matrix(df):
    df = df.pivot_table(index='prediction', columns='real_class', values='confusion_matrix', aggfunc='count')
    if 0 not in df.columns.tolist():
        df[0] = np.NaN
    if 1 not in df.columns.tolist():
        df[1] = np.NaN

    if 0 not in df.index.values.tolist():
        df.loc[0] = [np.NaN, np.NaN]

    if 1 not in df.index.values.tolist():
        df.loc[1] = [np.NaN, np.NaN]

    df = df[[0,1]]
    df = df.sort_index()
    df = df.fillna(0)
    return df


def draw_confusion_matrix(df_predictions, selectedData):
    
    df_predictions['confusion_matrix'] = list(df_predictions['prediction'] - df_predictions['real_class'] * 2)
    df = df_predictions.iloc[selectedData]
    df_cm = create_confucion_matrix(df)
    fig = go.Figure()
    if len(selectedData)!= 0:
        fig = px.imshow(df_cm.to_numpy(),
                    labels=dict(x="The ground truth label", y="The predicted labeliction"),
                    x=['0', '1'],
                    y=['0', '1'],
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues'
                   )
        fig.update_traces(textfont=dict(size=42))
        fig.update_layout(font=dict(size=18))
        
    return fig

def draw_table_model_metrics(df_predictions, selected_data):
    
    if len(selected_data)== 0:
        return go.Figure()
    
    df_pred = df_predictions.iloc[selected_data]
    y_pred = df_pred['prediction'].to_numpy()
    y_true = df_pred['real_class'].to_numpy()
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    qnt_0 = 0
    qnt_1 = 0
    
    if 0 in df_pred['real_class'].value_counts().index.to_list():
        qnt_0 = df_pred['real_class'].value_counts()[0]
    if 1 in df_pred['real_class'].value_counts().index.to_list():
        qnt_1 = df_pred['real_class'].value_counts()[1]
    
    d = {'#instances': [df_pred.shape[0]],
         'accuracy_score': [acc],
         'balanced_accuracy_score': [bal_acc],
         'f1_score': [f1],
         'precision_score': [prec],
         'recall_score': [recall],
         '#class_0': [qnt_0],
         '#class_1': [qnt_1]}
    df = pd.DataFrame(data=d)
    
    df = df.T
    df['Metric'] = df.index
    df.columns = ['Value', 'Metric']
    df = df[['Metric', 'Value']]
    df['Value'] = df['Value'].round(decimals = 3)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='#add8e6',
                    align='left'),
        cells=dict(values=[df.Metric, df.Value],
                   fill_color='#e4f2f7',
                   align='left'))
    ])
    
    return fig
    


#####################################################################################################################
##########################                   Callbacks
#####################################################################################################################
        

@app.callback(
    Output(lamp_graph, component_property='figure'),
    Output(umap_graph, component_property='figure'),
    Output('lamp-value', 'data'),
    Output('matrix-points-value', 'data'),
    Output('list-methods-value', 'data'),
    Output("explanation_methods_dropdown", "options"),
    Output("explanation_methods_dropdown", "value"),
    Output("methods_warning", "children"),
    Output('btn-disagrement', 'disabled'),
    Output('btn-agreement', 'disabled'),
    Output('btn-clear', 'disabled'),
    Input(lamp_graph, 'selectedData'),
    Input('lamp-value', 'data'),
    Input('matrix-points-value', 'data'),
    Input('list-methods-value', 'data'),
    Input(lamp_graph, 'figure'),
    Input(umap_graph, 'figure'),
    State('metric_dropdown', 'value'),
    State('ranking_size_dropdown', 'value'),
    State('explanation_methods_dropdown', 'value'),
    State('input_threshold_radio', 'value'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks'),
    Input('projector_dropdown', 'value'),
    State('calc_areas_dropdown', 'value'),
    Input('btn-disagrement', 'n_clicks'),##########
    Input('btn-agreement', 'n_clicks'),#######
    Input('btn-clear', 'n_clicks'),######
)
def plot_embeddings(selected_data, lamp_value, matrix_points_value, list_methods_value,
                    current_lamp_fig, current_emb_fig,
                    selected_metric, selected_ranking_size, selected_explanation_methods,
                    threshold_radius, selected_dataset,
                    button_explanations, selected_projector,
                    sel_area_marking, button_disagreement, button_agreement, button_clear):
    
    
    all_explanation_methods_options=[{'label': x, 'value': x} for x in sorted(explanation_methods_names)]
    input_warning = None
    methods = selected_explanation_methods
    
    triggered_id = ctx.triggered_id
    
    if sel_area_marking=='Manual':
        btn_disagree = False
        btn_agree = False
        btn_clear = False
        area_marking_automatic = False
    else:
        btn_disagree = True
        btn_agree = True
        btn_clear = True
        area_marking_automatic = True
        
    
    if triggered_id == 'projector_dropdown':
        dataset = json.loads(lamp_value)
        lamp_df = pd.DataFrame.from_dict(dataset)
        lamp_df = lamp_df.T
        html_umap_graph = draw_umap(lamp_df, selected_projector, selected_dataset)
        return current_lamp_fig, html_umap_graph, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
        
    if triggered_id == 'btn-clear':
        methods = selected_explanation_methods
        metric = options[selected_metric]
        html_lamp_graph, df_lamp, matrix_points, list_combinations_methods = draw_lamp(selected_ranking_size, metric, methods, threshold_radius, selected_dataset, area_marking_automatic)
        html_umap_graph = draw_umap(df_lamp, selected_projector, selected_dataset)
        df_lamp_json = df_lamp.to_json(date_format='iso', orient='index')
        return html_lamp_graph, html_umap_graph, df_lamp_json, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
    
    
    # caso 1: Primeira execução do app
    if (lamp_value is None) or (triggered_id == 'button-explanations'):
        if triggered_id == 'button-explanations':
            if len(selected_explanation_methods) != 4:
                input_warning = dbc.Alert("You need to choose 4 methods.", color="warning")
                methods = selected_explanation_methods
                return go.Figure(), go.Figure(), lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
            else:
                methods = selected_explanation_methods
                metric = options[selected_metric]
                html_lamp_graph, df_lamp, matrix_points, list_combinations_methods = draw_lamp(selected_ranking_size, metric, methods, threshold_radius, selected_dataset, area_marking_automatic)
                html_umap_graph = draw_umap(df_lamp, selected_projector, selected_dataset)
        else:
            methods = selected_explanation_methods
            metric = options[selected_metric]
            html_lamp_graph, df_lamp, matrix_points, list_combinations_methods = draw_lamp(selected_ranking_size, metric, methods, threshold_radius, selected_dataset, area_marking_automatic)
            html_umap_graph = draw_umap(df_lamp, selected_projector, selected_dataset)
    else:
        dataset = json.loads(lamp_value)
        lamp_df = pd.DataFrame.from_dict(dataset)
        lamp_df = lamp_df.T
        if selected_data == None:
            if triggered_id == 'btn-disagrement' or triggered_id == 'btn-agreement':
                return current_lamp_fig, current_emb_fig, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
            else:
                html_umap_graph = draw_umap(lamp_df, selected_projector, selected_dataset)
                return current_lamp_fig, html_umap_graph, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
        else:
            pt = pd.DataFrame(selected_data["points"])
            if pt.empty:
                return current_lamp_fig, current_emb_fig, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
            else:
                size_dataset = size_datasets[selected_dataset] #X_test.shape[0]
                indices = []
                for sel in pt['customdata']:
                    if sel[1] < size_dataset:
                        indices.append(sel[1])
                if len(indices) == 0:
                    return current_lamp_fig, current_emb_fig, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
                else:
                    if triggered_id == 'btn-disagrement' or triggered_id == 'btn-agreement':
                        if triggered_id == 'btn-disagrement':
                            area = 'Disagreement Area'
                        if triggered_id == 'btn-agreement':
                            area = 'Agreement Area'
                        
                        dataset = json.loads(lamp_value)
                        lamp_df = pd.DataFrame.from_dict(dataset)
                        lamp_df = lamp_df.T
                        methods = selected_explanation_methods
                        metric = options[selected_metric]
                        html_lamp_graph, df_lamp, matrix_points, list_combinations_methods = draw_lamp(selected_ranking_size, metric, methods, threshold_radius, selected_dataset, area_marking_automatic, sel_indices_manual=indices, sel_area_manual=area, df_lamp_prev=lamp_df)
                        html_umap_graph = draw_umap(df_lamp, selected_projector, selected_dataset)
                    else:
                        html_umap_graph = draw_umap(lamp_df, selected_projector, selected_dataset, indices)
                        return current_lamp_fig, html_umap_graph, lamp_value, matrix_points_value, list_methods_value, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear
    
    df_lamp_json = df_lamp.to_json(date_format='iso', orient='index')
    return html_lamp_graph, html_umap_graph, df_lamp_json, matrix_points, list_combinations_methods, all_explanation_methods_options, methods, input_warning, btn_disagree, btn_agree, btn_clear


@app.callback(
    Output(graph_measures_infidelity, component_property='figure'),
    Output(graph_measures_sensitivity, component_property='figure'),
    Input('lamp-value', 'data'),
    State('explanation_methods_dropdown', 'value'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks')
)
def plot_measures_charts(lamp_value, list_methods, selected_dataset, button_explanations):
    
    lamp_dataset = json.loads(lamp_value)
    lamp_df = pd.DataFrame.from_dict(lamp_dataset)
    lamp_df = lamp_df.T
    
    if len(list_methods) != 4:
        return go.Figure(), go.Figure()
    
    fig_inf, fig_sens = draw_boxplot_measures(lamp_df, list_methods, selected_dataset)
    return fig_inf, fig_sens


@app.callback(
    Output(heatmap_agreement, component_property='figure'),
    Output(heatmap_fuzzy, component_property='figure'),
    Output(heatmap_disagreement, component_property='figure'),
    Input('lamp-value', 'data'),
    State('matrix-points-value', 'data'),
    State('list-methods-value', 'data'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks')
)
def plot_heatmaps_charts(lamp_value, matrix_points_value, list_methods_value, selected_dataset,
                           button_explanations):
    
    lamp_dataset = json.loads(lamp_value)
    lamp_df = pd.DataFrame.from_dict(lamp_dataset)
    lamp_df = lamp_df.T    
    fig_agreement, fig_fuzzy, fig_disagreement = draw_heatmaps(lamp_df, matrix_points_value, list_methods_value, selected_dataset)
    return fig_agreement, fig_fuzzy, fig_disagreement

@app.callback(
    Output(confusion_matrix_agree, component_property='figure'),
    Output(confusion_matrix_fuzzy, component_property='figure'),
    Output(confusion_matrix_disagree, component_property='figure'),
    Input('lamp-value', 'data'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks')
)
def plot_confusion_matrix(lamp_value, selected_dataset, button_explanations):

    lamp_dataset = json.loads(lamp_value)
    lamp_df = pd.DataFrame.from_dict(lamp_dataset)
    lamp_df = lamp_df.T    

    
    agreement_points = lamp_df[lamp_df['Area'] == 'Agreement Area']['Index'].to_list()
    disagreement_points = lamp_df[lamp_df['Area'] == 'Disagreement Area']['Index'].to_list()
    fuzzy_points = lamp_df[lamp_df['Area'] == 'Neutral Area']['Index'].to_list()
    
    
    dataset = datasets_list[selected_dataset]
    df = dataset.get_predictions()
    
    fig_agree = draw_confusion_matrix(df, agreement_points)
    fig_fuzzy = draw_confusion_matrix(df, fuzzy_points)
    fig_disagree = draw_confusion_matrix(df, disagreement_points)
    
    return fig_agree, fig_fuzzy,fig_disagree


@app.callback(
    Output(model_metrics_agree, component_property='figure'),
    Output(model_metrics_fuzzy, component_property='figure'),
    Output(model_metrics_disagree, component_property='figure'),
    Input('lamp-value', 'data'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks')
)
def plot_model_metrics(lamp_value, selected_dataset, button_explanations):
    
    lamp_dataset = json.loads(lamp_value)
    lamp_df = pd.DataFrame.from_dict(lamp_dataset)
    lamp_df = lamp_df.T    
    
    agreement_points = lamp_df[lamp_df['Area'] == 'Agreement Area']['Index'].to_list()
    disagreement_points = lamp_df[lamp_df['Area'] == 'Disagreement Area']['Index'].to_list()
    fuzzy_points = lamp_df[lamp_df['Area'] == 'Neutral Area']['Index'].to_list()
    
    dataset = datasets_list[selected_dataset]
    df = dataset.get_predictions()
    
    fig_agree = draw_table_model_metrics(df, agreement_points)
    fig_fuzzy = draw_table_model_metrics(df, fuzzy_points)
    fig_disagree = draw_table_model_metrics(df, disagreement_points)
    
    return fig_agree, fig_fuzzy, fig_disagree
    

@app.callback(
    Output(features_graph_agree, component_property='figure'),
    Output(features_graph_fuzzy, component_property='figure'),
    Output(features_graph_disagree, component_property='figure'),
    Input('lamp-value', 'data'),
    State('dataset_dropdown', 'value'),
    Input('button-explanations', 'n_clicks'),
    State('metric_dropdown', 'value'),
    State('ranking_size_dropdown', 'value'),
    State('explanation_methods_dropdown', 'value'),
)
def plot_features_disagreement(lamp_value, selected_dataset, button_explanations,
                               selected_metric, selected_ranking_size, methods):
    
    lamp_dataset = json.loads(lamp_value)
    lamp_df = pd.DataFrame.from_dict(lamp_dataset)
    lamp_df = lamp_df.T    
    
    agreement_points = lamp_df[lamp_df['Area'] == 'Agreement Area']['Index'].to_list()
    disagreement_points = lamp_df[lamp_df['Area'] == 'Disagreement Area']['Index'].to_list()
    fuzzy_points = lamp_df[lamp_df['Area'] == 'Neutral Area']['Index'].to_list()    
    
    fig_agree = draw_features_disagreement(methods, selected_dataset, selected_ranking_size, selected_metric, agreement_points)
    fig_fuzzy = draw_features_disagreement(methods, selected_dataset, selected_ranking_size, selected_metric, fuzzy_points)
    fig_disagree = draw_features_disagreement(methods, selected_dataset, selected_ranking_size, selected_metric, disagreement_points)
    
    return fig_agree, fig_fuzzy,fig_disagree
            
#############################################################

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)