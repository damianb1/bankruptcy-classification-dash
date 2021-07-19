
import dash_table

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State, MATCH, ALL

import plotly.express as px
import plotly.graph_objects as go



import dash_bootstrap_components as dbc




from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
precision_score, recall_score, jaccard_score, roc_auc_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import plotly.figure_factory as ff



from app import app
 






scoring_parameters=[
    'accuracy', 'balanced_accuracy','average_precision','f1',
    'precision','recall','jaccard','roc_auc'
    ]

names = ["Nearest Neighbors", # "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "Quadratic Discriminant Analysis"]

def get_hp_controls(method):
    hp_header = [html.H5("Hyperparameters")]
    controls=[]
    if method in [None, "AdaBoost",
         "Naive Bayes", "Quadratic Discriminant Analysis","Gaussian Process"] :
         return ""
    elif method=="Nearest Neighbors":
        controls = [
        
            dbc.Label("Number of neighbors"),
            dbc.Input(id={'type':'hyperparameter','id':'n-neighbors'},
             type="number", min=1,max=20,step=1, value=1)
        ]
   
    elif method=="Decision Tree":
        controls = [
            dbc.Label("Max depth (leave empty for no limit)"),
            dbc.Input(id={'type':'hyperparameter','id':'tree-max-depth'},
             type="number", min=0, step=1)
        ]
    elif method=="Random Forest":
        controls = [
            dbc.Label("Max depth (leave empty for no limit)"),
            dbc.Input(id={'type':'hyperparameter','id':'forest-max-depth'},
             type="number", min=0, step=1),
            dbc.Label("n estimators"),
            dbc.Input(id={'type':'hyperparameter','id':'forest-n-estimators'},
             type="number", min=0, step=1, value=100)
        ]
    elif method=="Neural Net":
        controls = [
            dbc.Label("alpha"),
            dbc.Input(id={'type':'hyperparameter','id':'net-alpha'},
             type="number", min=0, value=1, step=0.001),
            dbc.Label("max iterations"),
            dbc.Input(id={'type':'hyperparameter','id':'net-max-iter'},
             type="number", min=0, step=1, value=10)

        ]
    
    return hp_header+controls


def get_classifier(method_name, **hparams):
    if method_name==None:
        return None 
    elif method_name=="Nearest Neighbors":
        clf = KNeighborsClassifier(hparams['n-neighbors'])
    elif method_name=="Decision Tree":
        clf = DecisionTreeClassifier(max_depth=hparams['tree-max-depth'])
    elif method_name=="Random Forest":
        clf = RandomForestClassifier(
            max_depth=hparams['forest-max-depth'], n_estimators=hparams['forest-n-estimators'])
    elif method_name=="Neural Net":
        clf = MLPClassifier(alpha=hparams['net-alpha'], max_iter=hparams['net-max-iter'])
    elif method_name=="AdaBoost":
        clf = AdaBoostClassifier()
    elif method_name=="Naive Bayes":
        clf = GaussianNB()
    elif method_name=="Quadratic Discriminant Analysis":
        clf = QuadraticDiscriminantAnalysis()
    
    return clf


def get_cv_graph(n_splits, split_dataset):
    y_train= pd.read_json(split_dataset[2], typ='series', orient='split')
       

    fig = go.Figure()
    #config = {'staticPlot': True}
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    split=1
    kf = KFold(n_splits=n_splits)
    for train, test in kf.split(y_train):
        showlegend=True
        if split != 1:
            showlegend=False
        fig.add_trace(go.Scatter(
            name='training',x=np.array([0,len(y_train)-1]), y=['split'+str(split)]*2,
                            mode='lines',line=dict(color='LightSeaGreen', width=20),
                            showlegend=showlegend))
        fig.add_trace(go.Scatter(name='test',x=test[::50], y=['split'+str(split)]*len(test[::50]),
                                mode='lines',line=dict(color='orange', width=20),
                                
                                showlegend=showlegend))
        split+=1
    fig.update_layout(transition_duration=1000)
    fig.update_layout(title="Training set validation splits")
    return fig

def get_cv_output(clf, n_splits, scoring, split_dataset):
    
    X_train=pd.read_json(split_dataset[0], orient='split')
    y_train= pd.read_json(split_dataset[2], typ='series', orient='split')
    

    scores = cross_val_score(clf, X_train, y_train, cv=n_splits, scoring=scoring)
    cv_output = []
    scores_str="Scores: "
    for score in scores:
        scores_str+=str(round(score, 3))+", "
    cv_output.append(html.P(scores_str))
    cv_output.append(html.P(
        "Average: "+str(round(scores.mean(), 3))
        +", Standard deviation: "+str(round(scores.std(), 3))
        ))
    
    return cv_output

def get_trained_model(clf, saved_data): # unused for now
    X_train=pd.read_json(saved_data[0], orient='split')
    y_train= pd.read_json(saved_data[2], typ='series', orient='split')
    clf.fit(X_train, y_train)
    
    return clf

def get_training_results(clf, saved_data):
    # returns trained model results as layout elements
    
    X_train=pd.read_json(saved_data[0], orient='split')
    X_test=pd.read_json(saved_data[1], orient='split')
    y_train= pd.read_json(saved_data[2], typ='series', orient='split')
    y_test= pd.read_json(saved_data[3], typ='series', orient='split')

    clf.fit(X_train, y_train)
    y_pred_train, y_pred_test = clf.predict(X_train), clf.predict(X_test)

    train_scores, test_scores, metric_names=[],[],[]
    clf_metrics ={
        'accuracy' : accuracy_score,
        'balanced_accuracy' : balanced_accuracy_score,
        'f1' : f1_score,
        'precision' : precision_score,
        'recall' : recall_score,
        'jaccard' : jaccard_score,
        'roc_auc_score' : roc_auc_score
    }
    for m in clf_metrics:
        train_scores.append(clf_metrics[m](y_train, y_pred_train))
        test_scores.append(clf_metrics[m](y_test, y_pred_test))
        metric_names.append(m)
    clf_scores = pd.DataFrame(
        {
            "name" : metric_names,
            "train set score" : train_scores,
            "test set score" : test_scores 
        }
    )

    table=dash_table.DataTable(
        
        columns=[{"name": i, "id": i, "type":"numeric", "format" : dict(specifier='.4~f')} 
                    for i in clf_scores.columns], 
        style_table={'overflowX': 'auto','width':500, 'max-height':500},
        data=clf_scores.to_dict('records'),
        editable=False
        )

    cm=confusion_matrix(y_test, y_pred_test)
    x = ['not bankrupt', 'bankrupt']
    y =  ['not bankrupt', 'bankrupt']
    cm_text = [[str(num) for num in vec] for vec in cm]
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text, colorscale='Viridis')
    fig.update_layout(
            #title_text='<b>Confusion matrix for test set</b>',
                  xaxis = dict(title='<b>Predicted</b>'),
                yaxis = dict(title='<b>True</b>')
                 )

    
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    try:
        rocauc=auc(fpr, tpr)
        rocauc=round(rocauc,4)
    except:
        rocauc=0
    roc_plot = px.area(
    x=fpr, y=tpr,
    title=f'<b>ROC Curve (AUC={rocauc if rocauc>0 else "?"}) (test)</b>',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
   # width=700, height=500
    )
    roc_plot.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    roc_plot.update_yaxes(scaleanchor="x", scaleratio=1)
    roc_plot.update_xaxes(constrain='domain')

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    
    try:
        pr_auc=auc(recall, precision)
        pr_auc=round(pr_auc,4)
    except:
        pr_auc=0
    precision_recall_plot = px.area(
        x=recall, y=precision,
        title=f'<b>Precision-Recall Curve (AUC={pr_auc if pr_auc>0 else "?"}) (test)</b>',
        labels=dict(x='Recall', y='Precision'),color_discrete_sequence=['orange','green'],
    #    width=700, height=500
    )
    precision_recall_plot.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    precision_recall_plot.update_yaxes(scaleanchor="x", scaleratio=1)
    precision_recall_plot.update_xaxes(constrain='domain')


    
    
    results_layout=[
        dbc.Row([
            dbc.Col([
                html.H5("Classification scores for various evaluation metrics"),
                html.Br(),html.Br(),
                table
            ]),
            dbc.Col([
                html.H5("Confusion matrix for the test set"),
                dcc.Graph(figure=fig)
                ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=roc_plot)
            ]),
            dbc.Col([
                dcc.Graph(figure=precision_recall_plot)
            ])
        ])
    ]
    return results_layout



method_card = dbc.Card([
        dbc.CardHeader(html.H4("Method specification")),
        dbc.CardBody([
            dbc.Row([
            dbc.Col([
            dbc.Label("Select classifier"),
            dcc.Dropdown(id='clf-dropdown',
                options=[
                    {"label" : n, "value" : n} for n in names
                ]
            ),

            dcc.Markdown('''
            _To interrupt training or cross validation, clear classifier choice
            and press training/cv button again._
            ''')
        ]),
            dbc.Col(id='hp-col')
        ])])

    ])

cv_card = dbc.Card([
    dbc.CardHeader(html.H4("Cross validation")),
    dbc.CardBody([
        dbc.Label("Number of cv iterations:"),
        dcc.Slider(
            id='n-splits-slider', min=2,max=10,step=1, value=4,
            marks={2:'2',5:'5',10:'10'}
        ),
        
        
        dcc.Graph(id='cv-graph'),
        dbc.Label("Scoring parameter"),
        dcc.Dropdown(
            id='scoring-dropdown', options=[
                {"label" : p, "value" : p} for p in scoring_parameters
            ], value='f1'
                ),
        dbc.Button(id='cv-btn', children="Get validation scores"),
        dcc.Loading(html.Div(id='cv-output'))



    ])
    
])

training_card = dbc.Card([
    dbc.CardHeader(html.H4("Training")),
    dbc.CardBody([
        
        
        dcc.Markdown(id='p-dataset', children="**Dataset:**"),
        dcc.Markdown('**Number of features:** 64'),
        dcc.Markdown(id='p-scalers'),
        dcc.Markdown(id='p-samples'),
        dcc.Markdown(id='p-classifier'), 
        dbc.Row([
            dbc.Button(id='train-btn', children="Train",style={'display':'inline'}),
            html.Div([
                dcc.Loading(dbc.Alert(id='train-alert', style={'display':'none'}))
            ], style={'width':'300px', 'padding':'10px'})
            
        ])
    ])
])

model_card = dbc.Card([
    dbc.CardHeader(html.H4("Model evaluation")),
    dbc.CardBody([
            dbc.Row("asd")
    ], id='train-results')
], id='model-card', style={'display':'none'})



layout = dbc.Container(
    [
    dbc.Row(html.H2("Classification")),
    html.Hr(),
    dbc.Row([dbc.Col([method_card, training_card]),
    dbc.Col(cv_card)]),
    
    model_card,
    dcc.Store(id='trained-model', data=None)
    ])


@app.callback(
    Output('hp-col', 'children'),
    Input('clf-dropdown', 'value'),
    prevent_initial_call=True
   
)
def update_hp_col(method):
    return get_hp_controls(method)



@app.callback(
    Output('cv-graph', 'figure'),
    Input('n-splits-slider', 'value'),
    State('saved-data','data'),
    prevent_initial_call=True
)
def update_cv_graph(n_splits, saved_data):
    return get_cv_graph(n_splits, saved_data)

@app.callback(
    Output('cv-output', 'children'),
    Input('cv-btn', 'n_clicks'),
    State('clf-dropdown', 'value'),
    State({'type': 'hyperparameter', 'id': ALL}, 'value'),
    State({'type': 'hyperparameter', 'id': ALL}, 'id'),
    State('n-splits-slider', 'value'),
    State('scoring-dropdown', 'value'),
    State('saved-data', 'data'),
    prevent_initial_call=True
)
def update_cv_output(cv_clicks, method, hp_values,
                     hp_indexes, n_splits, scoring, saved_data):
    if method is None:
        #raise PreventUpdate
        return None
    hp_indexes = [hp['id'] for hp in hp_indexes]
    hyperparameters = { hp_id : hp_value for hp_id,hp_value in zip(hp_indexes, hp_values) }    
    clf = get_classifier(method, **hyperparameters)
    #print(hyperparameters)
    try:
        cv_output=get_cv_output(clf, n_splits, scoring, saved_data)
    except:
        return "error"
    return cv_output


@app.callback(
    Output('p-dataset', 'children'),
    Input('dataset-dropdown', 'value'),
)
def update_p_dataset(year):
    year=int(year[0]) # ( 1year -> 1)
    year_str = {
        1 : '1st',
        2 : '2nd',
        3 : '3rd',
        4 : '4th',
        5 : '5th'
    }
    dataset_str = f"""
        **Dataset:** Financial rates from {year_str[year]} year of the forecasting period
         and corresponding class label that indicates bankruptcy status
          after {6-year} {'year' if year==5 else 'years'}.
    """
    return dataset_str

@app.callback(
    Output('p-scalers', 'children'),
    Input('saved-data', 'data'),
    State('scalers-after-split', 'data'),
    prevent_initial_call=True
)
def update_p_scalers(saved_data, scalers):
    scalers_str = f"""
            **Standardization:** \{scalers}
    """
    return scalers_str

@app.callback(
    Output('p-samples', 'children'),
    Input('saved-data', 'data'),
    prevent_initial_call=True
)
def update_p_samples(saved_data):
    if saved_data is None:
        return None
    y_train = pd.read_json(saved_data[2], typ='series', orient='split')
    y_test = pd.read_json(saved_data[3], typ='series', orient='split')

    samples_str = f"""
            **Training samples:** {len(y_train)}

            **Test samples:** {len(y_test)}"""
    return samples_str

@app.callback(
    Output('p-classifier', 'children'),
    Input('clf-dropdown', 'value'),
)
def update_p_classifier(clf):
    clf_str=f"""
            **Selected classifier:** {clf}
            """
    return clf_str

@app.callback(
    #Output('trained-model', 'data'),
    Output('train-alert', 'children'),
    Output('train-alert', 'style'),
    Output('model-card', 'style'),
    Output('train-results', 'children'),
    Input('train-btn', 'n_clicks'),
    State('clf-dropdown', 'value'),
    State({'type': 'hyperparameter', 'id': ALL}, 'value'),
    State({'type': 'hyperparameter', 'id': ALL}, 'id'),
    State('saved-data', 'data'),
    prevent_inital_call=True
    
)
def update_trained_model(train_btn_clicks, method, hp_values, hp_indexes, saved_data):
    

    disp = {
        0:{'display':'none'}, 1:{'display':'inline', 'spacing':'10px'}
    }
    if train_btn_clicks is None:
        return None, disp[0], disp[0], None


    if method is None:
        alert_str = "Select classifier"
        return alert_str, disp[1], disp[0], None
    if saved_data is None: 
        return "data not found", disp[1], disp[0], None
    
    hp_indexes = [hp['id'] for hp in hp_indexes]
    hyperparameters = { hp_id : hp_value for hp_id,hp_value in zip(hp_indexes, hp_values) }    
    clf = get_classifier(method, **hyperparameters)
    try:
        train_results = get_training_results(clf, saved_data)
    except:
        return "Check parameters", disp[1], disp[0], None
    
    return None, disp[0], disp[1], train_results

    # add saving model
    