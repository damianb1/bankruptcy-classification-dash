import dash

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State

import plotly.express as px


import dash_bootstrap_components as dbc

import json

from sklearn import preprocessing

from dash.exceptions import PreventUpdate

from sklearn.model_selection import train_test_split

from app import app
 

with open('attr_info.json') as json_file:
    attr_info = json.load(json_file)
features = [k for k in attr_info]

scaler_names=[
      "StandardScaler",
        "MinMaxScaler" ,
        "MaxAbsScaler" ,
        "RobustScaler",
        "YeoJohnson",
        "QuantileTransformerNormal",
        "QuantileTransformerUniform" 
]

def get_scaler(method_name):
    scalers = {
        "StandardScaler" : preprocessing.StandardScaler(),
        "MinMaxScaler" : preprocessing.MinMaxScaler(),
        "MaxAbsScaler" : preprocessing.MaxAbsScaler(),
        "RobustScaler" : preprocessing.RobustScaler(),
        "YeoJohnson" : preprocessing.PowerTransformer(method='yeo-johnson'),
        "QuantileTransformerNormal" : preprocessing.QuantileTransformer(output_distribution='normal'),
        "QuantileTransformerUniform" : preprocessing.QuantileTransformer(output_distribution='uniform')
    }
    return scalers[method_name]

def get_transformed_data(current_data, method_name):
    scaler = get_scaler(method_name)
    current_data = pd.read_json(current_data, orient='split')
    current_X = current_data.drop("class",axis=1)
    scaled_X = scaler.fit_transform(current_X)
    current_X = pd.DataFrame(scaled_X, columns=current_X.columns)
    current_X['class'] = current_data['class']
    return current_X.to_json(orient='split')

def get_histogram(df, attr, is_base, n_bins):
    df = pd.read_json(df, orient='split')
    if not is_base:
        title = 'Current set'
    else:
        title = 'Base set'
    
    fig = px.histogram(x=df[attr], nbins=n_bins)
    fig.update_layout(
    #autosize=False,
    width=350,
    height=350,
    title_text=title,
    title_x=0.5,
    title_y=0.9)


        

    return fig

def get_split_dataset(current_data, test_size, random_state):
    current_data=pd.read_json(current_data, orient='split')


    current_X = current_data.drop('class', axis=1)
    current_y = current_data['class']
    X_train, X_test, y_train, y_test = train_test_split(
        current_X, current_y, test_size=test_size, random_state=random_state)
    
    return [X_train.to_json(orient='split'),
     X_test.to_json(orient='split'),
      y_train.to_json(orient='split'),
      y_test.to_json(orient='split')]

def get_split_graph(split_dataset):
    if split_dataset is None:
        """
        count_y=(pd.DataFrame(
            y.value_counts()).reset_index()
            .rename(columns={'index':'class','class':'count'})
        )
        count_y.loc[count_y['class'] == 0, 'class'] = 'not bankrupt' 
        count_y.loc[count_y['class'] == 1, 'class'] = 'bankrupt'
        fig1 = px.sunburst(count_y, path=['class'], values='count',color='class')
        return fig1
        """
        # handled in the callback
        pass

    else:
        
        y_train= pd.read_json(split_dataset[2], typ='series', orient='split')
        y_test = pd.read_json(split_dataset[3], typ='series', orient='split')
        count_df=(pd.concat([y_test.value_counts(), y_train.value_counts()])
            .reset_index().rename(columns={'index' : 'class', 'class':'count'}))
        count_df.loc[count_df['class'] == 0, 'class'] = 'not bankrupt' 
        count_df.loc[count_df['class'] == 1, 'class'] = 'bankrupt'
        count_df['set']=['test']*len(y_test.value_counts())+['train']*len(y_train.value_counts())
        # xd
        fig = px.sunburst(count_df, path=['set', 'class'], values='count',color='class',color_discrete_sequence=['red','blue','#00CC96'])
        fig.update_layout(transition_duration=2000)
        return fig
    











layout = dbc.Container(
    [
    dbc.Row(html.H2("Preprocessing")),
    html.Hr(),
    dbc.Card([
        dbc.CardHeader(html.H4("Feature selection")),
        dbc.CardBody([
            
            html.P("Choose input variables for further analysis:", className="card-subtitle"),
            dbc.Row(dcc.Dropdown(
                id='select-features-dropdown',
                options=[ {'label' : i, 'value' : i} for i in features ],
                value=[i for i in features],
                    
                
                multi=True)),
            dbc.Row([
                dbc.Button(id="select-features-btn", children="Select"),
                dcc.Loading(html.Div(id='loading-select', children='',
                style={'width':'100px'}),
                 type='circle')
                ]),
            dbc.Alert(id="features-alert", children="x out of 64 features selected", color="info"),
            dbc.Row(html.Div("Changing choice of features resets further steps.")),
            

        ])
            
       
    ],),

    dbc.Card([
        dbc.CardHeader(html.H4("Missing values")),
        dbc.CardBody([
            dcc.Loading(
                dbc.Alert(id='nan-alert',
                children="x empty (NaN/None) values found", color="warning"),
            ),
            dbc.Row("Imputation of missing values"),
            dbc.Row(dcc.RadioItems(
                id='fill-nan-radio',
                options=[
                    {'label': 'Zero', 'value': 'Zero'},
                    {'label': 'Mean', 'value': 'Mean'},
                    {'label': 'Median', 'value': 'Median'}
                ],
                value='Mean',
                labelStyle={'display': 'inline-block'}
            )  ),
            
            dbc.Row([
                dbc.Button(id='fill-nan-btn', children="Fill"),
                dcc.Loading(html.Div(id='loading-fill', children='',
                style={'width':'100px'}
                ),
                 type='circle')
            ])
            

        ])

    ]),

   

        dbc.Row([
            dbc.Col(
                [
                    dbc.Card([
            dbc.CardHeader(html.H4("Feature standardization")),
            dbc.CardBody([
            dbc.Row("Transformation method:"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='method-dropdown',
                    options=[
                        {"label" : m, "value" : m} for m in scaler_names
                    ],
                    
                )),
                dbc.Col(html.P(children="""

                """, id='scaling-desc'))
            ]),
            dbc.Row([
                dbc.Button(id='transform-btn', children="Fit and transform"),
                dcc.Loading(html.Div(id='loading-transform', children='',
                style={'width':'100px'}), type='circle'),
            ]),
            
            html.Div("""Transforms current dataframe.
                        In order to transform base dataframe, reset changes first 
                        (by selecting features or reloading dataset)."""),
            html.Div("""(Scaler should usually be fit on a training set. Here it is fit on
                        the whole dataset.)
            """),
            
        ])
        ]),
                ], width=5
            ),
            dbc.Col([
            dbc.Card([
            dbc.CardHeader(html.H4("Histograms")),
            dbc.CardBody([

                html.P("""Compare frequency distributions for chosen variable
                 before and after scaling."""),
                dbc.Form([
                   
                    dbc.FormGroup([
                        dbc.Col(dbc.Label("Select variable", className="mr-2")),
                        dbc.Col(
                            dcc.Dropdown(
                        id='graph-dropdown',
                        options=[
                            {"label" : c, "value" : c} for c in features
                        ], 
                         style={'width':'100px'}
                        ),
                        ),

                   
                    
                    
                        dbc.Col(dbc.Label("Number of bins", className="mr-2")),
                        dbc.Col([
                            dbc.Input(id='n-bins',
                            type='number',min=0,step=1,value=200,
                             style={'width':'75px'}),
                        dbc.FormText("0 = Auto"),
                        ]),
                  
                    
                   
                        dbc.Col(
                        dbc.Button(id='update-graphs-btn',children='Update'),
                        )
                        
                    ], className="mr-3", row=True), 
                ] ),

      
               
                dcc.Loading(id='loading-histogram'),
                dbc.Row([
                    
                        dcc.Graph(
                            id='base-histogram',style={'height':'300px','width':'300px'}
                            ),
                        dcc.Graph(
                            id='current-histogram',style={'height' : '300px', 'width':'300px'}
                            )
                    
                   
                   
                ])
                
            ])
        ])
            ], width=7)
        ]),
        
       
        
 
    
    dbc.Card([
        dbc.CardHeader(html.H4("Data split")),
        dbc.CardBody([
            
            dbc.Row([

            
            dbc.Col([
                html.P("Shuffle and divide data into training and test sets."),
                html.P("""Specify test set size; 
                float value, should be between 0.0 and 1.0 and represent
                 the proportion of the dataset to include in the test split.
                  If None, it will be set to 0.25."""),
                html.P("""
                    To get the same shuffling results across different calls,
                    specify a random state as an integer value. 
                    To produce different results with every split,
                    leave the field empty."""),
                
                dbc.Label('Test size'),
                dbc.Input(id='test-size', type='number', min=0.001, max=0.999, step=0.001, value=0.2),
                dbc.Label('Random state'),
                dbc.Input(id='random-state', type='number', min=0, step=1, value=42),
                dbc.Row([
dbc.Button(id='split-btn',children=['Split']),
                dcc.Loading(html.Div(id='loading-split', children='',
                style={'width':'100px'}), type='circle'),
                ]),
                
            ]),
            dbc.Col([
                html.P("""Chart shows sizes of sets and proportions of label values."""),
                dcc.Graph(id='split-graph')
            ])
            ]) # row
            


        ])
    ]),
    
    
    


            dbc.Card([
                dbc.CardHeader(html.H4("Save data")),
                dbc.CardBody([
    
                html.P("""Data saved for further analysis is the last train-test dataset split.
                After making any changes to the data, you need to split it again before saving."""),
                dbc.Row([
                dbc.Button(id="save-button", children="Save"),
                dcc.Loading(
                    html.Div([
                dbc.Alert(
                    id='save-alert', color='success',
                     style={'display':'none'}),], 
                     style={'width':'500px','padding':'10px'})
                     
                )
                ])
                
                ])
            ]),
    
   

    dcc.Store(id="current-data", data=None),
    dcc.Store(id="selected-data", data=None),
    dcc.Store(id="filled-data", data=None),
    dcc.Store(id="transformed-data", data=None),
    dcc.Store(id="scalers-before-split", data=None),
    dcc.Store(id="scalers-after-split", data=None),
    dcc.Store(id="split-dataset-list", data=None),
    dcc.Store(id="saved-data", data=None),
    
    
 


    ])


@app.callback(
    
    Output(component_id='features-alert',component_property='children'),
    Output('graph-dropdown', 'options'),
    Input(component_id='select-features-btn', component_property='n_clicks'),
    State(component_id='select-features-dropdown', component_property='value')
)
def update_features_number(n_clicks, selected_features):
    features_str =  (f"{len(selected_features)} out of {len(features)} features selected"
                    if len(selected_features)>0 else "Select at least one feature.")
    
   # if len(selected_features==0):
    #    features_str="Select at least one feature."
    graph_options = [{"label" : f, "value" : f} for f in selected_features ]

    return features_str, graph_options


@app.callback(
    Output('current-data','data'),
    Input('selected-data', 'data'),
    Input('filled-data', 'data'),
    Input('transformed-data', 'data'),
    prevent_initial_call=True

)
def update_current_X(selected_data, filled_data, transformed_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]  

    inputs = {
        'selected-data' : selected_data,
        'filled-data' : filled_data,
        'transformed-data' : transformed_data
    }
    if input_id in inputs:
        if inputs[input_id] is None:
            raise PreventUpdate
        return inputs[input_id]
    else:
        raise PreventUpdate

    

@app.callback(
    Output('selected-data', 'data'),
    Output('loading-select', 'children'),
    Input('select-features-btn', 'n_clicks'),
    Input('loaded-data', 'data'),
    State(component_id='select-features-dropdown', component_property='value'),
    State('loaded-data', 'data'),
    prevent_initial_call=True

)
def update_selected_X(select_clicks, ld_change, selected_features, ld_state):
    if len(selected_features)==0:
        raise PreventUpdate
    ld_state = pd.read_json(ld_state, orient='split')
    selected_data = ld_state[selected_features+['class']].to_json(orient='split') 
    
    return selected_data, ""


@app.callback(
    Output('filled-data', 'data'),
    Output('loading-fill', 'children'),
    Input('fill-nan-btn', 'n_clicks'),
    State('fill-nan-radio', 'value'),
    State('current-data', 'data'),
    prevent_initial_call=True

)
def update_filled_data(fill_clicks, fill_nan_method, current_data):
    if current_data is None:
        raise PreventUpdate
    current_data = pd.read_json(current_data, orient='split')
    if fill_nan_method=='Zero':
        filled_data = current_data.fillna(0)
    elif fill_nan_method=='Mean':
        filled_data = current_data.fillna(current_data.mean())
    elif fill_nan_method=='Median':
        filled_data = current_data.fillna(current_data.median())
        #return current_data.to_dict()
    filled_data = filled_data.to_json(orient='split')

    return filled_data, ""


@app.callback(
    Output('transformed-data', 'data'),
    Output('loading-transform', 'children'),
    Input('transform-btn', 'n_clicks'),
    State('method-dropdown', 'value'),
    State('current-data', 'data'),
    prevent_initial_call=True

)
def update_transformed_X(transform_clicks, transform_method, current_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]  


    if (transform_method is None) or (current_data is None):
        raise PreventUpdate()
    transformed_data = get_transformed_data(current_data, transform_method)
    return transformed_data, ""



@app.callback(
    Output('nan-alert', 'children'),
    Input('current-data', 'data'),
    prevent_initial_call=True
)
def update_nan_alert(current_data):
    # current_X=pd.DataFrame(current_X)
    nan_values=0
    if current_data is not None:
        current_data = pd.read_json(current_data, orient='split')
        nan_values = int(current_data.isna().sum().sum())
        return f"{nan_values} empty (NaN/None) values found"
    else:
        return "Data not found, try selecting features or reloading dataset"
        
    




@app.callback(
    Output('base-histogram', 'figure'),
    Output('current-histogram', 'figure'),
    Output('loading-histogram', 'children'),
    Input('update-graphs-btn', 'n_clicks'),
    State('graph-dropdown', 'value'),
    State('current-data', 'data'),
    State('n-bins', 'value'),
    State('loaded-data', 'data'),
    prevent_initial_call=True
)
def update_histograms(update_graph, attr, current_data, n_bins, base_data):
    if current_data is None:
        raise PreventUpdate
    
    if n_bins is not None and (n_bins<0 or type(n_bins) is not int):
        raise PreventUpdate
    if attr is None:
        return px.histogram(), px.histogram(), ""
    return get_histogram(base_data, attr, True, n_bins), get_histogram(current_data, attr, False, n_bins),""


@app.callback(
    Output('split-dataset-list', 'data'),
    Output('loading-split', 'children'),
    Input('split-btn', 'n_clicks'),
    Input('current-data', 'data'),
    State('test-size', 'value'),
    State('current-data','data'),
    State('random-state', 'value'),
    prevent_initial_call=True
)
def split_dataset(split_clicks, cd_input, test_size, current_data,random_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]  
    if input_id=='current-data':
        return None, ""

    if current_data is None:
        raise PreventUpdate
    if (random_state is not None and 
    (type(random_state) is not int or random_state<0)):
        raise PreventUpdate
    if (test_size is not None and 
    type(test_size) is not float or test_size<=0 or test_size>0.999):
        raise PreventUpdate
    
    return get_split_dataset(current_data,  test_size, random_state), ""


@app.callback(
    Output('split-graph', 'figure'),
    Input('split-dataset-list', 'data'),
    prevent_initial_call=True
)
def update_split_graph(split_dataset):
    if split_dataset is None:
        raise PreventUpdate()
        #return get_split_graph(None)
    else:
        return get_split_graph(split_dataset)

@app.callback(
    Output('saved-data', 'data'),
    Output('save-alert', 'style'),
    Output('save-alert', 'children'),
    Output('save-alert', 'color'),
    Input('save-button', 'n_clicks'),
    State('split-dataset-list', 'data'),
    prevent_initial_call=True
)
def save_data(save_click, split_dataset):
    disp_st=[{'display':'none'}, {'display': 'inline'}]
    alerts = [
        "No split dataset.",
        "Data has missing values.",
        "Data saved."
    ]
    colors=['danger', 'info']

    if split_dataset is None:
        return None, disp_st[1], alerts[0], colors[0]
    
    X_train=pd.read_json(split_dataset[0], orient='split')
    X_test=pd.read_json(split_dataset[1], orient='split')
    nan_count=int(X_train.isna().sum().sum())+int(X_test.isna().sum().sum())
    if nan_count>0:
        return None, disp_st[1], alerts[1], colors[0]
    else:
        return split_dataset, disp_st[1], alerts[2], colors[1]

@app.callback(
    Output("scaling-desc","children"),
    Input('method-dropdown', 'value')
)
def update_method_desc(method_name):
    descriptions={
        "StandardScaler":
        """Standardize features by removing the mean and scaling to unit variance""",
        "MinMaxScaler":
        """Transform features by scaling each feature to a given range (here 0-1).""",
        "MaxAbsScaler":
        """Scale each feature by its maximum absolute value.""",
        "RobustScaler":
        """Scale features using statistics that are robust to outliers.""",
        "YeoJohnson":
        """Power transform, Yeo-Johnson method.""",
        "QuantileTransformerNormal":
        """QuantileTransformerNormal""",
        "QuantileTransformerUniform":
        """QuantileTransformerUniform""",
    }
    if method_name in descriptions:
        return descriptions[method_name]
    else:
        return ""



@app.callback(
    Output('scalers-before-split', 'data'),
    Input('loaded-data', 'data'),
    Input('selected-data', 'data'),
    Input('transformed-data', 'data'),
    State('method-dropdown', 'value'),
    State('scalers-before-split', 'data'),
    prevent_initial_call=True
)
def update_scalers_before_split(loaded_data, selected_data, transformed_data,
                    transform_method, scalers):
        
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]  
    
    if input_id=='transformed-data':
        if scalers is not None and scalers!=[None]:
            
            scalers.append(transform_method)
            return scalers
        else:
            return [transform_method]
    else:
        return None

@app.callback(
    Output('scalers-after-split', 'data'),
    Input('split-dataset-list', 'data'),
    State('scalers-before-split', 'data')
)
def update_scalers_after_split(split_data, scalers_before_split):
    # when the data is split, saves list of scalers
    return scalers_before_split


@app.callback(
    Output('tab4','disabled'),
    Input('saved-data','data')
)
def update_tab4(saved_data):
    if saved_data is None:
        return True
    else:
        return False