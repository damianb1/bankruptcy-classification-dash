import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px




import dash_bootstrap_components as dbc
import dash_daq as daq

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from dash.exceptions import PreventUpdate

import json

from app import app








with open('attr_info.json') as json_file:
    attr_info = json.load(json_file)
features = [k for k in attr_info]



def get_updated_graph(current_vars,
             color_class, mark_outliers,
              hide_outliers, y_pred,
              full_attr, loaded_data):

    df=pd.read_json(loaded_data, orient='split')
    df=df.fillna(0)
    df.loc[df['class'] == 0, 'class_str'] = 'not bankrupt' 
    df.loc[df['class'] == 1, 'class_str'] = 'bankrupt'

    if y_pred is not None:
        y_pred=['outlier' if x==-1 else 'inlier' for x in y_pred]

    current_vars={'x':current_vars[0], 'y':current_vars[1]}
        
    if not hide_outliers:
        if color_class:
            if mark_outliers:
                fig1=px.scatter(df, x=current_vars['x'], y=current_vars['y'],
                 color='class_str', symbol_sequence=['circle','x'], symbol=y_pred)
            else:
                fig1=px.scatter(df, x=current_vars['x'], y=current_vars['y'], color='class_str')
        else:
            if mark_outliers:
                fig1=px.scatter(df, x=current_vars['x'], y=current_vars['y'],
                  symbol_sequence=['circle','x'], symbol=y_pred)
            else:
                fig1=px.scatter(df, x=current_vars['x'], y=current_vars['y'])

       

    else: # if hide_outliers
        y_pred=pd.Series(y_pred)
        df_t=df[y_pred=='inlier'] 
        if color_class:
            fig1=px.scatter(df_t, x=current_vars['x'], y=current_vars['y'], color='class_str')
        else: # without selecting class
            fig1=px.scatter(df_t, x=current_vars['x'], y=current_vars['y'])
        
    #fig1.update_layout(showlegend=False)
    fig1.update_layout(legend_title=" ")
    if 'full_attr' in full_attr:
        fig1.update_xaxes(title=attr_info[current_vars['x']])
        fig1.update_yaxes(title=attr_info[current_vars['y']])
    return fig1



def get_new_graph(cvars, color_class, full_attr, loaded_data):

    df=pd.read_json(loaded_data, orient='split')
    df=df.fillna(0)
    df.loc[df['class'] == 0, 'class_str'] = 'not bankrupt' 
    df.loc[df['class'] == 1, 'class_str'] = 'bankrupt'

    cvars={'x':cvars[0], 'y':cvars[1]}
    
    if color_class:
        fig1=px.scatter(df, x=cvars['x'], y=cvars['y'], color='class_str')
        
    else:
        fig1=px.scatter(df, x=cvars['x'], y=cvars['y'])

#coloraxis_showscale=False
    #fig1.update_layout(transition_duration=2000)
    #fig1.update_layout(transition_duration=2000, coloraxis_showscale=False)
    
    if 'full_attr' in full_attr:
        fig1.update_xaxes(title=attr_info[cvars['x']])
        fig1.update_yaxes(title=attr_info[cvars['y']])

    return fig1



outlier_methods=[
    "Robust covariance",
    "One-Class SVM",
    "Isolation Forest",
    "Local Outlier Factor"]

def get_outlier_alg(method, outliers_fraction):
    anomaly_algorithms = {
    "Robust covariance" : EllipticEnvelope(contamination=outliers_fraction),
    "One-Class SVM" : svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1),
    "Isolation Forest" : IsolationForest(contamination=outliers_fraction,
                                         random_state=42),
    "Local Outlier Factor" : LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction)}
    
    return anomaly_algorithms[method]

def get_outlier_series(current_vars, method, fraction, loaded_data):
    df=pd.read_json(loaded_data, orient='split')
    df=df.fillna(0)
    df.loc[df['class'] == 0, 'class_str'] = 'not bankrupt' 
    df.loc[df['class'] == 1, 'class_str'] = 'bankrupt'


    algorithm=get_outlier_alg(method, fraction)
    if method == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(df[current_vars])
    else:
        y_pred = algorithm.fit(df[current_vars]).predict(df[current_vars])
     # 1 for outlier, -1 for regular observation ?
    
    return pd.Series(y_pred)

def return_data(vars_list, loaded_data):
    if len(vars_list)==0:
        return None

    df=pd.read_json(loaded_data, orient='split')
    df=df.fillna(0)
    df.loc[df['class'] == 0, 'class_str'] = 'not bankrupt' 
    df.loc[df['class'] == 1, 'class_str'] = 'bankrupt'

    return df[vars_list].describe().transpose().reset_index().to_dict('records')








scatter_controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("x variable"),
                dcc.Dropdown(
                id='xaxis-state',
                options=[{'label': i, 'value': i} for i in features],
                value='Attr3',
                clearable=False
                
            ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("y variable"),
                dcc.Dropdown(
                id='yaxis-state',
                options=[{'label': i, 'value': i} for i in features],
                value='Attr2',
                clearable=False
            ),
            ]
        ),
        
        dbc.FormGroup(
            [
                html.Button(id='submit-button-state', n_clicks=0, children='OK'),
                dcc.Checklist(id='full-attr-check',
        options=[
            {'label': 'Full attribute info', 'value': 'full_attr'},
            
        ],
        value=[]
        
    ),
                daq.BooleanSwitch(
        id='class-switch',
        on=True,
        label="color class",
        labelPosition="top"
        )  ,  html.Div(id='my-output'),
            ]
        ),

        
    ],
    body=True,
)



outlier_form = dbc.Card(dbc.Row(
    [
        dbc.Col([
        
            dbc.FormGroup(
                [
                    dbc.Label("Anomaly detection method"),
                    dcc.Dropdown(
                        id="outlier-method",
                        options=[{'label': i, 'value': i} for i in outlier_methods],
                        value="Robust covariance"
                        
                    )
                ]
            ),
        
        
            dbc.FormGroup(
                [
                    dbc.Label("percentage"),
                    dbc.Input(
                        type="number",
                        min=0,max=1, step=0.01,
                        id="outlier-percent",
                        value=0.1
                       
                    ),
                ]
            ),
            
        ]),
        
        
        
        
                  
                
        dbc.Col(
            [
                dbc.Button("Find", id='find-outliers-button', color="primary"),
            
            dcc.Loading(id='outlier-switches',
             children=[
                 dcc.Store(id='outliers_pred'),
                 daq.BooleanSwitch(
                    id='mark-outliers-switch',
                    on=False,
                    label="mark outliers",
                    labelPosition="top",
                    disabled=True
                    ),
                 daq.BooleanSwitch(
                    id='hide-outliers-switch',
                    on=False,
                    label="hide outliers",
                    labelPosition="top",
                    disabled=True
                    )
             ]
                 )
            ], align="center"
        ),
       
        


    ]
), body=True)



basic_stats = dbc.Card([
    dbc.Row(
        
        dcc.Dropdown(
            id='table-vars',
        options=[
            {'label': i, 'value': i} for i in features
            
        ],
        value=['Attr1','Attr2','Attr3'],
        multi=True, style={"min-width":200}
           
    )),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
            id='stats-table',
            columns=[{"name": i, "id": i, "type":"numeric", "format" : dict(specifier='.4~f')} for i in 
                            pd.DataFrame([1]).describe().transpose().reset_index().columns], 
        
            ), 

            


        ]), 
        dbc.Col([
             
            dcc.Markdown(id='dataset-info-t2', children="**Dataset:**"),
            
        ]),
            
       
        ]
        )

    
])






layout = dbc.Container(
    [
        html.H2("Introductory analysis"),
        html.Hr(),
        html.H3("Basic stats"),
        basic_stats,


        
        html.Br(), html.Br(),
        dbc.Row(html.H3("Correlation scatterplot")),
        dbc.Row(
            [
                
                dbc.Col(scatter_controls, md=4),
                dbc.Col(
                    dcc.Graph(
                        id='wykres',
                        ), md=8),
            ],
            align="center",
        ),
        
        dbc.Row(
            [
              dbc.Col([html.H5("Description"),
                  dbc.Card([html.P("""
                  Scatterplot shows relationship between two chosen variables.
                  To make it more readable, you can hide outliers. 
                  To find outliers select an algorithm and
                   specify proportion of outliers in data set.
                
                  This action doesn't remove the outliers from data for further analysis.
                  """)]),
              ]),
              dbc.Col([
                  html.H5("Outliers"), 
                  outlier_form
                  ], width=5)  
            ]
        ),

        
        dbc.Row(html.H3("Correlation matrix heatmap")),
        dbc.Row(
            [
                
                
                dbc.Col(
                    dcc.Graph(
                        id='corr-heatmap',
                        ), md=7),

                dbc.Col([
                    html.H5("Method"), 
                    dcc.Dropdown(
                        id='corr-method',
                        options=[
                            {'label':m, 'value':m} for m in ['pearson']#,'kendall','spearman']
                        ], value='pearson',clearable=False
                    ), html.Br(),
                    html.H5('Variables'),
                    dcc.Dropdown(
                        id='corr-heatmap-vars',
                        options=[
                        {'label': i, 'value': i} for i in features+['class']
                        ],
                        value=[x for x in features+['class']],
                        multi=True, style={"min-width":200}
                    )
                ], md=5),
            ],
            align="center",
        ),


       dcc.Store(id='corr_plot_vars', data=['Attr2', 'Attr3']),
       


      
        
            
            
        



    ],
    fluid=True,
)

@app.callback(
    Output(component_id='stats-table',component_property='data'),
    Input(component_id='table-vars', component_property='value'),
    Input('loaded-data', 'data'),
    State('loaded-data', 'data'),
    prevent_initial_call=True
)
def update_stats_table(vars_list, data_change, loaded_data):
    return return_data(vars_list, loaded_data)



@app.callback(Output('wykres', 'figure'),
              Input('corr_plot_vars', 'data') ,
              Input('hide-outliers-switch', 'on'),
              Input('class-switch', 'on'),
              Input('mark-outliers-switch', 'on'),
              State('corr_plot_vars','data'),
            State('outliers_pred','data'),
            State('full-attr-check', 'value'),
            State('loaded-data', 'data'),
            prevent_initial_call=True)
def update_output(new_vars, hide_outliers, color_class, mark_outliers,
                  current_vars, y_pred, full_attr, loaded_data):

    if loaded_data is None:
        raise PreventUpdate
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = '...'
    
        
        return get_new_graph(new_vars, color_class, full_attr)
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]  


    

    if button_id=='corr_plot_vars':
        return get_new_graph(new_vars, color_class, full_attr, loaded_data)
    elif button_id in['class-switch', 'mark-outliers-switch','hide-outliers-switch']:
        return get_updated_graph(current_vars, color_class,
                                 mark_outliers, hide_outliers, y_pred, full_attr, loaded_data)

    

@app.callback(Output('corr_plot_vars', 'data'),
                Input('submit-button-state', 'n_clicks'),
                State('xaxis-state', 'value'),
                State('yaxis-state', 'value'),
                
                )
def update_vars(submit_clicks, xaxis, yaxis):
    return [xaxis, yaxis]

@app.callback(Output('xaxis-state', 'options'),
                Output('yaxis-state', 'options'),
                Input('full-attr-check', 'value'))
def update_attr_names(full_attr):
    if 'full_attr' in full_attr:
        options=[{'label': attr_info[i], 'value': i} for i in features]
    else:
        options=[{'label': i, 'value': i} for i in features]
        
    
    return options,options







@app.callback(
    Output('outliers_pred', 'data'),
    Input('find-outliers-button', 'n_clicks'),
    Input('submit-button-state', 'n_clicks'),
    Input('loaded-data', 'data'),
    State('corr_plot_vars','data'),
    State('outlier-method', 'value'),
    State('outlier-percent', 'value'),
    State('loaded-data', 'data'),
    prevent_initial_call=True
)
def change_y_pred_state(find_click, load_click, loaded_data_inp,
                plot_vars, method, percent,
                        loaded_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        control_id = 'No input yet'
        return None
    else:
        control_id = ctx.triggered[0]['prop_id'].split('.')[0]  
    
  


    if control_id=='submit-button-state':
        return None
    elif control_id=='find-outliers-button':
        try:
            return get_outlier_series(plot_vars, method, percent, loaded_data)
        except:
            return None

    

@app.callback(
    Output('mark-outliers-switch','disabled'),
    Output('hide-outliers-switch','disabled'),
    Input('outliers_pred','data'),
    
)
def disable_switches(store_data):
    if store_data==None:
        return True, True
    else:
        return False, False

@app.callback(
    Output('mark-outliers-switch','on'),
    Output('hide-outliers-switch','on'),
    Input('outliers_pred','data'),
    )
def turn_off_switches(store_data):
    return False, False


@app.callback(
    Output('corr-heatmap','figure'),
    Input('corr-heatmap-vars', 'value'),
    Input('loaded-data', 'data'),
    Input('corr-method', 'value'),
    State('loaded-data', 'data'),
    State('corr-heatmap-vars', 'value'),
    State('corr-method', 'value'),
    prevent_initial_call=True
)
def update_corr_heatmap(heatmap_vars, data_inp, method_inp, loaded_data, hvars, method_st):
    df=pd.read_json(loaded_data, orient='split')
    
    df=df[hvars]
    figure=px.imshow(df.corr(method=method_st))
    figure.update_layout(transition_duration=1000)
    return figure

@app.callback(
    Output('dataset-info-t2', 'children'),
    Input('dataset-dropdown', 'value'),
)
def update_dataset_info_t2(year):
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








