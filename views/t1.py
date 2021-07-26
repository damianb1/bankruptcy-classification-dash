import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State


from scipy.io import arff

import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate

from app import app





def load_data(name):
    t_data = arff.loadarff("data/"+name+'.arff')
    t_df=pd.DataFrame(t_data[0])
    t_df['class']=pd.to_numeric(t_df['class']) # bytes to int
    return t_df.to_json(orient='split')

def get_datatable(dataf, show_index=False, option="random",
                 random_size=20, 
                 range_start=0, range_stop=0
                 ):
    if show_index:
        dataf=dataf.reset_index()

    if option=="whole":
        pass
    elif option=="random":
        if random_size not in range(0,len(dataf.index)):
            random_size=20
        dataf=dataf.sample(random_size)
    elif option=="range":
        if range_stop not in range(0, len(dataf.index)):
            range_stop=len(dataf.index)-1
        if range_start not in range(0,range_stop+1):
            range_start=0
        dataf=dataf[range_start:range_stop]


    
    table=dash_table.DataTable(
        id='table',
        columns=[
            {"name": i, "id": i,"type":"numeric", "format" : dict(specifier='.4~f')}
             for i in dataf.columns], 
        style_table={'overflowX': 'auto','width':750, 'max-height':750},
        data=dataf.to_dict('records'),
        editable=False
        )
    return table





display_options = dbc.Row([
    dbc.Col([

        "Dataset:",

        dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': '1st year', 'value': '1year'}, 
                    {'label': '2nd year', 'value': '2year'}, 
                    {'label': '3rd year', 'value': '3year'}, 
                    {'label': '4th year', 'value': '4year'}, 
                    {'label': '5th year', 'value': '5year'}, 
                        ],
                value='1year',
                clearable=False,
                style={'width':200}
            ),
            html.Br(),
        "Data display options:",
        dcc.RadioItems(
        id='display-radio',
        options=[
            {'label': 'Whole dataset', 'value': 'whole'},
            {'label': 'Random sample', 'value': 'random'},
            {'label': 'Records within a range', 'value': 'range'},
            ],
        value='random',
        labelStyle={'display': 'block'}
    ),
    html.Div(id='random-params', children=[
        dbc.Label("Sample size:"),
        dbc.Input(id='sample-size', type='number', min=0, step=1, value=10,style={'width':200})
    ], 
    ),
    html.Div(id='range-params', children=[
        dbc.Label("Range:"),
        dcc.RangeSlider(
        id='range-slider',
        min=0,
        max=10000,
        step=1,
        value=[5, 15]
    ),
    ], #style={'display':'none'}
    ),
    dbc.Button(id='display-btn', children='Ok')
    ]),

],) #style={'width':'600px'}
 
data_info = html.Div([
    #dbc.Row([
        html.H4("Data set information:"),
        dcc.Markdown("""
                Source:
                    https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

                    Zieba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. Expert Systems with Applications.

            The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service (EMIS), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.
            Basing on the collected data five classification cases were distinguished, that depends on the forecasting period:
            * 1st year - the data contains financial rates from 1st year of the forecasting period and corresponding class label
             that indicates bankruptcy status after 5 years. The data contains 7027 instances (financial statements), 271 represents bankrupted companies, 6756 firms that did not bankrupt in the forecasting period.
            * 2nd year - the data contains financial rates from 2nd year of the forecasting period and corresponding class label
             that indicates bankruptcy status after 4 years. The data contains 10173 instances (financial statements), 400 represents bankrupted companies, 9773 firms that did not bankrupt in the forecasting period.
            * 3rd year - the data contains financial rates from 3rd year of the forecasting period and corresponding class label
             that indicates bankruptcy status after 3 years. The data contains 10503 instances (financial statements), 495 represents bankrupted companies, 10008 firms that did not bankrupt in the forecasting period.
            * 4th year - the data contains financial rates from 4th year of the forecasting period and corresponding class label
             that indicates bankruptcy status after 2 years. The data contains 9792 instances (financial statements), 515 represents bankrupted companies, 9277 firms that did not bankrupt in the forecasting period.
            * 5th year - the data contains financial rates from 5th year of the forecasting period and corresponding class label
             that indicates bankruptcy status after 1 year. The data contains 5910 instances (financial statements), 410 represents bankrupted companies, 5500 firms that did not bankrupt in the forecasting period.
        
                
        """),

    #]),
        dbc.Row([
            
            dbc.Col([
                html.H4("Attribute information"),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Markdown("""

                    X1 net profit / total assets

                    X2 total liabilities / total assets

                    X3 working capital / total assets

                    X4 current assets / short-term liabilities

                    X5 \[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)\] * 365
                    
                    X6 retained earnings / total assets
                    X7 EBIT / total assets

                    X8 book value of equity / total liabilities

                    X9 sales / total assets

                    X10 equity / total assets

                    X11 (gross profit + extraordinary items + financial expenses) / total assets
                    
                    X12 gross profit / short-term liabilities
                    
                    X13 (gross profit + depreciation) / sales

                    X14 (gross profit + interest) / total assets

                    X15 (total liabilities * 365) / (gross profit + depreciation)

                    X16 (gross profit + depreciation) / total liabilities

                    X17 total assets / total liabilities

                    X18 gross profit / total assets

                    X19 gross profit / sales

                    X20 (inventory * 365) / sales

                    X21 sales (n) / sales (n-1)

                    X22 profit on operating activities / total assets

                    X23 net profit / sales

                    X24 gross profit (in 3 years) / total assets

                    X25 (equity - share capital) / total assets

                    X26 (net profit + depreciation) / total liabilities

                    X27 profit on operating activities / financial expenses

                    X28 working capital / fixed assets

                    X29 logarithm of total assets

                    X30 (total liabilities - cash) / sales

                    X31 (gross profit + interest) / sales

                    X32 (current liabilities * 365) / cost of products sold
                """)
            ]),
            dbc.Col([
                dcc.Markdown("""
                    X33 operating expenses / short-term liabilities

                    X34 operating expenses / total liabilities

                    X35 profit on sales / total assets

                    X36 total sales / total assets

                    X37 (current assets - inventories) / long-term liabilities

                    X38 constant capital / total assets

                    X39 profit on sales / sales
                    
                    X40 (current assets - inventory - receivables) / short-term liabilities
                    
                    X41 total liabilities / ((profit on operating activities + depreciation) * (12/365))
                    
                    X42 profit on operating activities / sales
                    
                    X43 rotation receivables + inventory turnover in days
                    
                    X44 (receivables * 365) / sales
                    
                    X45 net profit / inventory
                    
                    X46 (current assets - inventory) / short-term liabilities
                    
                    X47 (inventory * 365) / cost of products sold
                    
                    X48 EBITDA (profit on operating activities - depreciation) / total assets
                    
                    X49 EBITDA (profit on operating activities - depreciation) / sales
                    
                    X50 current assets / total liabilities
                    
                    X51 short-term liabilities / total assets
                    
                    X52 (short-term liabilities * 365) / cost of products sold)
                    
                    X53 equity / fixed assets
                    
                    X54 constant capital / fixed assets
                    
                    X55 working capital
                    
                    X56 (sales - cost of products sold) / sales
                    
                    X57 (current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)
                    
                    X58 total costs /total sales
                    
                    X59 long-term liabilities / equity
                    
                    X60 sales / inventory
                    
                    X61 sales / receivables
                    
                    X62 (short-term liabilities *365) / sales
                    
                    X63 sales / short-term liabilities

                    X64 sales / fixed assets
                """)
            ])


        ])

        

])

layout = dbc.Container([
    html.H4("Project description"),
    dcc.Markdown("""
        This app is an example of a dashboard for a machine learning experiment. 
        The main goal is to create a model that accurately predicts bankruptcy of firms.
        It is a binary classification problem.
        The app allows you to test various classification methods and techniques supporting
         this task such as data preprocessing and visualization tools. 

        Application is divided into 4 parts:

        1st tab: project description, data description, data loading

        2nd tab: basic analysis of input variables, correlation analysis, visualization

        3rd tab: data preprocessing; feature selection, missing values, standardization, train-test split
        
        4th tab: selecting classification algorithm, specyfying parameters, cross-validation, training, model evaluation

    """),


    html.Br(),
    html.H4("Data loading"),
    dbc.Row([
        dbc.Col([
            dcc.Loading(html.Div(
            id='table-div',

        )),
        

        dcc.Loading(html.Div(id='data-loading'), type='circle'),
    
            ]),

        dbc.Col([
            display_options
        ]),

    ]),


   

#style={'width':500}
html.Br(),

data_info,




    
dcc.Store(id='loaded-data'),
    

    
    ]
)

@app.callback(Output('loaded-data', 'data'),
                Output('data-loading', 'children'),
                Input('dataset-dropdown', 'value'))
def update_data(year):
    return load_data(year), ""


@app.callback(Output('table-div', 'children'),
                Output('sample-size', 'max'),
                Output('range-slider', 'max'),
                Output('range-slider', 'marks'),
                Input('loaded-data','data'),
                Input('display-btn', 'n_clicks'),
                State('display-radio', 'value'),
                State('sample-size', 'value'),
                State('range-slider', 'value'),
                State('loaded-data', 'data'))
def update_table(loaded_data, disp_btn, option, sample_size, range_values, data_state):
    


    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]  

 
    if input_id=='loaded-data':
        df=pd.read_json(loaded_data, orient='split')
        
 
    elif input_id=='display-btn':
        df=pd.read_json(data_state, orient='split')
        
    max_size=len(df)
    steps = [int(x*.25*max_size) for x in range(5)]
    marks={x:str(x) for x in steps}

    

    return (get_datatable(df, show_index=True, option=option,
     random_size=sample_size, range_start=range_values[0], range_stop=range_values[1]),
     max_size, max_size, marks)



    

@app.callback(
    Output('random-params', 'style'),
    Output('range-params', 'style'),
    Input('display-radio','value')
)
def update_display_params(disp_val):
    dstyles=[{'display':'none'},{'display':'block'}]
    if disp_val=='whole':
        return dstyles[0], dstyles[0]
    elif disp_val=='random':
        return dstyles[1], dstyles[0]
    elif disp_val=='range':
        return dstyles[0], dstyles[1]

#if __name__ == '__main__':
#    app.run_server(debug=True)
