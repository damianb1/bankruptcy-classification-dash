import dash_core_components as dcc
import dash_html_components as html


from app import app
from views import t1,t2,t3,t4




app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content',children=[
        dcc.Tabs(id='tabs', children=[
            dcc.Tab(id='tab1', children=t1.layout, label="Basic info and data loading"),
           dcc.Tab(id='tab2', children=t2.layout, label="Introductory analysis"),
           dcc.Tab(id='tab3', children=t3.layout, label="Preprocessing"),
           dcc.Tab(id='tab4',children=t4.layout, label="Training", disabled=True),
           dcc.Tab(id='tab5',label="Model comparison", disabled=True)
        ]),
    ]),
    
])



app.title="Bankruptcy prediction"


if __name__ == '__main__':
    app.run_server(debug=True)