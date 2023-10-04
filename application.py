import dash
import pickle
import xgboost 
import pandas as pd
from dash import Dash, html, dcc, callback, dash_table
from dash.dependencies import Input, Output, State
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',  # Basic Dash styling
    '/assets/styles.css'  # Custom styles

]



# Initialize the Dash app
application = Dash(__name__, external_stylesheets=external_stylesheets)
server=application.server
app=application
# Define the app layout
app.layout = html.Div([
    
    html.H1("Adverse event prediction", style={'textAlign': 'center', 'color': 'white', 'whiteSpace': 'nowrap'}),

 
    html.Div([
    html.Div([
        # Adding Age 
        html.Div([
            html.Label("Enter Age in years:", style={'color': 'white'}),
            dcc.Input(id='age-input', type='number', placeholder='Age', style={'width': '100%', 'height': '40px', 'fontSize': '18px'}),
        ], className='six columns',style={'marginBottom': '20px'}),
        
        
        # Adding Weight 
        html.Div([
            html.Label("Enter Weight in Kg:", style={'color': 'white'}),
            dcc.Input(id='weight-input', type='number', placeholder='Weight', style={'width': '100%', 'height': '40px', 'fontSize': '18px'}),
        ], className='six columns',style={'marginBottom': '20px'}),
        
        ], className='row'),
    html.Div([
        # Adding Height 
        html.Div([
            html.Label("Enter Height in cm:", style={'color': 'white'}),
            dcc.Input(id='height-input', type='number', placeholder='Height', style={'width': '100%', 'height': '40px', 'fontSize': '18px'}),
        ], className='six columns',style={'marginBottom': '20px'}),
        

        # Adding gender 
        html.Div([
            html.Label("Select Gender:", style={'color': 'white'}),
            dcc.Dropdown(
                id='gender-input',
                options=[
                    {'label': 'Male', 'value': '1'},
                    {'label': 'Female', 'value': '0'}
                ],
                value='1',
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),
        ], className='row'),
        # Adding medical history dropdown
    html.Div([
        html.Div([
            html.Label("Medical History:", style={'color': 'white'}),
            dcc.Dropdown(
                id='medical-history-input',
                options=[
                    {'label': 'Asthma', 'value': '0'},
                    {'label': 'Diabetes', 'value': '1'},
                    {'label': 'Hypertension', 'value': '2'},
                    {'label': 'None', 'value': '3'}
                ],
                value='3',
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),

         # Adding drug name dropdown
        html.Div([
            html.Label("Select Drug Name:", style={'color': 'white'}),
            dcc.Dropdown(
                id='drug-name-input',
                options=[
                    {'label': 'DrugA', 'value': '0'},
                    {'label': 'DrugB', 'value': '1'},
                    {'label': 'DrugC', 'value': '2'},
                    {'label': 'DrugD', 'value': '3'}
                ],
                value='0',  # default value
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),
        ], className='row'),
    html.Div([
        # Adding Dosage in mg
        html.Div([
            html.Label("Select Dosage in mg:", style={'color': 'white'}),
            dcc.Dropdown(
                id='dosage-input',
                options=[
                    {'label': '10mg', 'value': '0'},
                    {'label': '250mg', 'value': '1'},
                    {'label': '500mg', 'value': '2'},
                    {'label': '5mg', 'value': '3'}
                ],
                value='0',
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),


          # Adding Frequency of intake dropdown
        html.Div([
            html.Label("Select Intake per Day:", style={'color': 'white'}),
            dcc.Dropdown(
                id='frequency-input',
                options=[
                    {'label': 'Once daily', 'value': '0'},
                    {'label': 'Twice daily', 'value': '1'}
                ],
                value='0',  # default value
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),
        ], className='row'),
    html.Div([
        # Adding Country dropdown
        html.Div([
            html.Label("Select Country:", style={'color': 'white'}),
            dcc.Dropdown(
                id='country-input',
                options=[
                    {'label': 'Japan', 'value': '8'},
                    {'label': 'Mexico', 'value': '9'},
                    {'label': 'UK', 'value': '13'},
                    {'label': 'Spain', 'value': '12'},
                    {'label': 'Australia', 'value': '0'},
                    {'label': 'South Africa', 'value': '11'},
                    {'label': 'China', 'value': '3'},
                    {'label': 'Russia', 'value': '10'},
                    {'label': 'USA', 'value': '14'},
                    {'label': 'France', 'value': '4'},
                    {'label': 'Germany', 'value': '5'},
                    {'label': 'India', 'value': '6'},
                    {'label': 'Canada', 'value': '2'},
                    {'label': 'Italy', 'value': '7'},
                    {'label': 'Brazil', 'value': '1'}
                ],
                value='8',  # default value
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),


        # Adding Duration of Treatment dropdown
        html.Div([
            html.Label("Treatment duration:", style={'color': 'white'}),
            dcc.Dropdown(
                id='duration-input',
                 options=[
                    {'label': '1 month', 'value': '0'},
                    {'label': '2 weeks', 'value': '1'},
                    {'label': '3 months', 'value': '2'},
                    {'label': '6 months', 'value': '3'}
                ],
                value='0',  # default value
                style={'width': '100%', 'fontSize': '18px'}
            )
        ], className='six columns',style={'marginBottom': '20px'}),
        ], className='row'),

        html.Button('Submit', id='submit-button', n_clicks=0, style={'width': '100%', 'height': '40px', 'fontSize': '18px', 'margin': '10px 0px'}),
    
    ], style={'maxWidth': '400px', 'margin': 'auto'}),

    html.Div(id='prediction-output', style={'color': 'white'}),

### 2 part layout sentiment analysis

    html.Div([
        html.H1("Sentiment Analysis", style={'textAlign': 'center', 'color': 'white', 'whiteSpace': 'nowrap',
        'marginLeft': '0px','textAlign': 'center', 'marginBottom': '20px', 'color': 'white'}),

    html.Div([
    dcc.Input(id='user-input',
                      type='text',
                      placeholder='Enter some text...',
                      style={'width': '100%', 'height': '50px', 'fontSize': '18px'}),
     html.Button('Analyze Sentiment',
                        id='analyze-button',
                        style={'marginLeft': '0px', 'backgroundColor': '#3498DB', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}),
           
        ], style={'marginBottom': '20px'}),
    

    ], style={'maxWidth': '800px', 'margin': 'auto', 'marginTop': '20px'}),  # Adjust margin as per your design

    
    html.Div(id='sentiment-output', style={'color': 'white'})
    # html.Div(id='output-div', style={'color': 'white'})  
], style={'maxWidth': '400px', 'margin': 'auto'})   

# Placeholder list to store the input data
data_list = []

# Define callback to update the output-div when the submit button is clicked
@app.callback(
    [
        Output('prediction-output', 'children'),
        Output('sentiment-output', 'children')
    ],
    [
        Input('submit-button', 'n_clicks'),
        Input('analyze-button', 'n_clicks')
    ],
    [
        State('age-input', 'value'),
        State('weight-input', 'value'),
        State('height-input', 'value'),
        State('gender-input', 'value'),
        State('medical-history-input', 'value'),# Adding medical history input state
        State('drug-name-input', 'value'), ## Adding drug name input state   
        State('dosage-input', 'value'),## adding dosage  
        State('frequency-input', 'value'),  # Adding frequency input state
        State('country-input', 'value'),  # Adding country input state   
        State('duration-input', 'value'),  # Adding duration input state 
        State('user-input', 'value')
    ]
    
)

def update_output(submit_n_clicks, analyze_n_clicks, age, weight, height, gender, medical_history,drug_name,dosage,frequency,country,duration,sentiment_input):
    global data_list  
    
    
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]    
        
    prediction_output = ""
    sentiment_output = ""

    if button_id == 'analyze-button' and analyze_n_clicks > 0:
        # Predict sentiment of the entered text
        result = sentiment_pipeline(sentiment_input)
        sentiment = result[0]['label']

        # Choose the appropriate GIF based on sentiment
        if sentiment == 'POSITIVE':
            emoji_gif = "/assets/positive.gif"
        elif sentiment == 'NEGATIVE':
            emoji_gif = "/assets/negative.gif"
        else:  # Assuming neutral or any other label
            emoji_gif = "/assets/neutral.gif"

        sentiment_output = html.Div([
            html.Span(f'Sentiment of the text: {sentiment} ', style={"fontSize": "30px"}),
            html.Img(src=emoji_gif, style={"height": "50px", "verticalAlign": "middle", "marginLeft": "15px"})
        ])
    elif analyze_n_clicks is None:
        sentiment_output = "Get the Sentiment!!!"


   

    # Logic for submit-button
    if button_id == 'submit-button' and submit_n_clicks > 0:
        #... [Your logic for prediction] ...
        user_data = [
            age, 
            gender, 
            weight, 
            height, 
            medical_history, 
            drug_name, 
            dosage, 
            frequency, 
            duration, 
            country
        ]

        try:
            # Convert user_data values to integers and predict
            user_data = [int(value) for value in user_data]

            with open('data/xgboost_model3.pickle', 'rb') as f:
                loaded_bst = pickle.load(f)

            pred = loaded_bst.predict([user_data])
            label_mapping = {0: 'Chest Pain', 1: 'Dizziness', 2: 'None', 3: 'Rash'}
            prediction_label = label_mapping.get(pred[0], "Unknown Label")
            prediction_output = f"Possible Adverse event: {prediction_label}"
        except ValueError:
            prediction_output = "Submission Failed: All values must be convertible to integers."
        except Exception as e:
            prediction_output = f"An error occurred: {str(e)}"


    return prediction_output, sentiment_output

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=True,port=5000)


