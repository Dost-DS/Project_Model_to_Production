import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import logging
import time
import pandas as pd
import json
from threading import Thread
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__)


data_store = {
    'timestamps': [],
    'temperatures': [],
    'humidities': [],
    'sounds': [],
    'anomaly_scores': [],
    'last_update': None,
    'api_logs': [],
    'anomalies': [],
    'statistics': {}
}


data_queue = queue.Queue()


COLORS = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'grid': '#ecf0f1',
    'temperature': '#e74c3c',
    'humidity': '#3498db',
    'sound': '#2ecc71',
    'anomaly': '#e67e22',
    'table_header': '#34495e',
    'table_row': '#ffffff',
    'table_alt_row': '#f8f9fa',
    'table_border': '#dee2e6',
    'status_good': '#2ecc71',
    'status_warning': '#f1c40f',
    'status_error': '#e74c3c',
    'api_success': '#2ecc71',
    'api_error': '#e74c3c',
    'api_warning': '#f1c40f'
}


REQUEST_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 1

# Main app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('EWT Direct Wind', style={'textAlign': 'center', 'color': COLORS['text']}),
        html.P('Real-time monitoring of temperature, humidity, and sound levels',
               style={'textAlign': 'center', 'color': COLORS['text']}),
        html.Div([
            html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': COLORS['text'],
                               'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
            html.Div(id='last-update-status',
                     style={'display': 'inline-block', 'marginLeft': '20px', 'color': COLORS['text']})
        ], style={'textAlign': 'center'})
    ], style={'margin': '20px 0px'}),

    # Auto-refresh component (updates every 2 seconds)
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Changed from 5000 to 2000ms (2 seconds)
        n_intervals=0
    ),

    # Store for holding data between callbacks
    dcc.Store(id='data-store'),

    # Current Status Panel
    html.Div([
        html.H3('Current Status', style={'color': COLORS['text'], 'marginBottom': '20px'}),
        html.Div(id='current-readings', style={'display': 'flex', 'justifyContent': 'space-around'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': 'white',
              'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}),

    # Sensor Readings Graphs
    html.Div([
        html.Div([dcc.Graph(id='temperature-graph')], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='humidity-graph')], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='sound-graph')], style={'width': '33%', 'display': 'inline-block'})
    ], style={'margin': '20px'}),

    # Anomaly Analysis
    html.Div([
        html.Div([dcc.Graph(id='anomaly-score-graph')], style={'width': '100%'})
    ], style={'margin': '20px'}),

    # System Statistics Panel
    html.Div([
        html.H3('System Statistics', style={'color': COLORS['text'], 'marginBottom': '20px'}),
        html.Div(id='statistics-panel', style={'display': 'flex', 'justifyContent': 'space-around'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': 'white',
              'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}),

    # Recent Anomalies Table
    html.Div([
        html.H3('Recent Anomalies', style={'color': COLORS['text'], 'marginBottom': '20px'}),
        html.Div(id='anomaly-log-table', style={'overflowX': 'auto'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': 'white',
              'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}),

    # API Logs Table
    html.Div([
        html.H3('API Logs', style={'color': COLORS['text'], 'marginBottom': '20px'}),
        html.Div(id='api-logs-table', style={'overflowX': 'auto'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': 'white',
              'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'})
], style={'backgroundColor': COLORS['background'], 'padding': '20px'})


def make_request(url, method='GET', retries=MAX_RETRIES):
    """Make a request to the API with retry logic and logging."""
    start_time = time.time()
    for attempt in range(retries):
        try:
            response = requests.request(method, url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response_time = (time.time() - start_time) * 1000

            # Log the successful API call
            log_entry = {
                'timestamp': datetime.now(),
                'endpoint': url,
                'status': 'success',
                'response_time': response_time,
                'details': {
                    'status_code': response.status_code,
                    'response_size': len(response.content),
                    'headers': dict(response.headers)
                }
            }
            data_store['api_logs'].append(log_entry)

            # Keep only the last 50 logs
            if len(data_store['api_logs']) > 50:
                data_store['api_logs'].pop(0)

            return response
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                logger.error(f"Failed to make request to {url} after {retries} attempts: {e}")

                # Log the failed API call
                response_time = (time.time() - start_time) * 1000
                log_entry = {
                    'timestamp': datetime.now(),
                    'endpoint': url,
                    'status': 'error',
                    'response_time': response_time,
                    'details': {
                        'error': str(e),
                        'attempts': retries
                    }
                }
                data_store['api_logs'].append(log_entry)

                if len(data_store['api_logs']) > 50:
                    data_store['api_logs'].pop(0)

                return None
            logger.warning(f"Request attempt {attempt + 1} failed, retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    return None


def data_fetcher():

    while True:
        try:
            # Fetch latest data
            response = make_request('http://localhost:5000/latest')
            if response and response.status_code == 200:
                data = response.json()
                if data:
                    data_queue.put(('latest', data))


            stats_response = make_request('http://localhost:5000/stats')
            if stats_response and stats_response.status_code == 200:
                stats = stats_response.json()
                data_queue.put(('stats', stats))



            anomalies_response = make_request('http://localhost:5000/anomalies')
            if anomalies_response and anomalies_response.status_code == 200:
                anomalies = anomalies_response.json()
                data_queue.put(('anomalies', anomalies))

            # Sleep for a short time
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in data fetcher thread: {e}")
            time.sleep(2)


def fetch_historical_data():
    """Fetch historical data from the API."""
    try:
        response = make_request('http://localhost:5000/readings?limit=100')
        if response and response.status_code == 200:
            data = response.json()
            if data:
                data_store['timestamps'] = [datetime.fromisoformat(r['timestamp']) for r in data]
                data_store['temperatures'] = [r['temperature'] for r in data]
                data_store['humidities'] = [r['humidity'] for r in data]
                data_store['sounds'] = [r['sound'] for r in data]
                data_store['anomaly_scores'] = [r['anomaly_score'] for r in data]
                data_store['last_update'] = datetime.now()
                return True
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
    return False


def update_data_store():
    """Process queued data updates."""
    updated = False
    while not data_queue.empty():
        try:
            data_type, data = data_queue.get_nowait()

            if data_type == 'latest':
                input_data = data['input_data']
                prediction = data['prediction']

                # Update data lists with new readings
                data_store['timestamps'].append(datetime.fromisoformat(input_data['timestamp']))
                data_store['temperatures'].append(input_data['temperature'])
                data_store['humidities'].append(input_data['humidity'])
                data_store['sounds'].append(input_data['sound'])
                data_store['anomaly_scores'].append(prediction['anomaly_score'])

                # Keep only the last 100 readings
                if len(data_store['timestamps']) > 100:
                    data_store['timestamps'].pop(0)
                    data_store['temperatures'].pop(0)
                    data_store['humidities'].pop(0)
                    data_store['sounds'].pop(0)
                    data_store['anomaly_scores'].pop(0)

                data_store['last_update'] = datetime.now()
                updated = True

            elif data_type == 'stats':
                data_store['statistics'] = data
                updated = True

            elif data_type == 'anomalies':
                data_store['anomalies'] = data
                updated = True

            data_queue.task_done()
        except Exception as e:
            logger.error(f"Error processing data update: {e}")

    return updated


def create_gauge_indicator(value, title, min_val, max_val, normal_range):
    """Create a gauge indicator for current readings."""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkgray"},
            'steps': [
                {'range': [min_val, normal_range[0]], 'color': "lightgray"},
                {'range': normal_range, 'color': "lightgreen"},
                {'range': [normal_range[1], max_val], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))


def create_anomaly_table(anomalies):
    """Create a table to display anomalies."""
    if not anomalies:
        return html.Div("No anomalies detected")

    df = pd.DataFrame(anomalies)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('Timestamp', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Temperature (°C)', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Humidity (%)', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Sound (dB)', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Anomaly Score', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Issue Type', style={'backgroundColor': COLORS['table_header'], 'color': 'white'})
        ])),
        html.Tbody([
            html.Tr([
                html.Td(row['timestamp']),
                html.Td(f"{row['temperature']:.1f}"),
                html.Td(f"{row['humidity']:.1f}"),
                html.Td(f"{row['sound']:.1f}"),
                html.Td(f"{row['anomaly_score']:.2f}"),
                html.Td(row['issue_type'] or 'Unknown')
            ], style={'backgroundColor': COLORS['table_row'] if i % 2 == 0 else COLORS['table_alt_row']})
            for i, row in df.iterrows()
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'border': f'1px solid {COLORS["table_border"]}',
        'marginTop': '10px'
    })

    return table


def create_api_logs_table(logs):
    """Create a table to display API logs."""
    if not logs:
        return html.Div("No API logs available")

    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    def get_status_color(status):
        if status == 'success':
            return COLORS['api_success']
        elif status == 'error':
            return COLORS['api_error']
        else:
            return COLORS['api_warning']

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('Timestamp', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Endpoint', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Status', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Response Time (ms)', style={'backgroundColor': COLORS['table_header'], 'color': 'white'}),
            html.Th('Details', style={'backgroundColor': COLORS['table_header'], 'color': 'white'})
        ])),
        html.Tbody([
            html.Tr([
                html.Td(row['timestamp']),
                html.Td(row['endpoint']),
                html.Td(row['status'], style={'color': get_status_color(row['status'])}),
                html.Td(f"{row['response_time']:.0f}"),
                html.Td(html.Div([
                    html.Details([
                        html.Summary('View Details'),
                        html.Pre(json.dumps(row['details'], indent=2),
                                 style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'})
                    ])
                ]))
            ], style={'backgroundColor': COLORS['table_row'] if i % 2 == 0 else COLORS['table_alt_row']})
            for i, row in df.iterrows()
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'border': f'1px solid {COLORS["table_border"]}',
        'marginTop': '10px'
    })

    return table


def get_status_color():

    if data_store['last_update'] is None:
        return COLORS['status_error']

    time_diff = (datetime.now() - data_store['last_update']).total_seconds()
    if time_diff < 5:
        return COLORS['status_good']
    elif time_diff < 15:
        return COLORS['status_warning']
    else:
        return COLORS['status_error']


# Background data fetching callback
@app.callback(
    Output('data-store', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_data_store_callback(n_intervals):

    update_data_store()
    return {'timestamp': datetime.now().isoformat()}


# Main dashboard update callback
@app.callback(
    [Output('temperature-graph', 'figure'),
     Output('humidity-graph', 'figure'),
     Output('sound-graph', 'figure'),
     Output('anomaly-score-graph', 'figure'),
     Output('current-readings', 'children'),
     Output('statistics-panel', 'children'),
     Output('anomaly-log-table', 'children'),
     Output('last-update-status', 'children'),
     Output('api-logs-table', 'children')],
    [Input('data-store', 'data'),
     Input('refresh-button', 'n_clicks')]
)
def update_dashboard(data_store_timestamp, n_clicks):

    ctx = dash.callback_context
    if not ctx.triggered:
        button_clicked = False
    else:
        button_clicked = ctx.triggered[0]['prop_id'].split('.')[0] == 'refresh-button'

    if button_clicked or not data_store['timestamps']:
        fetch_historical_data()
        update_data_store()

    if not data_store['timestamps']:
        return [go.Figure()] * 4 + [html.Div("No data available")] * 2 + [html.Div("No anomalies detected")] + [
            html.Div("No data available")] + [html.Div("No API logs available")]

    # Temperature graph
    temp_fig = go.Figure(data=[go.Scatter(
        x=data_store['timestamps'],
        y=data_store['temperatures'],
        name='Temperature',
        line=dict(color=COLORS['temperature'])
    )])
    temp_fig.update_layout(
        title='Temperature Trend',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            range=[data_store['timestamps'][-1] - timedelta(minutes=5), data_store['timestamps'][-1]],
            showgrid=True,
            gridcolor=COLORS['grid']
        )
    )


    humid_fig = go.Figure(data=[go.Scatter(
        x=data_store['timestamps'],
        y=data_store['humidities'],
        name='Humidity',
        line=dict(color=COLORS['humidity'])
    )])
    humid_fig.update_layout(
        title='Humidity Trend',
        xaxis_title='Time',
        yaxis_title='Humidity (%)',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            range=[data_store['timestamps'][-1] - timedelta(minutes=5), data_store['timestamps'][-1]],
            showgrid=True,
            gridcolor=COLORS['grid']
        )
    )

    # Sound graph
    sound_fig = go.Figure(data=[go.Scatter(
        x=data_store['timestamps'],
        y=data_store['sounds'],
        name='Sound',
        line=dict(color=COLORS['sound'])
    )])
    sound_fig.update_layout(
        title='Sound Level Trend',
        xaxis_title='Time',
        yaxis_title='Sound (dB)',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            range=[data_store['timestamps'][-1] - timedelta(minutes=5), data_store['timestamps'][-1]],
            showgrid=True,
            gridcolor=COLORS['grid']
        )
    )

    # Anomaly score graph
    score_fig = go.Figure(data=[go.Scatter(
        x=data_store['timestamps'],
        y=data_store['anomaly_scores'],
        name='Anomaly Score',
        line=dict(color=COLORS['anomaly'])
    )])
    score_fig.update_layout(
        title='Anomaly Score Trend',
        xaxis_title='Time',
        yaxis_title='Anomaly Score',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            range=[data_store['timestamps'][-1] - timedelta(minutes=5), data_store['timestamps'][-1]],
            showgrid=True,
            gridcolor=COLORS['grid']
        )
    )

    # Current readings gauges
    current_readings = [
        html.Div([
            dcc.Graph(
                figure=create_gauge_indicator(
                    data_store['temperatures'][-1],
                    'Temperature (°C)',
                    10, 40, (18, 26)
                ),
                config={'displayModeBar': False}
            )
        ], style={'width': '33%'}),
        html.Div([
            dcc.Graph(
                figure=create_gauge_indicator(
                    data_store['humidities'][-1],
                    'Humidity (%)',
                    20, 90, (35, 55)
                ),
                config={'displayModeBar': False}
            )
        ], style={'width': '33%'}),
        html.Div([
            dcc.Graph(
                figure=create_gauge_indicator(
                    data_store['sounds'][-1],
                    'Sound (dB)',
                    30, 120, (60, 85)
                ),
                config={'displayModeBar': False}
            )
        ], style={'width': '33%'})
    ]

    # Statistics panel
    if data_store['statistics']:
        stats = data_store['statistics']
        statistics_panel = [
            html.Div([
                html.H4('Total Readings'),
                html.P(stats['total_readings'])
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4('Total Anomalies'),
                html.P(stats['total_anomalies'])
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4('Anomaly Rate'),
                html.P(f"{stats['anomaly_rate']}%")
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4('Avg Temperature'),
                html.P(f"{stats['average_temperature']:.1f}°C")
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4('Avg Humidity'),
                html.P(f"{stats['average_humidity']:.1f}%")
            ], style={'textAlign': 'center'}),
            html.Div([
                html.H4('Avg Sound'),
                html.P(f"{stats['average_sound']:.1f} dB")
            ], style={'textAlign': 'center'})
        ]
    else:
        statistics_panel = [html.Div("Statistics unavailable")]

    # Anomaly table
    anomaly_table = create_anomaly_table(data_store['anomalies'])

    # Update status
    status_color = get_status_color()
    last_update_text = f"Last Update: {data_store['last_update'].strftime('%H:%M:%S') if data_store['last_update'] else 'Never'}"
    status_indicator = html.Div([
        html.Span("●", style={'color': status_color, 'marginRight': '5px'}),
        html.Span(last_update_text)
    ])

    # API logs table
    api_logs_table = create_api_logs_table(data_store['api_logs'])

    return temp_fig, humid_fig, sound_fig, score_fig, current_readings, statistics_panel, anomaly_table, status_indicator, api_logs_table


if __name__ == '__main__':
    logger.info("Starting dashboard server...")
    print("Make sure the API server (app.py) is running on http://localhost:5000")

    # Start the background data fetcher thread
    data_thread = Thread(target=data_fetcher, daemon=True)
    data_thread.start()

    # Initial data fetch
    fetch_historical_data()

    print("Starting dashboard on http://localhost:8050")
    print("Dashboard will update every 2 seconds")
    app.run_server(debug=True, port=8050, host='127.0.0.1')