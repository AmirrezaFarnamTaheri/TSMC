import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# App initialization
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)
app.title = "Tehran Stock Market Prediction Dashboard"

# Theme configuration
DARK_THEME = "plotly_dark"
LIGHT_THEME = "plotly_white"

# Layout components
def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand("Tehran Stock Prediction", className="ms-2"),
                ], width="auto"),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="#", active=True)),
                        dbc.NavItem(dbc.NavLink("Analysis", href="#")),
                        dbc.NavItem(dbc.NavLink("Settings", href="#")),
                    ], navbar=True)
                ]),
                dbc.Col([
                    dbc.Switch(
                        id="theme-switch",
                        label="Dark Mode",
                        value=True,
                        className="d-inline-block"
                    )
                ], width="auto")
            ])
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_stock_selector():
    return dbc.Card([
        dbc.CardHeader("Select Stock"),
        dbc.CardBody([
            dcc.Dropdown(
                id="stock-dropdown",
                placeholder="Select a stock...",
                value=None,
                clearable=False
            ),
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button("1D", id="range-1d", size="sm", outline=True, color="primary"),
                    dbc.Button("1W", id="range-1w", size="sm", outline=True, color="primary"),
                    dbc.Button("1M", id="range-1m", size="sm", outline=True, color="primary"),
                    dbc.Button("3M", id="range-3m", size="sm", outline=True, color="primary"),
                    dbc.Button("1Y", id="range-1y", size="sm", outline=True, color="primary", active=True),
                ], className="d-grid gap-2")
            ], className="mt-3")
        ])
    ])

def create_price_chart_card():
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5("Price Prediction", className="d-inline-block me-2"),
                dbc.Badge("Live", color="success", className="me-1"),
            ])
        ]),
        dbc.CardBody([
            dcc.Loading(
                dcc.Graph(
                    id="price-chart", 
                    config={"displayModeBar": "hover"},
                    style={"height": "400px"}
                ),
                type="circle"
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("Prediction Horizon (Days)"),
                    dcc.Slider(
                        id="horizon-slider",
                        min=1, max=30, step=1, value=5,
                        marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=9),
                dbc.Col([
                    html.Label("Confidence"),
                    dbc.RadioItems(
                        id="confidence-level",
                        options=[
                            {"label": "80%", "value": 80},
                            {"label": "90%", "value": 90}
                        ],
                        value=80,
                        inline=True
                    )
                ], md=3)
            ], className="mt-3")
        ])
    ])

def create_indicators_card():
    return dbc.Card([
        dbc.CardHeader("Technical Analysis"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(
                            id="indicators-chart", 
                            config={"displayModeBar": "hover"},
                            style={"height": "350px"}
                        ),
                        type="circle"
                    )
                ], label="Indicators"),
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(
                            id="volume-chart", 
                            config={"displayModeBar": "hover"},
                            style={"height": "350px"}
                        ),
                        type="circle"
                    )
                ], label="Volume"),
                dbc.Tab([
                    dcc.Loading(
                        dcc.Graph(
                            id="momentum-chart", 
                            config={"displayModeBar": "hover"},
                            style={"height": "350px"}
                        ),
                        type="circle"
                    )
                ], label="Momentum")
            ])
        ])
    ])

def create_prediction_card():
    return dbc.Card([
        dbc.CardHeader("Prediction Summary"),
        dbc.CardBody([
            html.Div(id="prediction-summary", children="Select a stock to view predictions")
        ])
    ])

def create_regime_card():
    return dbc.Card([
        dbc.CardHeader("Market Regime"),
        dbc.CardBody([
            html.Div(id="regime-indicator", children="Select a stock to view market regime")
        ])
    ])

def create_recent_predictions_card():
    return dbc.Card([
        dbc.CardHeader("Recent Predictions"),
        dbc.CardBody([
            html.Div(id="recent-predictions", children="Loading recent predictions...")
        ])
    ])

def create_market_heatmap_card():
    return dbc.Card([
        dbc.CardHeader("Market Heatmap"),
        dbc.CardBody([
            dcc.Loading(
                dcc.Graph(
                    id="market-heatmap", 
                    config={"displayModeBar": "hover"},
                    style={"height": "400px"}
                ),
                type="circle"
            )
        ])
    ])

# Main layout
app.layout = dbc.Container([
    create_navbar(),
    dbc.Row([
        dbc.Col([
            create_stock_selector(),
            html.Br(),
            create_prediction_card(),
            html.Br(),
            create_regime_card(),
            html.Br(),
            create_recent_predictions_card()
        ], md=3),
        dbc.Col([
            create_price_chart_card(),
            html.Br(),
            create_indicators_card()
        ], md=9)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            create_market_heatmap_card()
        ])
    ]),
    
    # Data stores
    dcc.Store(id="historical-data"),
    dcc.Store(id="prediction-data"),
    dcc.Store(id="selected-range", data=365),
    
    # Interval for live updates
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute
        n_intervals=0
    )
], fluid=True)

# Helper functions
def safe_api_call(url, method='GET', params=None, json_data=None, timeout=30):
    """Safely make API calls with error handling"""
    try:
        if method == 'GET':
            response = requests.get(url, params=params, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=json_data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {e}")
        return None

# Callbacks
@app.callback(
    Output("stock-dropdown", "options"),
    Input("interval-component", "n_intervals")
)
def update_stock_list(n_intervals):
    """Update the list of available stocks"""
    data = safe_api_call(f"{API_BASE_URL}/tickers")
    
    if not data or "tickers" not in data:
        logger.warning("No tickers found in API response")
        return [{"label": "No tickers available", "value": "none"}]
    
    options = []
    for ticker in data["tickers"]:
        options.append({
            "label": f"{ticker['ticker']} - Last Updated: {ticker.get('last_updated', 'Unknown')}",
            "value": ticker["ticker"]
        })
    
    return options

@app.callback(
    [Output("historical-data", "data"),
     Output("selected-range", "data")],
    [Input("stock-dropdown", "value"),
     Input("range-1d", "n_clicks"),
     Input("range-1w", "n_clicks"),
     Input("range-1m", "n_clicks"),
     Input("range-3m", "n_clicks"),
     Input("range-1y", "n_clicks")]
)
def fetch_historical_data(ticker, *args):
    """Fetch historical data for the selected stock"""
    if not ticker or ticker in ["none", "error"]:
        return None, 365
    
    # Determine time range based on which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        days = 365  # Default to 1 year
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "range-1d":
            days = 1
        elif button_id == "range-1w":
            days = 7
        elif button_id == "range-1m":
            days = 30
        elif button_id == "range-3m":
            days = 90
        else:  # range-1y or default
            days = 365
    
    data = safe_api_call(f"{API_BASE_URL}/historical/{ticker}", params={"days": days})
    
    if not data or "data" not in data:
        logger.error(f"Error fetching historical data for {ticker}")
        return None, days
    
    # Convert to DataFrame for easier processing
    try:
        historical_df = pd.DataFrame(data["data"])
        if not historical_df.empty:
            historical_df["date"] = pd.to_datetime(historical_df["date"])
            return historical_df.to_json(date_format='iso', orient='split'), days
    except Exception as e:
        logger.error(f"Error processing historical data: {e}")
    
    return None, days

@app.callback(
    Output("prediction-data", "data"),
    [Input("stock-dropdown", "value"),
     Input("horizon-slider", "value"),
     Input("confidence-level", "value"),
     Input("interval-component", "n_intervals")]
)
def fetch_prediction_data(ticker, horizon, confidence, n_intervals):
    """Fetch prediction data for the selected stock"""
    if not ticker or ticker in ["none", "error"]:
        return None
    
    data = safe_api_call(
        f"{API_BASE_URL}/predict",
        method='POST',
        json_data={
            "ticker": ticker,
            "horizon": horizon,
            "include_quantiles": True,
            "include_regime": True
        }
    )
    
    if not data:
        logger.error(f"Error fetching prediction for {ticker}")
        return None
    
    return json.dumps(data)

@app.callback(
    Output("price-chart", "figure"),
    [Input("historical-data", "data"),
     Input("prediction-data", "data"),
     Input("theme-switch", "value")]
)
def update_price_chart(historical_json, prediction_json, dark_mode):
    """Update the price chart with historical data and prediction"""
    template = DARK_THEME if dark_mode else LIGHT_THEME
    
    fig = go.Figure()
    
    # Add historical data if available
    if historical_json:
        try:
            historical_df = pd.read_json(historical_json, orient='split')
            historical_df["date"] = pd.to_datetime(historical_df["date"])
            
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df["close"],
                mode="lines",
                name="Historical Price",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="<b>%{x}</b><br>Price: %{y:,.0f}<extra></extra>"
            ))
        except Exception as e:
            logger.error(f"Error processing historical data for chart: {e}")
    
    # Add prediction if available
    if prediction_json:
        try:
            prediction_data = json.loads(prediction_json)
            
            if historical_json:
                historical_df = pd.read_json(historical_json, orient='split')
                last_date = pd.to_datetime(historical_df["date"]).max()
                last_price = historical_df["close"].iloc[-1]
            else:
                last_date = datetime.now()
                last_price = prediction_data.get("latest_price", 0)
            
            horizon = prediction_data["horizon"]
            future_date = last_date + timedelta(days=horizon)
            point_pred = prediction_data["prediction"]
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[last_price, point_pred],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#2ca02c", width=3, dash="dot"),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Predicted Price: %{y:,.0f}<extra></extra>"
            ))
            
            # Add confidence interval if available
            if "quantiles" in prediction_data:
                lower_bound = prediction_data["quantiles"]["p10"]
                upper_bound = prediction_data["quantiles"]["p90"]
                
                fig.add_trace(go.Scatter(
                    x=[future_date, future_date],
                    y=[lower_bound, upper_bound],
                    mode="markers",
                    name="90% Confidence Interval",
                    marker=dict(color="#2ca02c", size=12, symbol="line-ns-open"),
                    hovertemplate="<b>Confidence Interval</b><br>Upper: %{y[1]:,.0f}<br>Lower: %{y[0]:,.0f}<extra></extra>"
                ))
                
        except Exception as e:
            logger.error(f"Error processing prediction data for chart: {e}")
    
    # Update layout
    fig.update_layout(
        title="Stock Price & Prediction",
        xaxis_title="Date",
        yaxis_title="Price (IRR)",
        hovermode="x unified",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    return fig

@app.callback(
    Output("indicators-chart", "figure"),
    [Input("historical-data", "data"),
     Input("theme-switch", "value")]
)
def update_indicators_chart(historical_json, dark_mode):
    """Update the technical indicators chart"""
    template = DARK_THEME if dark_mode else LIGHT_THEME
    
    fig = go.Figure()
    
    if historical_json:
        try:
            historical_df = pd.read_json(historical_json, orient='split')
            historical_df["date"] = pd.to_datetime(historical_df["date"])
            
            # Calculate indicators
            historical_df['SMA20'] = historical_df['close'].rolling(window=20).mean()
            historical_df['SMA50'] = historical_df['close'].rolling(window=50).mean()
            
            # Bollinger Bands
            historical_df['BB_middle'] = historical_df['close'].rolling(window=20).mean()
            stddev = historical_df['close'].rolling(window=20).std()
            historical_df['BB_upper'] = historical_df['BB_middle'] + 2 * stddev
            historical_df['BB_lower'] = historical_df['BB_middle'] - 2 * stddev
            
            # Add price and moving averages
            fig.add_trace(go.Scatter(
                x=historical_df["date"], 
                y=historical_df["close"],
                mode="lines",
                name="Price",
                line=dict(color="#1f77b4", width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_df["date"], 
                y=historical_df["SMA20"],
                mode="lines",
                name="SMA20",
                line=dict(color="#ff7f0e", dash="dot")
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_df["date"], 
                y=historical_df["SMA50"],
                mode="lines",
                name="SMA50",
                line=dict(color="#d62728", dash="dot")
            ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df["BB_upper"],
                mode="lines",
                line=dict(color="#9467bd", width=1),
                name="BB Upper",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df["BB_lower"],
                mode="lines",
                line=dict(color="#9467bd", width=1),
                fill="tonexty",
                fillcolor="rgba(148, 103, 189, 0.1)",
                name="Bollinger Bands"
            ))
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    fig.update_layout(
        title="Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig

@app.callback(
    Output("volume-chart", "figure"),
    [Input("historical-data", "data"),
     Input("theme-switch", "value")]
)
def update_volume_chart(historical_json, dark_mode):
    """Update the volume chart"""
    template = DARK_THEME if dark_mode else LIGHT_THEME
    
    fig = go.Figure()
    
    if historical_json:
        try:
            historical_df = pd.read_json(historical_json, orient='split')
            historical_df["date"] = pd.to_datetime(historical_df["date"])
            
            # Create color array for volume bars
            colors = ['#ef553b' if row['close'] < row['open'] else '#00cc96' 
                     for _, row in historical_df.iterrows()]
            
            fig.add_trace(go.Bar(
                x=historical_df["date"],
                y=historical_df["volume"],
                marker=dict(color=colors),
                name="Volume",
                hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>"
            ))
            
            # Add volume moving average
            historical_df['volume_ma'] = historical_df['volume'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df['volume_ma'],
                line=dict(color='#636efa', width=2),
                name="20D Avg Volume"
            ))
            
        except Exception as e:
            logger.error(f"Error processing volume data: {e}")
    
    fig.update_layout(
        title="Volume Analysis",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig

@app.callback(
    Output("momentum-chart", "figure"),
    [Input("historical-data", "data"),
     Input("theme-switch", "value")]
)
def update_momentum_chart(historical_json, dark_mode):
    """Update the momentum indicators chart"""
    template = DARK_THEME if dark_mode else LIGHT_THEME
    
    fig = go.Figure()
    
    if historical_json:
        try:
            historical_df = pd.read_json(historical_json, orient='split')
            historical_df["date"] = pd.to_datetime(historical_df["date"])
            
            # Calculate RSI
            delta = historical_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            historical_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Add RSI
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df["rsi"],
                line=dict(color="#17becf", width=2),
                name="RSI",
                hovertemplate="<b>%{x}</b><br>RSI: %{y:.1f}<extra></extra>"
            ))
            
            # Add RSI reference lines
            fig.add_shape(
                type="line",
                x0=historical_df["date"].iloc[0],
                y0=70,
                x1=historical_df["date"].iloc[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                name="Overbought"
            )
            
            fig.add_shape(
                type="line",
                x0=historical_df["date"].iloc[0],
                y0=30,
                x1=historical_df["date"].iloc[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                name="Oversold"
            )
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
    
    fig.update_layout(
        title="RSI Momentum Indicator",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig

@app.callback(
    Output("prediction-summary", "children"),
    Input("prediction-data", "data")
)
def update_prediction_summary(prediction_json):
    """Update the prediction summary card"""
    if not prediction_json:
        return html.Div([
            html.I(className="fas fa-chart-line fa-3x text-muted mb-3"),
            html.P("Select a stock to view predictions", className="text-muted")
        ], className="text-center")
    
    try:
        prediction_data = json.loads(prediction_json)
        
        latest_price = prediction_data["latest_price"]
        predicted_price = prediction_data["prediction"]
        change_pct = prediction_data["change_pct"]
        
        # Determine color based on prediction
        color_class = "text-success" if change_pct > 0 else "text-danger"
        icon_class = "fas fa-arrow-up" if change_pct > 0 else "fas fa-arrow-down"
        
        summary_components = [
            html.H4(prediction_data["ticker"], className="text-center mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Small("Current Price", className="text-muted"),
                        html.H5(f"{latest_price:,.0f} IRR", className="mb-0")
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Small(f"Prediction ({prediction_data['horizon']}d)", className="text-muted"),
                        html.H5(f"{predicted_price:,.0f} IRR", className="mb-0")
                    ])
                ], width=6)
            ], className="mb-3"),
            
            html.Div([
                html.I(className=f"{icon_class} me-2"),
                html.Span(f"{change_pct:+.2f}%", className=f"h5 {color_class}")
            ], className="text-center mb-3"),
            
            html.Hr()
        ]
        
        # Add confidence interval if available
        if "confidence_interval" in prediction_data:
            ci = prediction_data["confidence_interval"]
            summary_components.extend([
                html.H6("90% Confidence Interval:", className="text-muted"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Small("Lower Bound", className="text-muted"),
                            html.Div(f"{ci['lower']:,.0f} IRR", style={"fontWeight": "bold"})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Small("Upper Bound", className="text-muted"),
                            html.Div(f"{ci['upper']:,.0f} IRR", style={"fontWeight": "bold"})
                        ])
                    ], width=6)
                ], className="mb-2"),
                html.Div([
                    html.Small("Uncertainty: ", className="text-muted"),
                    html.Span(f"±{ci['width_pct']:.1f}%", style={"fontWeight": "bold"})
                ])
            ])
        
        return html.Div(summary_components)
        
    except Exception as e:
        logger.error(f"Error updating prediction summary: {e}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-2"),
            html.P(f"Error loading prediction data", className="text-muted")
        ], className="text-center")

@app.callback(
    Output("regime-indicator", "children"),
    Input("prediction-data", "data")
)
def update_regime_indicator(prediction_json):
    """Update the market regime indicator"""
    if not prediction_json:
        return html.Div([
            html.I(className="fas fa-chart-area fa-3x text-muted mb-3"),
            html.P("Select a stock to view market regime", className="text-muted")
        ], className="text-center")
    
    try:
        prediction_data = json.loads(prediction_json)
        
        if "regime" not in prediction_data:
            return html.Div([
                html.I(className="fas fa-info-circle fa-2x text-info mb-2"),
                html.P("Regime data not available", className="text-muted")
            ], className="text-center")
        
        regime = prediction_data["regime"]
        dominant = regime["dominant"]
        confidence = regime["confidence"] * 100
        
        # Color and icon based on regime
        regime_config = {
            "Bull": {"color": "success", "icon": "fas fa-arrow-trend-up"},
            "Bear": {"color": "danger", "icon": "fas fa-arrow-trend-down"},
            "Consolidation": {"color": "warning", "icon": "fas fa-arrows-left-right"}
        }
        
        config = regime_config.get(dominant, {"color": "secondary", "icon": "fas fa-question"})
        
        # Create regime distribution pie chart
        regime_dist = regime["distribution"]
        labels = list(regime_dist.keys())
        values = [v * 100 for v in regime_dist.values()]  # Convert to percentages
        
        colors = ['#00cc96', '#ef553b', '#ffa500']  # Green, Red, Orange
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            height=200,
            showlegend=False,
            font=dict(size=10)
        )
        
        return html.Div([
            html.H5("Market Regime Analysis", className="text-center mb-3"),
            
            dbc.Alert([
                html.Div([
                    html.I(className=f"{config['icon']} fa-2x mb-2"),
                    html.H4(dominant, className="mb-1"),
                    html.P(f"Confidence: {confidence:.1f}%", className="mb-0")
                ], className="text-center")
            ], color=config["color"], className="mb-3"),
            
            dcc.Graph(
                figure=fig, 
                config={"displayModeBar": False},
                style={"height": "200px"}
            )
        ])
        
    except Exception as e:
        logger.error(f"Error updating regime indicator: {e}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-2"),
            html.P("Error loading regime data", className="text-muted")
        ], className="text-center")

@app.callback(
    Output("market-heatmap", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("theme-switch", "value")]
)
def update_market_heatmap(n_intervals, dark_mode):
    """Update the market heatmap with top stocks"""
    template = DARK_THEME if dark_mode else LIGHT_THEME
    
    # For demonstration, create sample heatmap data
    # In production, this would fetch real market data
    
    try:
        sectors = ["Technology", "Finance", "Energy", "Industrial", "Healthcare"]
        tickers = [
            "TEDPIX", "KHODRO", "FOLD", "VBMELLAT", "SHBANDAR",
            "PICICO", "BMELLAT", "SHPARS", "IRAN", "SAIPA"
        ]
        
        # Generate sample data (in production, fetch from API)
        np.random.seed(42)  # For consistent demo data
        data = []
        
        for i, ticker in enumerate(tickers):
            data.append({
                "ticker": ticker,
                "sector": sectors[i % len(sectors)],
                "change_pct": np.random.normal(0, 3),
                "market_cap": np.random.uniform(1e10, 1e12)  # Market cap in IRR
            })
        
        df = pd.DataFrame(data)
        
        # Create treemap
        fig = px.treemap(
            df,
            path=['sector', 'ticker'],
            values='market_cap',
            color='change_pct',
            color_continuous_scale=['#ef553b', '#ffa500', '#00cc96'],  # Red, Orange, Green
            color_continuous_midpoint=0,
            title="Market Overview by Sector and Stock",
            hover_data={'change_pct': ':.2f%'}
        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            template=template,
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating market heatmap: {e}")
        return go.Figure(layout=dict(
            title="Error loading market data",
            template=template,
            height=400
        ))

@app.callback(
    Output("recent-predictions", "children"),
    Input("interval-component", "n_intervals")
)
def update_recent_predictions(n_intervals):
    """Update the recent predictions list"""
    try:
        # In production, this would fetch from a database or API
        # For demo, create sample recent predictions
        predictions = [
            {"ticker": "TEDPIX", "change": 2.3, "accuracy": 0.82, "time": "2 hours ago"},
            {"ticker": "KHODRO", "change": -1.5, "accuracy": 0.76, "time": "3 hours ago"},
            {"ticker": "VBMELLAT", "change": 0.8, "accuracy": 0.91, "time": "4 hours ago"},
            {"ticker": "FOLD", "change": 3.1, "accuracy": 0.88, "time": "5 hours ago"},
            {"ticker": "SHBANDAR", "change": -0.7, "accuracy": 0.79, "time": "6 hours ago"}
        ]
        
        items = []
        for pred in predictions:
            color_class = "text-success" if pred["change"] > 0 else "text-danger"
            icon_class = "fas fa-arrow-up" if pred["change"] > 0 else "fas fa-arrow-down"
            
            items.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span(pred["ticker"], style={"fontWeight": "bold"}),
                                html.Br(),
                                html.Small(pred["time"], className="text-muted")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.I(className=f"{icon_class} me-1"),
                                html.Span(f"{pred['change']:+.1f}%", className=color_class),
                                html.Br(),
                                html.Small(f"Acc: {pred['accuracy']:.0%}", className="text-muted")
                            ], className="text-end")
                        ], width=6)
                    ])
                ], className="py-2")
            )
        
        return html.Div([
            html.H6("Recent Predictions", className="mb-3"),
            dbc.ListGroup(items, flush=True)
        ])
        
    except Exception as e:
        logger.error(f"Error updating recent predictions: {e}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-2"),
            html.P("Error loading recent predictions", className="text-muted")
        ], className="text-center")

# Run server
if __name__ == "__main__":
    import os
    
    # Configuration
    debug_mode = os.getenv("DEBUG", "True").lower() == "true"
    port = int(os.getenv("DASHBOARD_PORT", 8050))
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    
    app.run_server(
        debug=debug_mode, 
        host=host, 
        port=port,
        dev_tools_hot_reload=debug_mode
    )
