from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faicons as fa
import os
from sklearn.cluster import KMeans
import joblib 
from typing import Any, cast 

# =============================================================================
# 0. FAST MODEL LOADING
# =============================================================================

print("Initializing Prediction Model...")
rf_model: Any = None 
le_weather: Any = None
unique_weather_options = ["Clear", "Rain", "Snow", "Fog", "Overcast"] 

model_save_path = "accident_model_data.joblib"

# Try to Load Existing Model
if os.path.exists(model_save_path):
    try:
        print(f"Loading model from {model_save_path}...")
        saved_data = joblib.load(model_save_path)
        
        # Restore Model and Encoder
        rf_model = saved_data['model']
        le_weather = saved_data['encoder']
        
        # Restore Weather Options
        weather_classes = getattr(le_weather, 'classes_', [])
        unique_weather_options = sorted([str(x) for x in weather_classes])
        print("Model Loaded Successfully.")
        
    except Exception as e:
        print(f"Error loading saved model: {e}")
else:
    print(f"Error: {model_save_path} not found. Predictions will not work.")

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Load data
file_path = "us_accidents_ca_only.csv"
print(f"Loading data from {file_path}...")

if os.path.exists(file_path):
    try:
        # Optimization: Parse dates directly on read
        df = pd.read_csv(file_path, parse_dates=['Start_Time'])
        
        # Extract time components
        df['Year'] = df['Start_Time'].dt.year
        df['Month'] = df['Start_Time'].dt.month_name()
        df['DayOfWeek'] = df['Start_Time'].dt.day_name()
        df['Hour'] = df['Start_Time'].dt.hour
        
        print("Data loaded successfully.")

    except Exception as e:
        print(f"Error processing data: {e}")
        df = pd.DataFrame()
else:
    print("Warning: CSV file not found.")
    df = pd.DataFrame()

# Fallback structure
if df.empty:
    df = pd.DataFrame({
        'Start_Time': pd.Series(dtype='datetime64[ns]'),
        'Severity': pd.Series(dtype='int'),
        'Start_Lat': pd.Series(dtype='float'),
        'Start_Lng': pd.Series(dtype='float'),
        'Year': pd.Series(dtype='int'),
        'Month': pd.Series(dtype='object'),
        'DayOfWeek': pd.Series(dtype='object'),
        'Hour': pd.Series(dtype='int')
    })

# Orders
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# =============================================================================
# 2. UI LAYOUT
# =============================================================================

app_ui = ui.page_sidebar(
    # --- Sidebar ---
    ui.sidebar(
        ui.h4("Filters", class_="mb-3"),
        ui.p("Apply filters to Analytics tab", class_="small text-muted"),
        ui.hr(),
        
        # Filter 1: Year Range
        ui.input_slider("filter_year", "Year Range", min=2016, max=2023, value=[2016, 2023], sep=""),
        
        # Filter 2: Severity Level
        ui.input_checkbox_group(
            "filter_severity", "Severity Level",
            choices={"1": "1 - Minor", "2": "2 - Moderate", "3": "3 - Serious", "4": "4 - Severe"},
            selected=["1", "2", "3", "4"]
        ),
        
        # Filter 3: Weather Condition
        ui.input_selectize("filter_weather", "Weather Condition", choices=[], multiple=True, options={"placeholder": "All Conditions"}),
        
        # Filter 4: Time of Day
        ui.input_slider("filter_hour", "Hour of Day", min=0, max=23, value=[0, 23], step=1),
        
        ui.hr(),
        ui.input_action_button("update_btn", "Update Dashboard", class_="btn-primary w-100 mb-3"),
        ui.input_action_button("reset_btn", "Reset All Filters", class_="btn-danger w-100"),
        bg="transparent"
    ),

     # Custom CSS
    ui.head_content(ui.tags.style("""
        /* GLOBAL THEME */
        body, .bslib-page-sidebar {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d1b1b 25%, #3d2020 50%, #2d1b1b 75%, #1a1a1a 100%) !important;
            background-attachment: fixed !important;
        }
        .container-fluid, main { background: transparent !important; }

        /* HEADER */
        h2 {
            color: #fff !important; font-weight: 800 !important;
            text-shadow: 0 0 20px rgba(255, 87, 34, 0.5), 0 0 40px rgba(255, 193, 7, 0.3) !important;
            font-size: 2rem !important;
        }
        .btn-about {
            background: linear-gradient(135deg, #FF5722 0%, #FF9800 100%) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important; font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(255, 87, 34, 0.4) !important;
            border-radius: 10px !important;
        }

        /* VALUE BOXES */
        .bslib-value-box .value-box-showcase svg { fill: white !important; color: white !important; }
        .value-box-area { padding: 15px; }
        .bslib-value-box { 
            border-radius: 20px !important;
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(20px) !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }
        .bslib-value-box .value-box-title { color: rgba(255, 255, 255, 0.9) !important; }
        .bslib-value-box .value-box-value { color: #fff !important; font-weight: 800 !important; }

        /* CARDS */
        .card {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(20px) !important;
            border: 2px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }
        .card-header {
            background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(255, 152, 0, 0.15) 100%) !important;
            border-bottom: 2px solid rgba(255, 152, 0, 0.3) !important;
            color: #fff !important; font-weight: 700 !important;
        }
        .nav-tabs { border-bottom: 2px solid rgba(255, 152, 0, 0.3) !important; }
        .nav-tabs .nav-link.active {
            color: #fff !important;
            background: linear-gradient(135deg, #FF5722 0%, #FF9800 100%) !important;
        }
        .card .card-body { padding: 0 !important; overflow: hidden !important; }

        /* PREDICTOR STYLES */
        .pred-container {
            background: linear-gradient(135deg, rgba(255, 87, 34, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%) !important;
            border-radius: 25px !important; padding: 30px !important;
            border: 2px solid rgba(255, 152, 0, 0.2) !important;
        }
        .input-card {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(15px) !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 18px !important;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(20px) !important;
            border: 2px solid rgba(255, 152, 0, 0.3) !important;
            border-radius: 20px !important;
            padding: 30px !important;
        }
        .sev-number {
            font-size: 6rem !important; font-weight: 900 !important;
            background: linear-gradient(45deg, #FF5722, #FF9800, #FFC107) !important;
            -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
        }
        .btn-predict {
            background: linear-gradient(135deg, #FF5722 0%, #FF9800 50%, #FFC107 100%) !important;
            border: none !important; color: white !important;
            padding: 15px !important; font-weight: 700 !important;
            border-radius: 12px !important;
        }
        
        /* SIDEBAR & CONTROLS */
        aside, .sidebar {
            background: linear-gradient(135deg, #D32F2F 0%, #F57C00 50%, #FFA000 100%) !important;
            border-radius: 0 25px 25px 0 !important;
            min-height: 100vh !important;
        }
        aside .btn-primary, .sidebar .btn-primary, aside .btn-danger, .sidebar .btn-danger {
            background: rgba(255, 255, 255, 0.25) !important;
            border: 3px solid rgba(255, 255, 255, 0.6) !important;
            color: white !important; font-weight: 800 !important;
        }
        
        /* MODALS */
        .modal-content {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d1b1b 100%) !important;
            border: 2px solid rgba(255, 152, 0, 0.3) !important;
        }
        .modal-title, .modal-body strong { color: #fff !important; }
        .modal-body, .modal-body p { color: rgba(255, 255, 255, 0.9) !important; }
        
        /* TEXT VISIBILITY FIXES */
        .card p, .card .small, .card li, .card label { color: rgba(255, 255, 255, 0.85) !important; }
        .pred-container select, .pred-container input {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #fff !important; border: 2px solid rgba(255, 152, 0, 0.4) !important;
        }
        .pred-container select option { background: #2d1b1b !important; color: #fff !important; }
    """)),

    # --- Header ---
    ui.div(
        ui.h2("California Road Accidents Dashboard", class_="mb-0"),
        ui.div(
            ui.input_action_button("narrative_btn", "Narrative", icon=fa.icon_svg("book-open"), class_="btn-about me-2"),
            ui.input_action_button("about_btn", "About", icon=fa.icon_svg("circle-info"), class_="btn-about"),
            class_="d-flex"
        ),
        class_="d-flex justify-content-between align-items-center mb-3 mt-2"
    ),

    # --- KPI Row ---
    ui.layout_columns(
        ui.value_box("Total Accidents", ui.output_text("kpi_total"), showcase=fa.icon_svg("car-burst"), theme="bg-gradient-red-orange", full_screen=False, fill=False),
        ui.value_box("Avg Accidents / Day", ui.output_text("kpi_daily_avg"), showcase=fa.icon_svg("calendar-day"), theme="bg-gradient-blue-indigo", full_screen=False, fill=False),
        ui.value_box("Avg Cases / Hour", ui.output_text("kpi_hourly_avg"), showcase=fa.icon_svg("stopwatch"), theme="bg-gradient-teal-green", full_screen=False, fill=False),
        fill=False, class_="value-box-area"
    ),

    # --- Tabs ---
    ui.navset_card_tab(
        
        # TAB 1: Analytics
        ui.nav_panel("Analytics",
            ui.layout_columns(
                ui.card(ui.card_header("Geospatial Severity Visualization"), output_widget("map_plot"), full_screen=True),
                ui.card(ui.card_header("Impact of Road Features (POIs)"), output_widget("poi_plot"), full_screen=True),
                col_widths=[6, 6]
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Accidents by Day of Week"), output_widget("day_plot")),
                ui.card(ui.card_header("Accidents by Month"), output_widget("month_plot")),
                col_widths=[6, 6]
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Accident Frequency by Hour of Day"), output_widget("hour_plot"), full_screen=True),
                ui.card(ui.card_header("Accidents by Weather Condition"), output_widget("weather_plot"), full_screen=True),
                col_widths=[6, 6]
            )
        ),

        # TAB 2: Hotspot Analysis
        ui.nav_panel("Hotspot Clustering",
            ui.layout_columns(
                ui.card(ui.card_header("Cluster Configuration"),
                    ui.div(
                        ui.input_slider("n_clusters", "Number of Zones (K)", min=2, max=10, value=5),
                        ui.hr(),
                        ui.h6("How to Read this Map", class_="fw-bold text-primary"),
                        ui.p("This map uses K-Means to find 'centers of gravity' for accidents.", class_="small text-muted"),
                        ui.tags.ul(
                            ui.tags.li(ui.strong("Colored Zones:"), " Distinct geographic clusters.", class_="small"),
                            ui.tags.li(ui.strong("Resource Allocation:"), " Place HQs in center of zones.", class_="small"),
                        ),
                        class_="p-4 h-100", style="background: #2d1b1b;"
                    ), height="100%"
                ),
                ui.card(ui.card_header("Identified Accident Hotspots"), output_widget("cluster_map", width="100%", height="600px"), full_screen=True, style="padding: 0px;"),
                col_widths=[3, 9] 
            )
        ),

        # TAB 3: Severity Predictor
        ui.nav_panel("Severity Predictor",
            ui.div(
                ui.row(
                    ui.column(4,
                        ui.div(
                            ui.h4("Accident Scenario", class_="mb-3 text-primary fw-bold"),
                            ui.div(
                                ui.h6(fa.icon_svg("road"), " Road Infrastructure", class_="fw-bold text-muted"),
                                ui.div(
                                    ui.input_switch("pred_signal", "Traffic Signal", False),
                                    ui.input_switch("pred_junction", "Junction Area", False),
                                    ui.input_switch("pred_crossing", "Pedestrian Crossing", False),
                                    class_="d-flex flex-wrap gap-3"
                                ), class_="p-3 mb-3 input-card"
                            ),
                            ui.div(
                                ui.h6(fa.icon_svg("cloud-sun"), " Environmental Factors", class_="fw-bold text-muted"),
                                ui.input_select("pred_weather", "Weather Condition", choices=unique_weather_options, selected="Fair"),
                                ui.input_slider("pred_hour", "Hour of Day (24h)", 0, 23, 17, step=1),
                                class_="p-3 mb-3 input-card"
                            ),
                            ui.div(
                                ui.h6(fa.icon_svg("temperature-half"), " Atmospheric Physics", class_="fw-bold text-muted"),
                                ui.layout_columns(ui.input_numeric("pred_temp", "Temp (F)", 70), ui.input_numeric("pred_hum", "Humidity (%)", 50)),
                                class_="p-3 mb-3 input-card"
                            ),
                            ui.input_action_button("predict_btn", "Analyze Severity Risk", class_="btn-predict w-100"),
                            class_="h-100"
                        )
                    ),
                    ui.column(8,
                        ui.div(
                            ui.div(
                                ui.span("PREDICTED SEVERITY LEVEL", class_="sev-label"),
                                ui.output_ui("prediction_text"), 
                                ui.p("Severity ranges from 1 (Low Impact) to 4 (High Impact)", class_="small text-muted mt-2"),
                                class_="result-card mb-3"
                            ),
                            ui.card(ui.card_header("Confidence Analysis"), output_widget("pred_prob_plot", height="400px"), full_screen=False, class_="h-100"),
                        )
                    )
                ), class_="pred-container"
            )
        ),
    )
)


# =============================================================================
# 3. SERVER LOGIC
# =============================================================================

def server(input, output, session):

    # Dynamic Weather Filter
    @reactive.effect
    def _():
        if not df.empty and 'Weather_Condition' in df.columns:
            weather_options = sorted(df['Weather_Condition'].dropna().unique().tolist())
            ui.update_selectize("filter_weather", choices=weather_options)

    # Prediction Logic
    pred_result: Any = reactive.Value(None) 
    pred_probs: Any = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def compute_prediction():
        if rf_model is None:
            return

        w_encoded = 0
        try:
            if le_weather:
                weather_classes = [str(x) for x in getattr(le_weather, 'classes_', [])]
                if str(input.pred_weather()) in weather_classes:
                    w_encoded = le_weather.transform([str(input.pred_weather())])[0]
        except Exception: 
            w_encoded = 0

        input_data = pd.DataFrame({
            'Hour': [input.pred_hour()],
            'Weather_Encoded': [w_encoded],
            'Temperature(F)': [input.pred_temp()],
            'Humidity(%)': [input.pred_hum()],
            'Traffic_Signal': [int(input.pred_signal())], 
            'Junction': [int(input.pred_junction())],
            'Crossing': [int(input.pred_crossing())]
        })

        try:
            raw_prediction = rf_model.predict(input_data)[0]
            prediction = int(raw_prediction) 
            probabilities = rf_model.predict_proba(input_data)[0]
            pred_result.set(prediction)
            pred_probs.set(probabilities)
        except Exception as e:
            print(f"Prediction Error: {e}")

    @render.ui
    def prediction_text():
        val = pred_result.get()
        if val is None: return ui.div("?", class_="sev-number")
        return ui.div(str(val), class_="sev-number")

    @render_widget
    def pred_prob_plot():
        probs = pred_probs.get()
        if probs is None or rf_model is None:
            fig = go.Figure()
            fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                annotations=[{'text': "Adjust inputs and click Analyze", 'showarrow': False, 'font': {'size': 20, 'color': 'rgba(255,255,255,0.5)'}}])
            return go.FigureWidget(fig)
        
        classes = rf_model.classes_
        colors = ['#00b894', '#fdcb6e', '#e17055', '#d63031']
        fig = go.Figure(data=[go.Pie(labels=[f"Severity {c}" for c in classes], values=probs, hole=.65, 
            marker=dict(colors=colors[:len(classes)], line=dict(color='#2d1b1b', width=4)),
            textinfo='label+percent', textposition='outside', hoverinfo='label+percent+value', textfont=dict(color='white', size=13),
            pull=[0.05 if p == max(probs) else 0 for p in probs])])
        
        max_prob = max(probs)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f"<span style='font-size:40px; font-weight:bold; color:white;'>{max_prob*100:.0f}%</span><br><span style='font-size:16px; color:#ccc;'>CONFIDENCE</span>", x=0.5, y=0.5, showarrow=False)])
        return go.FigureWidget(fig)

    # Filtered Data
    @reactive.calc
    @reactive.event(input.update_btn, input.reset_btn, ignore_none=False, ignore_init=False)
    def filtered_df() -> pd.DataFrame:
        data = df.copy()
        if data.empty: return data
        
        # Year
        yr = input.filter_year()
        if yr[0] != 2016 or yr[1] != 2023: data = data[(data['Year'] >= yr[0]) & (data['Year'] <= yr[1])]
        
        # Severity
        sev = input.filter_severity()
        if sev and len(sev) < 4: data = data[data['Severity'].isin([int(s) for s in sev])]
        
        # Weather
        w = input.filter_weather()
        if w: data = data[data['Weather_Condition'].isin(w)]
        
        # Hour
        hr = input.filter_hour()
        if hr[0] != 0 or hr[1] != 23: data = data[(data['Hour'] >= hr[0]) & (data['Hour'] <= hr[1])]
        
        return cast(pd.DataFrame, data)

    # Reset
    @reactive.effect
    @reactive.event(input.reset_btn)
    def reset_filters():
        ui.update_slider("filter_year", value=(2016, 2023))
        ui.update_checkbox_group("filter_severity", selected=["1", "2", "3", "4"])
        ui.update_selectize("filter_weather", selected=[])
        ui.update_slider("filter_hour", value=(0, 23))

    # KPIs
    @render.text
    def kpi_total(): return f"{len(filtered_df()):,}"

    @render.text
    def kpi_daily_avg():
        d = filtered_df()
        if d.empty: return "0"
        n_days = max((d['Start_Time'].max() - d['Start_Time'].min()).days, 1)
        return f"{len(d)/n_days:.2f}"

    @render.text
    def kpi_hourly_avg():
        d = filtered_df()
        if d.empty: return "0"
        n_days = max((d['Start_Time'].max() - d['Start_Time'].min()).days, 1)
        return f"{len(d)/(n_days*24):.2f}"

    # Plots
    @render_widget
    def map_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        map_d = d.sample(2500) if len(d) > 2500 else d
        fig = px.scatter_mapbox(map_d, lat="Start_Lat", lon="Start_Lng", color="Severity", zoom=5, center={"lat": 36.7783, "lon": -119.4179}, color_continuous_scale=px.colors.sequential.OrRd, mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, uirevision='constant')
        return fig

    @render_widget
    def poi_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        cols = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Station', 'Amenity', 'Bump', 'Give_Way', 'No_Exit', 'Roundabout']
        res = [{'Feature': c.replace('_',' '), 'Label': f"{c.replace('_',' ')} ({(d[c].sum()/len(d)*100):.2f}%)", 'Count': d[c].sum()} for c in cols if c in d.columns and d[c].sum() > 0]
        df_poi = pd.DataFrame(res).sort_values('Count')
        fig = px.bar_polar(df_poi, r="Count", theta="Label", color="Feature", color_discrete_sequence=px.colors.qualitative.Bold, template="plotly_white")
        fig.update_layout(showlegend=True, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(showticklabels=False)), height=None)
        return fig

    @render_widget
    def day_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        c = d['DayOfWeek'].value_counts().reindex(day_order)
        fig = go.Figure(go.Bar(x=c.index, y=c.values, marker=dict(color=c.values, colorscale=[[0, '#2ecc71'], [0.5, '#f1c40f'], [1, '#e74c3c']])))
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=None, plot_bgcolor='white')
        return fig

    @render_widget
    def month_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        c = d['Month'].value_counts().reindex(month_order)
        colors = ["#a9cce3", "#5dade2", "#a2d9ce", "#73c6b6", "#45b39d", "#f7dc6f", "#f0b27a", "#e59866", "#d35400", "#a04000", "#6e2c00", "#2e86c1"]
        fig = go.Figure(go.Bar(x=c.index, y=c.values, marker=dict(color=colors)))
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=None, plot_bgcolor='white')
        return fig

    @render_widget
    def hour_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        c = d['Hour'].value_counts().sort_index()
        colors = ["#2c3e50"]*5 + ["#7f8c8d"]*2 + ["#f0b27a"]*2 + ["#f9e79f"]*3 + ["#f1c40f"]*4 + ["#d35400"]*3 + ["#2e4053"]*5
        fig = go.Figure(go.Bar(x=c.index, y=c.values, marker=dict(color=colors[:len(c)]), name="Hourly Volume"))
        fig.add_trace(go.Scatter(x=c.index, y=c.values, mode='lines+markers', line=dict(color='rgba(100,110,120,0.7)', width=3)))
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=None, plot_bgcolor='white', barmode='overlay', hovermode="x unified")
        return fig

    @render_widget
    def weather_plot():
        d = filtered_df()
        if d.empty: return go.Figure()
        w = d['Weather_Condition'].value_counts().head(15).sort_values()
        fig = go.Figure(go.Bar(x=w.values, y=w.index, orientation='h', marker=dict(color='#87CEEB')))
        fig.update_layout(margin=dict(l=10, r=0, t=20, b=10), height=None, plot_bgcolor='white')
        return fig

    @render_widget
    def cluster_map():
        d = df.sample(5000, random_state=42) if len(df) > 5000 else df
        if d.empty: return go.Figure()
        k = input.n_clusters()
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(d[['Start_Lat', 'Start_Lng']])
        d = d.copy()
        d['Cluster Name'] = [f"Zone {i+1}" for i in labels]
        fig = px.scatter_mapbox(d, lat="Start_Lat", lon="Start_Lng", color="Cluster Name", category_orders={"Cluster Name": [f"Zone {i+1}" for i in range(k)]}, zoom=4, center={"lat": 37.0, "lon": -119.5}, color_discrete_sequence=px.colors.qualitative.G10, mapbox_style="carto-positron")
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0}, uirevision='constant')
        return fig

    # Modals
    @reactive.effect
    @reactive.event(input.narrative_btn)
    def show_narrative_modal():
        ui.modal_show(ui.modal(ui.div(ui.h4("Descriptive Narrative", class_="mb-3"), ui.p("Analysis text goes here...", class_="text-muted")), title="Project Analysis", size="xl", easy_close=True))

    @reactive.effect
    @reactive.event(input.about_btn)
    def show_about_modal():
        ui.modal_show(ui.modal(ui.div(ui.h4("Research Authors", class_="mb-3"), ui.p("Author info goes here...", class_="text-muted")), title="About this Project", size="xl", easy_close=True))

static_dir = os.path.join(os.path.dirname(__file__), "assets")
app = App(app_ui, server, static_assets=static_dir)