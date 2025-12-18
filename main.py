from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faicons as fa
import os
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from typing import Any, cast 

# =============================================================================
# 0. GLOBAL MODEL TRAINING (Runs once on startup)
# =============================================================================

print("Initializing Prediction Model...")
rf_model: Any = None # <--- CHANGE THIS LINE (Added : Any)
le_weather = LabelEncoder()
unique_weather_options = ["Clear", "Rain", "Snow", "Fog", "Overcast"] # Default fallback

try:
    if os.path.exists("us_accidents_ca_only.csv"):
        # Load necessary columns for training
        train_cols = ['Severity', 'Start_Time', 'Weather_Condition', 'Temperature(F)', 
                    'Humidity(%)', 'Traffic_Signal', 'Junction', 'Crossing']
        
        # Read sample (10k rows is enough for a demo)
        df_train = pd.read_csv("us_accidents_ca_only.csv", usecols=lambda c: c in train_cols)
        df_train = df_train.sample(n=min(10000, len(df_train)), random_state=42)
        
        # Preprocessing
        df_train['Start_Time'] = pd.to_datetime(df_train['Start_Time'], errors='coerce')
        df_train['Hour'] = df_train['Start_Time'].dt.hour
        df_train['Weather_Condition'] = df_train['Weather_Condition'].fillna('Clear')
        df_train['Temperature(F)'] = df_train['Temperature(F)'].fillna(60)
        df_train['Humidity(%)'] = df_train['Humidity(%)'].fillna(50)
        
        # Encode Weather
        df_train['Weather_Encoded'] = le_weather.fit_transform(df_train['Weather_Condition'].astype(str))
        
        # [FIX] Safely access classes_ using getattr to satisfy strict type checker
        # This defaults to [] if classes_ is missing, preventing the "None" error
        weather_classes = getattr(le_weather, 'classes_', [])
        unique_weather_options = sorted([str(x) for x in weather_classes])
        
        # Features & Target
        X = df_train[['Hour', 'Weather_Encoded', 'Temperature(F)', 'Humidity(%)', 
                    'Traffic_Signal', 'Junction', 'Crossing']]
        y = df_train['Severity']
        
        # Train Model
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X, y)
        print("Prediction Model Trained Successfully.")
except Exception as e:
    print(f"Model training failed: {e}")

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

# Load data
# We check if file exists first to avoid generic errors
file_path = "us_accidents_ca_only.csv"

if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # --- Preprocessing ---
        # Convert Start_Time to datetime
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        
        # Extract time components for analysis
        df['Year'] = df['Start_Time'].dt.year
        df['Month'] = df['Start_Time'].dt.month_name()
        
        # Order for months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        df['DayOfWeek'] = df['Start_Time'].dt.day_name()
        
        # Order for days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        df['Hour'] = df['Start_Time'].dt.hour
        
        print("Data loaded successfully.")

    except Exception as e:
        print(f"Error processing data: {e}")
        # Create an empty dataframe with correct types if loading fails
        df = pd.DataFrame()
else:
    print("Warning: CSV file not found. Loading empty dashboard.")
    df = pd.DataFrame()

# Fallback if df is empty (prevents crashing if file missing)
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


# =============================================================================
# 2. UI LAYOUT
# =============================================================================

app_ui = ui.page_sidebar(
   

    # --- Sidebar ---
    ui.sidebar(
        ui.h4("Filters", class_="mb-3"),
        ui.p("Apply filters to Map and Time Analysis tab", class_="small text-muted"),
        
        ui.hr(),
        
        # Filter 1: Year Range
        ui.input_slider(
            "filter_year",
            "Year Range",
            min=2016,
            max=2023,
            value=[2016, 2023],
            sep=""
        ),
        
        # Filter 2: Severity Level
        ui.input_checkbox_group(
            "filter_severity",
            "Severity Level",
            choices={"1": "1 - Minor", "2": "2 - Moderate", "3": "3 - Serious", "4": "4 - Severe"},
            selected=["1", "2", "3", "4"]
        ),
        
        # Filter 3: Weather Condition
        ui.input_selectize(
            "filter_weather",
            "Weather Condition",
            choices=[],  # Will be populated dynamically
            multiple=True,
            options={"placeholder": "All Weather Conditions"}
        ),
        
        # Filter 4: Time of Day
        ui.input_slider(
            "filter_hour",
            "Hour of Day",
            min=0,
            max=23,
            value=[0, 23],
            step=1
        ),
        
        ui.hr(),
        
        ui.input_action_button("reset_btn", "Reset All Filters", class_="btn-danger w-100"),
        
        bg="#f8f9fa"
    ),

     # [CHANGE 1] Add Custom CSS for aesthetic shadows and gradients
    ui.head_content(ui.tags.style("""
        .value-box-area { padding: 10px; }
        .bslib-value-box { 
            border-radius: 12px !important; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
            border: none !important;
            transition: transform 0.2s;
        }
        .bslib-value-box:hover { transform: translateY(-5px); }
        .card { 
            border: none !important; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important; 
            border-radius: 15px !important;
        }
        /* --- NEW CSS FOR ABOUT SECTION --- */
        .btn-about {
            background-color: white;
            border: 1px solid #ddd;
            color: #333;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .btn-about:hover {
            background-color: #f8f9fa;
            border-color: #ccc;
        }
        .author-card {
            background: #fff;
            border: 1px solid #eee;
            border-radius: 8px;
            overflow: hidden; 
            height: 100%;
        }
        .author-header {
            padding: 10px;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        .author-body {
            padding: 20px;
            text-align: center;
        }
        /* NEW: Styles for the Author Images */
        .author-photo {
            width: 100px; 
            height: 100px; 
            border-radius: 50%; 
            object-fit: cover; /* Ensures image doesn't stretch */
            border: 4px solid white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            margin: 0 auto 15px auto;
            display: block;
        }

        /* --- SEVERITY PREDICTOR STYLES --- */
        .pred-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 20px;
            padding: 20px;
        }
        .input-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        }
        .result-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .sev-number {
            font-size: 5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FF512F, #DD2476);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sev-label {
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .btn-predict {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            border: none;
            color: white;
            padding: 12px;
            font-weight: bold;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(24, 40, 72, 0.3);
        }

    """)),
    # --- Main Header Area with About Button ---
    ui.div(
        ui.h2("California Road Accidents Dashboard", class_="mb-0"),
        ui.input_action_button(
            "about_btn", 
            "About", 
            icon=fa.icon_svg("circle-info"), 
            class_="btn-about"
        ),
        class_="d-flex justify-content-between align-items-center mb-3 mt-2"
    ),

   # --- KPI Row ---
    ui.layout_columns(
        ui.value_box(
            "Total Accidents",
            ui.output_text("kpi_total"),
            showcase=fa.icon_svg("car-burst"),
            theme="bg-gradient-red-orange", # Gradient theme
            full_screen=False,
            fill=False
        ),
        ui.value_box(
            "Avg Accidents / Day",
            ui.output_text("kpi_daily_avg"),
            showcase=fa.icon_svg("calendar-day"),
            theme="bg-gradient-blue-indigo", # Gradient theme
            full_screen=False,
            fill=False
        ),
        ui.value_box(
            "Avg Cases / Hour",
            ui.output_text("kpi_hourly_avg"),
            showcase=fa.icon_svg("stopwatch"), # Changed icon to stopwatch
            theme="bg-gradient-teal-green", # Gradient theme
            full_screen=False,
            fill=False
        ),
        fill=False,
        class_="value-box-area" # Applies our custom padding
    ),

    # --- Tabs ---
    ui.navset_card_tab(
        
        # TAB 1: Map and Time Analysis
        ui.nav_panel(
            "Map and Time Analysis",
            
            # Row 1: Map (Left) and POI Pies (Right)
            ui.layout_columns(
                ui.card(
                    ui.card_header("Geospatial Severity Visualization"),
                    output_widget("map_plot"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Impact of Road Features (POIs)"),
                    output_widget("poi_plot"),
                    full_screen=True
                ),
                col_widths=[6, 6]
            ),

            # Row 2: Day of Week (Left) and Monthly Trend (Right)
            ui.layout_columns(
                ui.card(
                    ui.card_header("Accidents by Day of Week"),
                    output_widget("day_plot")
                ),
                ui.card(
                    ui.card_header("Accidents by Month"),
                    output_widget("month_plot")
                ),
                col_widths=[6, 6]
            ),
            
            # Row 3: Hourly Trend
            # Row 3: Hourly Trend and Weather Distribution
            ui.layout_columns(
                ui.card(
                    ui.card_header("Accident Frequency by Hour of Day"),
                    output_widget("hour_plot"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Accidents by Weather Condition"),
                    output_widget("weather_plot"),
                    full_screen=True
                ),
                col_widths=[6, 6]
            )
        ),

        # TAB 2: Hotspot Analysis (Updated)
        ui.nav_panel(
            "Hotspot Clustering",
            ui.layout_columns(
                # Left Sidebar: Controls & Interpretation
                ui.card(
                    ui.card_header("Cluster Configuration"),
                    ui.input_slider("n_clusters", "Number of Zones (K)", min=2, max=10, value=5),
                    
                    ui.hr(),
                    
                    # New Interpretation Section for Non-Technical Users
                    ui.h6("How to Read this Map", class_="fw-bold text-primary"),
                    ui.p("This map uses an algorithm called K-Means to mathematically find 'centers of gravity' for accidents.", class_="small text-muted"),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Colored Zones:"), " Each color represents a distinct geographic cluster of accidents.", class_="small"),
                        ui.tags.li(ui.strong("Resource Allocation:"), " Agencies can place 1 HQ or Response Team in the center of each color to minimize travel time.", class_="small"),
                        ui.tags.li(ui.strong("K-Slider:"), " Change the slider to split the state into more specific local zones.", class_="small"),
                    ),
                    height="100%"
                ),
                
                # Right Side: The Map
                ui.card(
                    ui.card_header("Identified Accident Hotspots"),
                    # Increased height to 700px for better visibility
                    output_widget("cluster_map", width="100%", height="600px"),
                    full_screen=True,
                    style="padding: 0px;" # Removes padding so map touches edges
                ),
                col_widths=[3, 9] 
            )
        ),

        # [CHANGE] Modern Severity Prediction Tab
        ui.nav_panel(
            "Severity Predictor",
            ui.div(
                ui.row(
                    # --- Left Column: Inputs (Organized by correlation importance) ---
                    ui.column(4,
                        ui.div(
                            ui.h4("Accident Scenario", class_="mb-3 text-primary fw-bold"),
                            
                            # Group 1: Infrastructure (High Importance)
                            ui.div(
                                ui.h6(fa.icon_svg("road"), " Road Infrastructure", class_="fw-bold text-muted"),
                                ui.div(
                                    ui.input_switch("pred_signal", "Traffic Signal", False),
                                    ui.input_switch("pred_junction", "Junction Area", False),
                                    ui.input_switch("pred_crossing", "Pedestrian Crossing", False),
                                    class_="d-flex flex-wrap gap-3"
                                ),
                                class_="p-3 mb-3 input-card"
                            ),

                            # Group 2: Time & Weather (High Importance)
                            ui.div(
                                ui.h6(fa.icon_svg("cloud-sun"), " Environmental Factors", class_="fw-bold text-muted"),
                                ui.input_select("pred_weather", "Weather Condition", choices=unique_weather_options, selected="Fair"),
                                ui.input_slider("pred_hour", "Hour of Day (24h)", 0, 23, 17, step=1),
                                class_="p-3 mb-3 input-card"
                            ),

                            # Group 3: Physics (Medium Importance)
                            ui.div(
                                ui.h6(fa.icon_svg("temperature-half"), " Atmospheric Physics", class_="fw-bold text-muted"),
                                ui.layout_columns(
                                    ui.input_numeric("pred_temp", "Temp (F)", 70),
                                    ui.input_numeric("pred_hum", "Humidity (%)", 50),
                                ),
                                class_="p-3 mb-3 input-card"
                            ),

                            ui.input_action_button("predict_btn", "Analyze Severity Risk", class_="btn-predict w-100"),
                            class_="h-100"
                        )
                    ),

                    # --- Right Column: Results & Visualization ---
                    ui.column(8,
                        ui.row(
                            # Top: Large Number Display
                            ui.column(12,
                                ui.div(
                                    ui.span("PREDICTED SEVERITY LEVEL", class_="sev-label"),
                                    ui.output_ui("prediction_text"), # Note: inline=True to fix spacing
                                    ui.p("Severity ranges from 1 (Low Impact) to 4 (High Impact)", class_="small text-muted mt-2"),
                                    class_="result-card p-4 mb-4"
                                )
                            ),
                            # Bottom: Modern Donut Chart
                            ui.column(12,
                                ui.card(
                                    ui.card_header("Confidence Analysis"),
                                    output_widget("pred_prob_plot"),
                                    full_screen=True
                                )
                            )
                        )
                    )
                ),
                class_="pred-container" # Applies the gradient background
            )
        ),
    )
)


# =============================================================================
# 3. SERVER LOGIC
# =============================================================================

def server(input, output, session):

    # [NEW] Populate weather filter choices dynamically
    @reactive.effect
    def _():
        if not df.empty and 'Weather_Condition' in df.columns:
            weather_options = sorted(df['Weather_Condition'].dropna().unique().tolist())
            ui.update_selectize("filter_weather", choices=weather_options)

    # [CHANGE] Prediction Logic
    # Adding ': Any' forces the strict checker to allow updates later
    pred_result: Any = reactive.Value(None) 
    pred_probs: Any = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def compute_prediction():
        if rf_model is None:
            return

        # 1. Prepare Input Data
        # Ensure we handle the Weather Encoding safely
        w_encoded = 0
        try:
            weather_classes = [str(x) for x in getattr(le_weather, 'classes_', [])]
            if str(input.pred_weather()) in weather_classes:
                w_encoded = le_weather.transform([str(input.pred_weather())])[0]
        except Exception: 
            w_encoded = 0

        # Construct DataFrame (Features ordered exactly as trained)
        input_data = pd.DataFrame({
            'Hour': [input.pred_hour()],
            'Weather_Encoded': [w_encoded],
            'Temperature(F)': [input.pred_temp()],
            'Humidity(%)': [input.pred_hum()],
            # Convert switches (bool) to int (0/1)
            'Traffic_Signal': [int(input.pred_signal())], 
            'Junction': [int(input.pred_junction())],
            'Crossing': [int(input.pred_crossing())]
        })

        # 2. Predict
        try:
            raw_prediction = rf_model.predict(input_data)[0]
            prediction = int(raw_prediction) 
            probabilities = rf_model.predict_proba(input_data)[0]
            
            pred_result.set(prediction)
            pred_probs.set(probabilities)
        except Exception as e:
            print(f"Prediction Error: {e}")

    @render.ui# type: ignore
    def prediction_text():
        val = pred_result.get()
        if val is None:
            # Return the placeholder '?' using a native ui.div
            return ui.div("?", class_="sev-number")
        
        # Return the actual number using a native ui.div
        # This prevents the [object Object] error
        return ui.div(str(val), class_="sev-number")

    @render_widget
    def pred_prob_plot():
        probs = pred_probs.get()
        
        if probs is None or rf_model is None:
            # Return an empty placeholder plot with instruction
            fig = go.Figure()
            fig.update_layout(
                xaxis={'visible': False}, yaxis={'visible': False}, 
                annotations=[{'text': "Adjust inputs and click Analyze", 'showarrow': False, 'font': {'size': 20, 'color': '#ccc'}}]
            )
            return go.FigureWidget(fig)
        
        classes = rf_model.classes_
        
        # Create a Modern Donut Chart
        # Colors: Green(1), Yellow(2), Orange(3), Red(4)
        colors = ['#00b894', '#fdcb6e', '#e17055', '#d63031']
        
        fig = go.Figure(data=[go.Pie(
            labels=[f"Severity {c}" for c in classes],
            values=probs,
            hole=.6, # Makes it a donut
            marker=dict(colors=colors[:len(classes)]),
            textinfo='label+percent',
            textposition='outside',
            hoverinfo='label+percent+value',
            pull=[0.1 if p == max(probs) else 0 for p in probs] # Slightly pull out the winning slice
        )])
        
        # Add the "Winning" % in the center of the donut
        max_prob = max(probs)
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            annotations=[dict(text=f"{max_prob*100:.0f}%<br>Conf.", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return go.FigureWidget(fig)
    
    

    # Reactive calculation for filtered data
    @reactive.calc
    def filtered_df() -> pd.DataFrame:
        # Start with full dataset
        data: pd.DataFrame = df.copy()
        
        if data.empty:
            return data
        
        # Apply Year Filter
        if 'Year' in data.columns:
            year_range = input.filter_year()
            # Only filter if range is not the full default (2016-2023)
            if year_range[0] != 2016 or year_range[1] != 2023:
                mask = (data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])
                data = cast(pd.DataFrame, data[mask])
        
        # Apply Severity Filter (Convert strings back to integers)
        if 'Severity' in data.columns:
            selected_severities = input.filter_severity()
            # Only filter if not all severities are selected
            if selected_severities and len(selected_severities) < 4:
                severity_ints = [int(s) for s in selected_severities]
                mask = data['Severity'].isin(severity_ints)
                data = cast(pd.DataFrame, data[mask])
        
        # Apply Weather Filter
        if 'Weather_Condition' in data.columns:
            selected_weather = input.filter_weather()
            # Only filter if weather conditions are actually selected
            if selected_weather and len(selected_weather) > 0:
                mask = data['Weather_Condition'].isin(selected_weather)
                data = cast(pd.DataFrame, data[mask])
        
        # Apply Hour Filter
        if 'Hour' in data.columns:
            hour_range = input.filter_hour()
            # Only filter if range is not the full default (0-23)
            if hour_range[0] != 0 or hour_range[1] != 23:
                mask = (data['Hour'] >= hour_range[0]) & (data['Hour'] <= hour_range[1])
                data = cast(pd.DataFrame, data[mask])
        
        return data

    # --- ABOUT MODAL LOGIC ---
    @reactive.effect
    @reactive.event(input.about_btn)
    def show_about_modal():
        m = ui.modal(
            ui.div(
                # --- Authors Section ---
                ui.h4("Research Authors", class_="mb-3"),
                ui.layout_columns(
                    # Author 1: Khinje (Cyan)
                    ui.div(
                        ui.div("Khinje Louis P. Curugan", class_="author-header", style="background-color: #00bcd4;"),
                        ui.div(
                            ui.img(src="BSCS3_Khin.jpg", class_="author-photo"),
                            ui.p("BSCS Student", class_="fw-bold"),
                            ui.p("University of Southeastern Philippines", class_="small text-muted mb-0"),
                            ui.p("College of Information and Computing", class_="small text-muted mb-0"),
                            ui.p("BS Computer Science - Major in Data Science", class_="small text-muted mb-0"),
                            ui.p("CSDS 313 Business Intelligence [AY 2025-2026]", class_="small text-muted"),
                            class_="author-body"
                        ),
                        class_="author-card"
                    ),
                    # Author 2: Rui (Magenta/Red)
                    ui.div(
                        ui.div("Rui Manuel A. Palabon", class_="author-header", style="background-color: #c2185b;"),
                        ui.div(
                            ui.img(src="BSCS3_Rui.jpg", class_="author-photo"),
                            ui.p("BSCS Student", class_="fw-bold"),
                            ui.p("University of Southeastern Philippines", class_="small text-muted mb-0"),
                            ui.p("College of Information and Computing", class_="small text-muted mb-0"),
                            ui.p("BS Computer Science - Major in Data Science", class_="small text-muted mb-0"),
                            ui.p("CSDS 313 Business Intelligence [AY 2025-2026]", class_="small text-muted"),
                            class_="author-body"
                        ),
                        class_="author-card"
                    ),
                    # Author 3: Aj Ian (Blue)
                    ui.div(
                        ui.div("Aj Ian L. Resurreccion", class_="author-header", style="background-color: #0d6efd;"),
                        ui.div(
                            ui.img(src="BSCS3_Ian.jpeg", class_="author-photo"),
                            ui.p("BSCS Student", class_="fw-bold"),
                            ui.p("University of Southeastern Philippines", class_="small text-muted mb-0"),
                            ui.p("College of Information and Computing", class_="small text-muted mb-0"),
                            ui.p("BS Computer Science - Major in Data Science", class_="small text-muted mb-0"),
                            ui.p("CSDS 313 Business Intelligence [AY 2025-2026]", class_="small text-muted"),
                            class_="author-body"
                        ),
                        class_="author-card"
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.hr(),
                
                # --- Dataset Section (Restructured) ---
                ui.h4("Dataset Information", class_="mb-3"),
                ui.div(
                    ui.h5("US-Accidents: A Countrywide Traffic Accident Dataset (2016 - 2023)"),
                    ui.p("This is a countrywide traffic accident dataset covering 49 states. The data is collected from February 2016 to March 2023 using multiple APIs that provide streaming traffic event data."),
                    ui.hr(),

                    ui.h6("Acknowledgments & Citations"),
                    
                    # Paper 1 Card
                    ui.div(
                        ui.strong("Paper 1: A Countrywide Traffic Accident Dataset"),
                        ui.p("Moosavi, Sobhan, et al., arXiv preprint arXiv:1906.05409 (2019).", class_="text-muted small mb-1"),
                        ui.a(fa.icon_svg("up-right-from-square"), " https://doi.org/10.48550/arXiv.1906.05409", 
                             href="https://doi.org/10.48550/arXiv.1906.05409", target="_blank", 
                             style="color: #0d6efd; text-decoration: none; font-weight: 500;"),
                        class_="p-3 border rounded mb-2 bg-white"
                    ),

                    # Paper 2 Card
                    ui.div(
                        ui.strong("Paper 2: Accident Risk Prediction based on Heterogeneous Sparse Data"),
                        ui.p("Moosavi, Sobhan, et al. ACM SIGSPATIAL 2019.", class_="text-muted small mb-1"),
                        ui.a(fa.icon_svg("up-right-from-square"), " https://doi.org/10.1145/3347146.3359078", 
                             href="https://doi.org/10.1145/3347146.3359078", target="_blank", 
                             style="color: #0d6efd; text-decoration: none; font-weight: 500;"),
                        class_="p-3 border rounded mb-3 bg-white"
                    ),
                    
                    ui.hr(),
                    ui.h6("Source"),
                    ui.div(
                        ui.span("Official Source: ", class_="fw-bold"),
                        ui.a("smoosavi.org/datasets/us_accidents", href="https://smoosavi.org/datasets/us_accidents", target="_blank", style="text-decoration: underline; color: #0d6efd;"),
                        ui.br(),
                        ui.span("Kaggle Repository: ", class_="fw-bold"),
                        ui.a("Kaggle - US Accidents (2016-2023)", href="https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents", target="_blank", style="text-decoration: underline; color: #0d6efd;"),
                    ),
                    
                    class_="p-3 bg-light border rounded"
                )
            ),
            title="About this Project",
            size="xl",
            easy_close=True,
            footer=ui.modal_button("Close")
        )
        ui.modal_show(m)

    # --- RESET FILTER LOGIC ---
    @reactive.effect
    @reactive.event(input.reset_btn)
    def reset_filters():
        ui.update_slider("filter_year", value=(2016, 2023))  # Changed to tuple
        ui.update_checkbox_group("filter_severity", selected=["1", "2", "3", "4"])
        ui.update_selectize("filter_weather", selected=[])
        ui.update_slider("filter_hour", value=(0, 23))  # Changed to tuple

    # --- KPI Calculations ---
    @render.text
    def kpi_total():
        data = filtered_df()
        return f"{len(data):,}"

    @render.text
    def kpi_daily_avg():
        data = filtered_df()
        if data.empty:
            return "0"
        
        # Calculate unique days in the dataset
        n_days = (data['Start_Time'].max() - data['Start_Time'].min()).days
        if n_days < 1:
            n_days = 1
        
        avg = len(data) / n_days
        return f"{avg:.2f}"

    @render.text
    def kpi_hourly_avg():
        data = filtered_df()
        if data.empty:
            return "0"
        
        # Average accidents occurring in a specific hour slot per day
        n_days = (data['Start_Time'].max() - data['Start_Time'].min()).days
        if n_days < 1:
            n_days = 1
        total_hours = n_days * 24
        
        avg = len(data) / total_hours
        return f"{avg:.2f}"

    # --- PLOTS ---

    @render_widget # type: ignore
    def map_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()

        # PERFORMANCE NOTE: Mapping >100k points crashes browsers.
        # We sample the data for the map if it's too large.
        map_data = data
        if len(data) > 10000:
            map_data = data.sample(10000)

        fig = px.scatter_mapbox(
            map_data,
            lat="Start_Lat",
            lon="Start_Lng",
            color="Severity",
            zoom=5,
            center={"lat": 36.7783, "lon": -119.4179}, # Center of California
            color_continuous_scale=px.colors.sequential.OrRd,
            mapbox_style="carto-positron", # Clean white style
            title="Accident Locations (Sampled)"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    @render_widget # type: ignore
    def poi_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()
        
        # 1. Data Preparation
        poi_cols = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 
                    'Station', 'Amenity', 'Bump', 'Give_Way', 'No_Exit', 'Roundabout']
        
        existing_cols = [c for c in poi_cols if c in data.columns]
        
        results = []
        total_accidents = len(data)
        
        for col in existing_cols:
            count = data[col].sum()
            # Calculate percentage
            pct = (count / total_accidents * 100)
            
            if count > 0: 
                # Create a label that includes the percentage
                # e.g., "Junction (14.5%)"
                pretty_name = col.replace('_', ' ')
                label_with_pct = f"{pretty_name} ({pct:.2f}%)"
                
                results.append({
                    'Feature': pretty_name,       # Clean name for sorting
                    'Label': label_with_pct,      # Name + Pct for Display
                    'Count': count,
                    'Percentage': pct
                })
        
        # Sort by Count so the spiral shape looks good
        poi_df = pd.DataFrame(results).sort_values(by='Count', ascending=True)

        # 2. Build Aesthetic Polar Chart
        fig = px.bar_polar(
            poi_df, 
            r="Count", 
            theta="Label", # Use the new label with %
            color="Feature", # Color by the new label to match legend
            color_discrete_sequence=px.colors.qualitative.Bold, 
            template="plotly_white"
        )
        
        # 3. Customizing the "Flower" Look
        fig.update_layout(
            title="",
            margin=dict(t=20, b=20, l=40, r=80), # Increased right margin for wider legend
            height=380,
            showlegend=True,
            legend=dict(
                title=dict(text="Feature", font=dict(size=12, weight="bold")),
                orientation="v", 
                yanchor="middle", y=0.5, 
                xanchor="left", x=0.95, # Move legend further right
                font=dict(size=11)
            ),
            polar=dict(
                bgcolor="rgba(255,255,255,0)", 
                radialaxis=dict(
                    visible=True,
                    showticklabels=False, 
                    ticks='',
                    gridcolor='#f0f0f0', 
                    gridwidth=1
                ),
                angularaxis=dict(
                    tickfont=dict(size=11.5, color='#555'),
                    rotation=90, 
                    direction="clockwise"
                )
            )
        )
        
        # 4. Enhance Tooltips
        fig.update_traces(
            hovertemplate="<b>%{color}</b><br>" +
                          "Accidents: %{r}<br>"
        )
        
        return fig


    @render_widget # type: ignore
    def day_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()

        # 1. Data Preparation
        counts = data['DayOfWeek'].value_counts().reindex(day_order)
        
        # 2. CUSTOM GREEN-TO-RED COLOR SCALE
        # This defines a gradient: 
        # Low values (0.0) -> Green
        # Mid values (0.5) -> Yellow
        # High values (1.0) -> Red
        custom_scale = [
            [0.0, '#2ecc71'], # Green (Low Accidents)
            [0.5, '#f1c40f'], # Yellow (Medium)
            [1.0, '#e74c3c']  # Red (High Accidents)
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=counts.index, 
            y=counts.values,
            text=counts.values,
            texttemplate='%{text:.2s}', 
            textposition='outside',
            textfont=dict(size=11, color='#555'),
            marker=dict(
                color=counts.values,          # Color based on magnitude
                colorscale=custom_scale,      # Apply our Green-Red scale
                showscale=True,               # <--- Adds the Color Bar (Like your image)
                colorbar=dict(
                    title="Severity",
                    thickness=15,
                    len=0.7,                  # Length of the bar
                    yanchor="top", y=1,
                    xanchor="right", x=1.15    # Position on the right
                ),
                line=dict(color='white', width=1),
                pattern=dict(shape="") 
            ),
            name="Daily Accidents",
            showlegend=False
        ))

        # 3. Aesthetic Layout
        fig.update_layout(
            title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title=dict(text="Day of Week", font=dict(size=13, color='#777')),
                showgrid=False,
                linecolor='#e0e0e0',
                tickfont=dict(color='#555')
            ),
            yaxis=dict(
                title=dict(text="Accident Cases", font=dict(size=13, color='#777')),
                showgrid=True,
                gridcolor='#f9f9f9',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=400, 
            hovermode="x unified"
        )
        return fig


    @render_widget # type: ignore
    def month_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()

        # 1. Data Preparation
        # Reindex ensures months appear in Jan-Dec order, not by count
        counts = data['Month'].value_counts().reindex(month_order)
        
        # 2. SEASONAL AESTHETIC PALETTE
        # Colors representing the feeling of each month
        seasonal_colors = [
            "#a9cce3", "#5dade2", # Jan, Feb (Cool Winter Blues)
            "#a2d9ce", "#73c6b6", "#45b39d", # Mar, Apr, May (Spring Greens)
            "#f7dc6f", "#f0b27a", "#e59866", # Jun, Jul, Aug (Summer Warmth)
            "#d35400", "#a04000", "#6e2c00", # Sep, Oct, Nov (Autumn Earth Tones)
            "#2e86c1"                        # Dec (Deep Winter)
        ]

        # 3. Build Chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=counts.index, 
            y=counts.values,
            text=counts.values, # Show numbers
            texttemplate='%{text:.2s}', # Format numbers (e.g. 1.2k)
            textposition='outside',
            textfont=dict(size=11, color='#555'),
            marker=dict(
                color=seasonal_colors,
                line=dict(color='white', width=1), # Clean white separator
                # Soft rounded corners for bars
                pattern=dict(shape="") 
            ),
            name="Monthly Accidents",
            showlegend=False
        ))

        # 4. Aesthetic Layout
        fig.update_layout(
            title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title=dict(text="Month", font=dict(size=13, color='#777')),
                showgrid=False,
                linecolor='#e0e0e0',
                tickfont=dict(color='#555')
            ),
            yaxis=dict(
                title=dict(text="Accident Cases", font=dict(size=13, color='#777')),
                showgrid=True,
                gridcolor='#f9f9f9',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=400, # Matches the height of the row
            hovermode="x unified"
        )
        return fig

    @render_widget # type: ignore
    def hour_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()

        # 1. Data Preparation
        counts = data['Hour'].value_counts().sort_index()
        total_accidents = counts.sum()
        
        plot_df = pd.DataFrame({'Hour': counts.index, 'Count': counts.values})
        plot_df['Percentage'] = (plot_df['Count'] / total_accidents * 100).round(2)
        
        # 2. SOFT PASTEL SKY CYCLE GRADIENT
        sky_gradient = [
            "#2c3e50", "#34495e", "#2c3e50", "#34495e", # Night
            "#5d6d7e", "#7f8c8d",                       # Dawn
            "#e59866", "#f0b27a",                       # Sunrise
            "#f7dc6f", "#f9e79f", "#fcf3cf", "#f9e79f", # Morning
            "#f1c40f", "#f4d03f", "#f1c40f", "#f4d03f", # Noon
            "#f0b27a", "#e59866", "#d35400",            # Afternoon
            "#884ea0", "#76448a",                       # Dusk
            "#2e4053", "#283747", "#212f3d"             # Night
        ]
        
        # Helper to get color for a specific hour
        def get_color(h):
            return sky_gradient[int(h) % 24]

        # Generate colors for all bars
        bar_colors = [get_color(h) for h in plot_df['Hour']]

        # 3. Identify Peaks for Legend
        m_trace_data = None
        e_trace_data = None
        
        try:
            # Morning Peak (05:00 - 11:00)
            morning_data = plot_df[(plot_df['Hour'] >= 5) & (plot_df['Hour'] <= 11)]
            if not morning_data.empty:
                # FIX: Use argmax() to get the INTEGER position of the max value.
                # This works on both Series and Arrays, satisfying strict linters.
                max_pos = int(morning_data['Count'].argmax())
                # Use .iloc to get the row by integer position
                m_row = morning_data.iloc[max_pos]
                
                m_trace_data = {
                    'x': [m_row['Hour']],
                    'y': [m_row['Count']],
                    'pct': m_row['Percentage'],
                    'color': get_color(m_row['Hour'])
                }

            # Evening Peak (14:00 - 20:00)
            evening_data = plot_df[(plot_df['Hour'] >= 14) & (plot_df['Hour'] <= 20)]
            if not evening_data.empty:
                # FIX: Use argmax() + iloc
                max_pos = int(evening_data['Count'].argmax())
                e_row = evening_data.iloc[max_pos]
                
                e_trace_data = {
                    'x': [e_row['Hour']],
                    'y': [e_row['Count']],
                    'pct': e_row['Percentage'],
                    'color': get_color(e_row['Hour'])
                }
        except Exception:
            pass

        # 4. Build Combo Chart
        fig = go.Figure()

        # TRACE 1: Main Volume (All Bars)
        fig.add_trace(go.Bar(
            x=plot_df['Hour'],
            y=plot_df['Count'],
            text=plot_df['Percentage'].apply(lambda x: f"{x}%"),
            textposition='outside',
            textfont=dict(size=10, color='#666'),
            marker=dict(color=bar_colors),
            name="Hourly Volume",
            showlegend=True
        ))

        # TRACE 2: Morning Peak Highlight (Overlay)
        if m_trace_data:
            fig.add_trace(go.Bar(
                x=m_trace_data['x'],
                y=m_trace_data['y'],
                # Legend Label with Percentage
                name=f"Morning Max ({m_trace_data['pct']}%)",
                marker=dict(
                    color=m_trace_data['color'], 
                    line=dict(color='#555555', width=3) # Highlight border
                ),
                text=[f"{m_trace_data['pct']}%"],
                textposition='outside',
                textfont=dict(size=10, color='#555'),
                hoverinfo="x+y+name"
            ))

        # TRACE 3: Evening Peak Highlight (Overlay)
        if e_trace_data:
            fig.add_trace(go.Bar(
                x=e_trace_data['x'],
                y=e_trace_data['y'],
                # Legend Label with Percentage
                name=f"Evening Max ({e_trace_data['pct']}%)",
                marker=dict(
                    color=e_trace_data['color'], 
                    line=dict(color='#555555', width=3) # Highlight border
                ),
                text=[f"{e_trace_data['pct']}%"],
                textposition='outside',
                textfont=dict(size=10, color='#555'),
                hoverinfo="x+y+name"
            ))

        # TRACE 4: Trend Line
        fig.add_trace(go.Scatter(
            x=plot_df['Hour'],
            y=plot_df['Count'],
            mode='lines+markers',
            line=dict(color='rgba(100, 110, 120, 0.7)', width=3, shape='spline'),
            marker=dict(color='white', size=6, line=dict(width=1, color='#888')),
            name="Trend Line",
            showlegend=True
        ))

        # 5. Aesthetic Layout
        fig.update_layout(
            autosize=True, 
            title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Important: Overlay allows the peak bars to sit ON TOP of the main bars
            barmode='overlay', 
            legend=dict(
                orientation="h", 
                yanchor="bottom",
                y=1.1, 
                xanchor="right",
                x=1,
                font=dict(size=11, color="#555")
            ),
            xaxis=dict(
                tickmode='linear', 
                dtick=1, 
                title=dict(text="Hour of Day", font=dict(size=12, color='#777')),
                showgrid=False,
                linecolor='#e0e0e0',
                tickfont=dict(color='#666')
            ),
            yaxis=dict(
                title=dict(text="Number of Accidents", font=dict(size=12, color='#777')),
                showgrid=True,
                gridcolor='#f9f9f9',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            hovermode="x unified"
        )
        
        return fig

    @render_widget # type: ignore
    def weather_plot():
        data = filtered_df()
        if data.empty or 'Weather_Condition' not in data.columns:
            return go.Figure()

        # 1. Data Preparation - Handle Series properly
        weather_series = data['Weather_Condition'].value_counts()
        
        # Convert to DataFrame immediately to avoid type issues
        weather_df = pd.DataFrame({
            'Weather': weather_series.index.tolist(),
            'Count': weather_series.values.tolist()
        })
        
        total = weather_df['Count'].sum()
        weather_df['Percentage'] = (weather_df['Count'] / total * 100).round(2)
        
        # Sort by count descending and take top 15
        weather_df = weather_df.sort_values('Count', ascending=True).tail(15)
        
        # 2. Weather-Themed Color Mapping
        weather_color_map = {
            'fair': '#FFD700',
            'clear': '#87CEEB',
            'cloudy': '#B0C4DE',
            'mostly cloudy': '#9CA3AF',
            'partly cloudy': '#D3D3D3',
            'overcast': '#708090',
            'rain': '#4682B4',
            'light rain': '#6495ED',
            'heavy rain': '#000080',
            'snow': '#FFFFFF',
            'light snow': '#F0F8FF',
            'heavy snow': '#E6E6FA',
            'fog': '#DCDCDC',
            'mist': '#F5F5F5',
            'haze': '#FFF8DC',
            'thunderstorm': '#483D8B',
            'drizzle': '#B0E0E6',
            'sleet': '#C0C0C0',
            'wintry mix': '#E0E0E0',
            'smoke': '#A9A9A9'
        }
        
        # Assign colors based on weather keywords
        def get_weather_color(weather_name: str) -> str:
            weather_lower = str(weather_name).lower()
            for key, color in weather_color_map.items():
                if key in weather_lower:
                    return color
            return '#95A5A6'  # Default gray
        
        weather_df['Color'] = weather_df['Weather'].apply(get_weather_color)
        
        # 3. Create Modern Horizontal Bar Chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=weather_df['Count'].tolist(),
            y=weather_df['Weather'].tolist(),
            orientation='h',
            text=weather_df['Percentage'].apply(lambda x: f"{x}%").tolist(),
            textposition='outside',
            textfont=dict(size=11, color='#555', weight='bold'),
            marker=dict(
                color=weather_df['Color'].tolist(),
                line=dict(color='white', width=1.5),
                pattern=dict(shape="")
            ),
            hovertemplate="<b>%{y}</b><br>" +
                         "Accidents: %{x:,}<br>" +
                         "Percentage: %{text}<br>" +
                         "<extra></extra>",
            showlegend=False
        ))
        
        # 4. Modern Layout
        fig.update_layout(
            title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title=dict(text="Number of Accidents", font=dict(size=12, color='#777')),
                showgrid=True,
                gridcolor='#f5f5f5',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            yaxis=dict(
                title=dict(text="Weather Condition", font=dict(size=12, color='#777')),
                showgrid=False,
                tickfont=dict(color='#555', size=10)
            ),
            margin=dict(l=20, r=80, t=20, b=20),
            height=400,
            hovermode="y unified"
        )
        
        return fig


    @render_widget # type: ignore
    def cluster_map():
        # 1. Get Data
        data = df
        if data.empty: 
            return go.Figure()
        
        # Performance & Sampling (Clustering is heavy)
        cluster_data = data
        if len(data) > 5000:
            cluster_data = data.sample(5000, random_state=42)
            
        # 2. Prepare Data
        X = cluster_data[['Start_Lat', 'Start_Lng']]
        
        # 3. K-Means Algorithm
        k = input.n_clusters()
        # n_init="auto" fixes the warning you saw earlier
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        
        cluster_data = cluster_data.copy()
        labels = kmeans.fit_predict(X)
        
        # [FIX] Use List Comprehension to avoid NumPy 'ufunc' string error
        # This creates a simple list of strings like ["Zone 1", "Zone 2", ...]
        cluster_data['Cluster Name'] = [f"Zone {i+1}" for i in labels]

        # [FIX] Create a strictly sorted list for the Legend (Zone 1, Zone 2, ... Zone K)
        sort_order = [f"Zone {i+1}" for i in range(k)]
        
        # 4. Professional Aesthetic Map
        fig = px.scatter_mapbox(
            cluster_data,
            lat="Start_Lat",
            lon="Start_Lng",
            color="Cluster Name",
            # [FIX] Force Plotly to use our numerical sort order for the legend
            category_orders={"Cluster Name": sort_order}, 
            zoom= 4, # Optimized Zoom for California
            # Center Coordinates for California
            center={"lat": 37.0, "lon": -119.5},
            # High contrast colors for easy distinction
            color_discrete_sequence=px.colors.qualitative.G10, 
            mapbox_style="carto-positron",
            title="",
            hover_data={
                "Start_Lat": False, 
                "Start_Lng": False, 
                "Cluster Name": True, 
                "City": True,
                "Severity": True
            }
        )
        
        # Clean Layout (Removes whitespace)
        fig.update_layout(
            height=600,
            mapbox_zoom=5,
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                yanchor="top", y=0.98,
                xanchor="left", x=0.02,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#ccc",
                borderwidth=1,
                # Ensure the items are stacked vertically in the sorted order
                traceorder="normal"
            )
        )
        return fig

static_dir = os.path.join(os.path.dirname(__file__), "assets")
# Run the App
app = App(app_ui, server, static_assets=static_dir)

# version 2.0