from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faicons as fa
import os

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
        ui.h4("Filters"),
        ui.p("Filter options will be placed here."),
        ui.input_action_button("reset_btn", "Reset Filters", class_="btn-danger"),
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
    """)),
    # --- Main Content ---
    ui.h2("California Road Accidents Dashboard"),

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
            ui.card(
                ui.card_header("Accident Frequency by Hour of Day"),
                output_widget("hour_plot")
            )
        ),

        # TAB 2: Weather Analysis (Empty for now)
        ui.nav_panel(
            "Weather Analysis",
            ui.p("Weather impact analysis coming soon...")
        ),

        # TAB 3: Prediction (Empty for now)
        ui.nav_panel(
            "Prediction",
            ui.p("ML Prediction models coming soon...")
        ),
    )
)


# =============================================================================
# 3. SERVER LOGIC
# =============================================================================

def server(input, output, session):
    
    # Reactive calculation for filtered data
    @reactive.calc
    def filtered_df():
        # Listen to reset button (dummy dependency for now)
        input.reset_btn() 
        return df

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
        
        # Features to check
        pois = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop']
        
        # Create a subplot grid for Pie charts (2x2)
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}],
                                                   [{'type':'domain'}, {'type':'domain'}]],
                            subplot_titles=pois)

        # Loop to create pie charts
        positions = [(1,1), (1,2), (2,1), (2,2)]
        for i, poi in enumerate(pois):
            if poi in data.columns:
                counts = data[poi].value_counts()
                fig.add_trace(
                    go.Pie(labels=counts.index, values=counts.values, name=poi),
                    row=positions[i][0], col=positions[i][1]
                )

        fig.update_layout(height=400, showlegend=False)
        return fig

    @render_widget # type: ignore
    def day_plot():
        data = filtered_df()
        if data.empty:
            return go.Figure()

        # Count per day
        counts = data['DayOfWeek'].value_counts().reindex(day_order)
        
        fig = px.bar(
            x=counts.index, 
            y=counts.values,
            labels={'x': 'Day', 'y': 'Accidents'},
            color=counts.values,
            color_continuous_scale='Viridis'
        )
        return fig

    @render_widget # type: ignore
    def hour_plot():
        data = filtered_df()
        # FIX: Split into two lines to fix "Multiple statements on one line" error
        if data.empty:
            return go.Figure()

        # 1. Data Preparation
        counts = data['Hour'].value_counts().sort_index()
        total_accidents = counts.sum()
        
        # Create a temporary dataframe for plotting
        plot_df = pd.DataFrame({'Hour': counts.index, 'Count': counts.values})
        plot_df['Percentage'] = (plot_df['Count'] / total_accidents * 100).round(2)
        
        # 2. Define Colors based on Time (Morning vs Night)
        # Logic: 6 AM to 6 PM (18:00) is "Day" (Cyan), else "Night" (Dark Blue)
        colors = []
        for h in plot_df['Hour']:
            if 6 <= h <= 18:
                colors.append('#26d0ce')  # Cyan/Teal for Morning
            else:
                colors.append('#1a2980')  # Dark Blue for Night

        # 3. Build the Combo Chart
        fig = go.Figure()

        # Add Bar Chart
        fig.add_trace(go.Bar(
            x=plot_df['Hour'],
            y=plot_df['Count'],
            text=plot_df['Percentage'].apply(lambda x: f"{x}%"), # Add % text
            textposition='outside', # Text above bar
            textfont=dict(size=10),
            marker_color=colors,
            name="Accidents",
            showlegend=False
        ))

        # Add Line Overlay (White line connecting tops)
        fig.add_trace(go.Scatter(
            x=plot_df['Hour'],
            y=plot_df['Count'],
            mode='lines+markers',
            line=dict(color='white', width=3, shape='spline'), # Smooth line
            marker=dict(color='white', size=6, line=dict(width=1, color='#333')),
            name="Trend",
            showlegend=False
        ))

        # 4. Aesthetic Layout
        fig.update_layout(
            title="",
            # Clean White Background
            plot_bgcolor='white',
            paper_bgcolor='white',
            # X-Axis configuration
            xaxis=dict(
                tickmode='linear', 
                dtick=1, 
                title="Hour of Day",
                showgrid=False,
                linecolor='#e0e0e0'
            ),
            # Y-Axis configuration
            yaxis=dict(
                title="Accident Cases",
                showgrid=True,
                gridcolor='#f5f5f5',
                zeroline=False
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            height=500,
            hovermode="x unified"
        )
        
        return fig

# Run the App
app = App(app_ui, server)

# version 1.1