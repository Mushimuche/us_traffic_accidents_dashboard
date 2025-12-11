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

# Run the App
app = App(app_ui, server)

# version 1.3