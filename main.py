from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import faicons as fa
import os
from sklearn.cluster import KMeans

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
            ui.card(
                ui.card_header("Accident Frequency by Hour of Day"),
                output_widget("hour_plot")
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
    def cluster_map():
        # 1. Get Data
        data = filtered_df()
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

# version 1.7