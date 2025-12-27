import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel

st.set_page_config(page_title="River Water Quality Model", layout="wide")

# =============================================================================
# SIDEBAR: CONFIGURATION & RUN
# =============================================================================

st.sidebar.title("Configuration")
sim_duration = st.sidebar.number_input("Duration (Days)", value=1.0)
dt = st.sidebar.number_input("Time Step (s)", value=200.0)
dt_print = st.sidebar.number_input("Print Interval (s)", value=3600.0)
st.sidebar.divider()
time_disc = st.sidebar.selectbox("Discretisation", ["semi", "imp", "exp"])
advection = st.sidebar.selectbox("Advection", ["Yes", "No"])
diffusion = st.sidebar.selectbox("Diffusion", ["Yes", "No"])

st.sidebar.divider()
run_btn = st.sidebar.button("Run Simulation", type="primary")

# =============================================================================
# MAIN TABS
# =============================================================================

st.title("1D River Water Quality Model")

main_tabs = st.tabs(["River Geometry", "Atmosphere", "Discharges", "Constituents"])

with main_tabs[0]: # River Geometry
    st.header("Channel Properties")
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Length (m)", value=12000.0)
        nc = st.number_input("Cells", value=300)
        width = st.number_input("Width (m)", value=100.0)
    with col2:
        depth = st.number_input("Depth (m)", value=0.5)
        slope = st.number_input("Slope (m/m)", value=0.0001, format="%.6f")
        manning = st.number_input("Manning n", value=0.025, format="%.4f")
    
    # Real-time calc
    area = width * depth
    perimeter = width + 2*depth
    rh = area/perimeter if perimeter > 0 else 0
    vel = (1.0/manning)*(rh**(2/3))*(slope**0.5) if manning > 0 else 0
    st.success(f"Calculated Velocity: {vel:.4f} m/s | Discharge: {vel*area:.4f} m³/s")

with main_tabs[1]: # Atmosphere
    st.header("Meteo Data")
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air Temp (°C)", value=20.0)
        wind = st.number_input("Wind Speed (m/s)", value=0.0)
        humidity = st.number_input("Humidity (%)", value=80.0)
    with col2:
        solar = st.number_input("Solar Const. (W/m²)", value=1370.0)
        cloud = st.number_input("Cloud Cover (%)", value=0.0)
        lat = st.number_input("Latitude", value=38.0)
    
    c1, c2 = st.columns(2)
    sunrise = c1.number_input("Sunrise (h)", value=6.0)
    sunset = c2.number_input("Sunset (h)", value=18.0)

with main_tabs[2]: # Discharges
    st.header("Discharges Configuration")
    # Default Table
    def_data = pd.DataFrame({
        "Cell": [1, 60, 100, 140],
        "Flow (m3/s)": [0.0, 0.0, 0.0, 0.0],
        "Temp (C)": [30.0, 30.0, 50.0, 50.0],
        "BOD (mg/L)": [100.0, 100.0, 200.0, 200.0],
        "DO (mg/L)": [0.0, 0.0, 0.0, 0.0],
        "CO2 (mg/L)": [1.0, 1.0, 1.0, 1.0],
        "Generic": [100000.0, 100000.0, 100000.0, 100000.0]
    })
    edited_discharges = st.data_editor(def_data, num_rows="dynamic")

with main_tabs[3]: # Constituents
    st.header("Constituents Parameters")
    
    # Nested Tabs for each constituent
    c_subtabs = st.tabs(["Temperature", "DO", "BOD", "CO2", "Generic"])
    
    # Dictionary to hold all config for engine
    constituents_config = {}

    def render_constituent_tab(key, def_active, def_val, unit, extra_param=None):
        col_a, col_b = st.columns(2)
        active = col_a.selectbox(f"Active?", ["Yes", "No"], index=0 if def_active else 1, key=f"{key}_act")
        default = col_b.number_input(f"Global Initial Value ({unit})", value=def_val, key=f"{key}_def")
        
        # Boundary Condition
        bc_left = st.number_input(f"Left Boundary Value (x=0) ({unit})", value=def_val, key=f"{key}_bc")
        
        # Extra (T90 for Generic)
        t90 = 0.0
        if extra_param == "T90":
            t90 = st.number_input("T90 Decay Time (Hours)", value=10.0, key="gen_t90")
            
        st.markdown("**Special Initial Conditions (Intervals)**")
        st.markdown("Define specific ranges where the initial value differs from global default.")
        
        # Table for intervals
        df_init = pd.DataFrame(columns=["Start Dist (m)", "End Dist (m)", f"Value ({unit})"])
        # Pre-fill example for Temperature just to show usage, others empty
        if key == "Temperature":
             df_init = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["Start Dist (m)", "End Dist (m)", f"Value ({unit})"]).iloc[0:0]

        edited_inits = st.data_editor(df_init, num_rows="dynamic", key=f"{key}_init_table")
        
        # Parse table
        special_inits = []
        for idx, row in edited_inits.iterrows():
            try:
                # Filter empty rows
                if row[0] is not None:
                     special_inits.append({
                         "start_x": float(row[0]),
                         "end_x": float(row[1]),
                         "value": float(row[2])
                     })
            except: pass
            
        return {
            "active": active == "Yes",
            "unit": unit,
            "default": default,
            "bc_left": bc_left,
            "t90": t90,
            "special_inits": special_inits
        }

    with c_subtabs[0]:
        constituents_config["Temperature"] = render_constituent_tab("Temperature", True, 15.0, "ºC")
    with c_subtabs[1]:
        constituents_config["DO"] = render_constituent_tab("DO", True, 8.0, "mg/L")
    with c_subtabs[2]:
        constituents_config["BOD"] = render_constituent_tab("BOD", True, 5.0, "mg/L")
    with c_subtabs[3]:
        constituents_config["CO2"] = render_constituent_tab("CO2", True, 0.7, "mg/L")
    with c_subtabs[4]:
        constituents_config["Generic"] = render_constituent_tab("Generic", True, 0.0, "conc", "T90")

# =============================================================================
# RUN LOGIC
# =============================================================================

if run_btn:
    model = RiverModel()
    
    with st.spinner("Simulating..."):
        # 1. Setup Grid & Atmos
        model.setup_grid(L, int(nc), width, depth, slope, manning)
        model.setup_atmos(air_temp, wind, humidity, solar, lat, cloud, sunrise, sunset)
        
        # 2. Config options
        model.config.duration_days = sim_duration
        model.config.dt = dt
        model.config.dt_print = dt_print
        model.config.time_discretisation = time_disc
        model.config.advection_active = (advection == "Yes")
        model.config.diffusion_active = (diffusion == "Yes")
        
        # 3. Discharges
        dis_list = []
        for idx, row in edited_discharges.iterrows():
            try:
                dis_list.append({
                    "cell": int(row["Cell"]) - 1, # UI is 1-based, Engine 0-based
                    "flow": float(row["Flow (m3/s)"]),
                    "temp": float(row["Temp (C)"]),
                    "bod": float(row["BOD (mg/L)"]),
                    "do": float(row["DO (mg/L)"]),
                    "co2": float(row["CO2 (mg/L)"]),
                    "generic": float(row["Generic"])
                })
            except: pass
        model.set_discharges(dis_list)
        
        # 4. Constituents
        for name, cfg in constituents_config.items():
            model.add_constituent(
                name=name,
                active=cfg["active"],
                unit=cfg["unit"],
                default_val=cfg["default"],
                left_boundary_val=cfg["bc_left"],
                t90=cfg["t90"],
                special_inits=cfg["special_inits"]
            )
            
        # 5. Execute
        try:
            results = model.run()
            st.session_state['results'] = results
            st.session_state['grid'] = model.grid.xc
            st.toast("Simulation Finished Successfully!", icon="✅")
        except Exception as e:
            st.error(f"Simulation Error: {e}")

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if 'results' in st.session_state:
    res = st.session_state['results']
    xc = st.session_state['grid']
    times = res['times']
    
    st.divider()
    st.header("Simulation Results")
    
    r_tabs = st.tabs(["Spatial Profiles", "Time Series", "Table Data"])
    
    with r_tabs[0]:
        st.caption("Distribution along the river at a specific time.")
        if len(times) > 0:
            time_idx = st.slider("Time Selector (Days)", 0, len(times)-1, len(times)-1)
            t_display = times[time_idx]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for name in ["Temperature", "DO", "BOD", "CO2", "Generic"]:
                if name in res and len(res[name]) > 0:
                    ax.plot(xc, res[name][time_idx], label=name)
            
            ax.set_title(f"Profile at T = {t_display:.3f} days")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
    with r_tabs[1]:
        st.caption("Evolution over time at a specific location.")
        if len(xc) > 0:
            loc_opts = [f"{x:.1f} m" for x in xc]
            sel_loc = st.selectbox("Location Selector", loc_opts)
            loc_idx = loc_opts.index(sel_loc)
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for name in ["Temperature", "DO", "BOD", "CO2", "Generic"]:
                if name in res and len(res[name]) > 0:
                    ts = [step[loc_idx] for step in res[name]]
                    ax2.plot(times, ts, label=name)
            
            ax2.set_title(f"Time Series at X = {xc[loc_idx]:.1f} m")
            ax2.set_xlabel("Time (Days)")
            ax2.set_ylabel("Value")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
    with r_tabs[2]:
        # Simple export for Temperature as example, could expand
        if "Temperature" in res and len(res["Temperature"]) > 0:
            df_res = pd.DataFrame(res["Temperature"], index=np.round(times, 3), columns=np.round(xc, 1))
            st.write("Temperature Data (Rows=Time, Cols=Distance)")
            st.dataframe(df_res)
