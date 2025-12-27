import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel

st.set_page_config(page_title="River Advection-Diffusion Model", layout="wide")

st.title("1D River Water Quality Model")
st.markdown("Enter simulation parameters directly below, exactly as in the original Excel tool.")

# =============================================================================
# HELPER: Construct DataFrames for Engine
# =============================================================================
def make_kv_df(data_dict):
    """Creates a 2-column DataFrame (Key, Value) to mimic Excel vertical parameters."""
    return pd.DataFrame(list(data_dict.items()), columns=["Key", "Value"])

# =============================================================================
# INPUT TABS
# =============================================================================
tabs = st.tabs(["Configuration", "River Geometry", "Atmosphere", "Discharges", "Constituents"])

with tabs[0]:
    st.header("Simulation Options")
    col1, col2 = st.columns(2)
    with col1:
        sim_duration = st.number_input("Simulation Duration (Days)", value=1.0)
        dt = st.number_input("Time Step (Seconds)", value=200.0)
        dt_print = st.number_input("Print Interval (Seconds)", value=3600.0)
    with col2:
        time_disc = st.selectbox("Time Discretisation", ["semi", "imp", "exp"], index=0)
        advection = st.selectbox("Advection Active", ["Yes", "No"], index=0)
        adv_type = st.selectbox("Advection Type", ["QUICK", "upwind", "central"], index=0)
        diffusion = st.selectbox("Diffusion Active", ["Yes", "No"], index=0)

    # Bridge: Construct "Main" DataFrame
    main_config = {
        "SimulationDuration(Days)": sim_duration,
        "TimeStep(Seconds)": dt,
        "Dtprint(seconds)": dt_print,
        "TimeDiscretisation": time_disc,
        "Advection": advection,
        "AdvectionType": adv_type,
        "Diffusion": diffusion
    }

with tabs[1]:
    st.header("River & Flow Properties")
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Channel Length (m)", value=12000.0)
        nc = st.number_input("Number of Cells", value=300, step=1)
        width = st.number_input("River Width (m)", value=100.0)
    with col2:
        depth = st.number_input("Water Depth (m)", value=0.5)
        slope = st.number_input("River Slope (m/m)", value=0.0001, format="%.6f")
        manning = st.number_input("Manning Coef (n)", value=0.025, format="%.4f")
        
    # Real-time Calculation Display
    area = width * depth
    perimeter = width + 2*depth
    rh = area/perimeter if perimeter > 0 else 0
    vel = (1.0/manning)*(rh**(2/3))*(slope**0.5) if manning > 0 else 0
    disch = vel * area
    
    st.info(f"Calculated Flow Velocity: {vel:.4f} m/s | Discharge: {disch:.4f} m³/s")

    river_config = {
        "ChannelLength": L,
        "NumberOfCells": nc,
        "RiverWidth": width,
        "WaterDepth": depth,
        "RiverSlope": slope,
        "ManningCoef(n)": manning
    }

with tabs[2]:
    st.header("Atmospheric Data")
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air Temperature (°C)", value=20.0)
        wind = st.number_input("Wind Speed (m/s)", value=0.0)
        humidity = st.number_input("Air Humidity (%)", value=80.0)
        solar = st.number_input("Solar Constant (W/m²)", value=1370.0)
    with col2:
        lat = st.number_input("Latitude (Degrees)", value=38.0)
        cloud = st.number_input("Cloud Cover (%)", value=0.0)
        sunrise = st.number_input("Sunrise Hour", value=6.0)
        sunset = st.number_input("Sunset Hour", value=18.0)

    atmos_config = {
        "AirTemperature": air_temp,
        "WindSpeed": wind,
        "AirHumidity": humidity,
        "SolarConstant": solar,
        "Latitude": lat,
        "CloudCover": cloud,
        "SunRizeHour": sunrise,
        "SunSetHour": sunset,
        # Hidden Defaults
        "h_min": 6.9,
        "SkyTemperature": -40,
        "O2PartialPressure": 0.2095,
        "CO2PartialPressure": 0.000395
    }

with tabs[3]:
    st.header("Discharges")
    st.markdown("Edit the discharge properties below. Use row index for Discharge 1, 2, 3, 4.")
    
    # Initial Data for the Table
    data = {
        "Name": ["Descarga 1", "Descarga 2", "Descarga 3", "Descarga 4"],
        "Cell": [1, 60, 100, 140],
        "Flow (m3/s)": [0.0, 0.0, 0.0, 0.0],
        "Temp (C)": [30.0, 30.0, 50.0, 50.0],
        "BOD (mg/L)": [100.0, 100.0, 200.0, 200.0],
        "DO (mg/L)": [0.0, 0.0, 0.0, 0.0],
        "CO2 (mg/L)": [1.0, 1.0, 1.0, 1.0],
        "Generic": [100000.0, 100000.0, 100000.0, 100000.0]
    }
    df_dis_input = pd.DataFrame(data)
    edited_discharges = st.data_editor(df_dis_input, num_rows="dynamic")

with tabs[4]:
    st.header("Constituents")
    
    c_tabs = st.tabs(["Temperature", "DO", "BOD", "CO2"])
    
    constituents_data = {}
    
    def render_constituent_ui(key, default_val, unit, active_default="Yes"):
        st.subheader(f"{key} Properties")
        active = st.selectbox(f"{key} Active?", ["Yes", "No"], index=0 if active_default=="Yes" else 1, key=f"{key}_act")
        val = st.number_input(f"Default Initial Value ({unit})", value=default_val, key=f"{key}_val")
        return {
            "PropertyName": key,
            "PropertyUnits": unit,
            "PropertyActive": active,
            "DefaultValue": val
        }

    with c_tabs[0]:
        constituents_data["Temperature"] = render_constituent_ui("Temperature", 15.0, "ºC")
    with c_tabs[1]:
        constituents_data["DO"] = render_constituent_ui("DO", 0.0, "mg/L")
    with c_tabs[2]:
        constituents_data["BOD"] = render_constituent_ui("BOD", 5.0, "mg/L")
    with c_tabs[3]:
        constituents_data["CO2"] = render_constituent_ui("CO2", 0.7, "mg/L")

# =============================================================================
# RUN LOGIC
# =============================================================================

if st.button("Run Simulation", type="primary"):
    model = RiverModel()
    
    with st.spinner("Setting up Simulation..."):
        # 1. Load Main, River, Atmos from Constructed DataFrames
        model.load_main_config(make_kv_df(main_config))
        model.load_river_config(make_kv_df(river_config))
        model.load_atmosphere_config(make_kv_df(atmos_config))
        
        # 2. Transform Discharges to Engine Format (Horizontal/Transposed)
        # The engine expects: Row 1=Cells, Row 2=Flows...
        # We need to construct a DataFrame where columns are D1, D2... and rows are properties
        # Keys expected by engine: DischargeCells, DischargeFlowRates, etc.
        
        d_map = {
            "DischargeCells": edited_discharges["Cell"].values,
            "DischargeFlowRates": edited_discharges["Flow (m3/s)"].values,
            "DischargeTemperatures": edited_discharges["Temp (C)"].values,
            "DischargeConcentrations_BOD": edited_discharges["BOD (mg/L)"].values,
            "DischargeConcentrations_DO": edited_discharges["DO (mg/L)"].values,
            "DischargeConcentrations_CO2": edited_discharges["CO2 (mg/L)"].values,
            "DischargeGeneric": edited_discharges["Generic"].values
        }
        
        # Create a DF where index is the Property Name, and cols are 0, 1, 2...
        # Then reset index to put Property Name in Col 0
        df_dis_engine = pd.DataFrame(d_map).T.reset_index()
        # This results in:
        # index (Col 0) | 0 | 1 | 2 ...
        # DischargeCells| 1 | 60| 100
        model.load_discharges(df_dis_engine)
        
        # 3. Load Constituents
        for name, data in constituents_data.items():
            model.load_constituent(name, make_kv_df(data))
            
    # 4. Run
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        runner = model.run_simulation()
        for progress in runner:
            progress_bar.progress(progress)
        progress_bar.progress(1.0)
        status_text.success("Simulation Complete!")
        
        st.session_state['results'] = model.results
        st.session_state['grid'] = model.grid.xc
        
    except Exception as e:
        st.error(f"Simulation Error: {e}")

# =============================================================================
# VISUALIZATION
# =============================================================================
if 'results' in st.session_state:
    res = st.session_state['results']
    xc = st.session_state['grid']
    times = res['times']
    
    st.divider()
    st.header("Results")
    
    tab1, tab2, tab3 = st.tabs(["Longitudinal Profiles", "Time Series", "Raw Data"])
    
    with tab1:
        if len(times) > 0:
            time_idx = st.slider("Select Time (Days)", 0, len(times)-1, len(times)-1)
            t_val = times[time_idx]
            st.write(f"**Profile at T = {t_val:.3f} days**")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for name in ["Temperature", "DO", "BOD", "CO2"]:
                if name in res and len(res[name]) > time_idx:
                    ax.plot(xc, res[name][time_idx], label=name)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Concentration / Temp")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
    with tab2:
        if len(xc) > 0:
            loc_opts = [f"{x:.1f} m" for x in xc]
            sel_loc = st.selectbox("Select Location to Plot", loc_opts, index=0)
            loc_idx = loc_opts.index(sel_loc)
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for name in ["Temperature", "DO", "BOD", "CO2"]:
                if name in res and len(res[name]) > 0:
                    # Extract time series for this specific cell
                    ts = [step[loc_idx] for step in res[name]]
                    ax2.plot(times, ts, label=name)
            ax2.set_xlabel("Time (Days)")
            ax2.set_ylabel("Value")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
            
    with tab3:
        if "Temperature" in res:
            st.dataframe(pd.DataFrame(res["Temperature"], index=[f"T={t:.2f}d" for t in times], columns=[f"x={x:.0f}" for x in xc]))
