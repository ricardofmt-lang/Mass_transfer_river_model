import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverSimulation
from default_configs import DEFAULT_CSVS
from io import StringIO

st.set_page_config(page_title="River Advection-Diffusion Model", layout="wide")

st.title("1D River Water Quality Model")
st.markdown("""
This application is a faithful replica of the VBA/Excel `Channel_TempOD&BOD&CO2` model.
It solves the 1D Advection-Diffusion Equation coupled with biological kinetics and heat transfer.
""")

# =============================================================================
# SIDEBAR: CONFIGURATION
# =============================================================================

st.sidebar.header("Configuration")

# Function to load default or user file
def load_config_file(label, key_name):
    uploaded = st.sidebar.file_uploader(label, type=["csv"], key=key_name)
    if uploaded:
        return pd.read_csv(uploaded, header=None)
    else:
        # Load default string
        return pd.read_csv(StringIO(DEFAULT_CSVS[key_name]), header=None)

# Load all configs
df_main = load_config_file("Main Config (Main.csv)", "Main")
df_river = load_config_file("River Geometry (River.csv)", "River")
df_atmos = load_config_file("Atmosphere (Atmosphere.csv)", "Atmosphere")
df_dis = load_config_file("Discharges (Discharges.csv)", "Discharges")

st.sidebar.subheader("Constituents")
df_temp = load_config_file("Temperature", "Temperature")
df_do = load_config_file("Dissolved Oxygen", "DO")
df_bod = load_config_file("BOD", "BOD")
df_co2 = load_config_file("CO2", "CO2")

# =============================================================================
# SIMULATION CONTROL
# =============================================================================

if st.button("Run Simulation", type="primary"):
    sim = RiverSimulation()
    
    # 1. Parse Data
    with st.spinner("Initializing Model..."):
        sim.load_main_config(df_main)
        sim.load_river_config(df_river)
        sim.load_atmosphere_config(df_atmos)
        sim.load_discharges(df_dis)
        
        # Load Constituents
        sim.load_constituent("Temperature", df_temp)
        sim.load_constituent("DO", df_do)
        sim.load_constituent("BOD", df_bod)
        sim.load_constituent("CO2", df_co2)

    # 2. Run Loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    runner = sim.run_simulation()
    
    for progress in runner:
        progress_bar.progress(progress)
    
    status_text.success("Simulation Complete!")
    
    # Store results in session state
    st.session_state['results'] = sim.results
    st.session_state['grid'] = sim.grid.xc
    st.session_state['length'] = sim.grid.length

# =============================================================================
# RESULTS VISUALIZATION
# =============================================================================

if 'results' in st.session_state:
    res = st.session_state['results']
    xc = st.session_state['grid']
    times = res['times']
    
    tab1, tab2, tab3 = st.tabs(["Longitudinal Profiles", "Time Series", "Raw Data"])
    
    with tab1:
        st.subheader("Spatial Profiles (Distance vs Concentration)")
        
        # Slider to select time
        time_idx = st.slider("Select Time (Days)", 0, len(times)-1, len(times)-1, format="%d")
        t_val = times[time_idx]
        st.write(f"Showing profile at T = {t_val:.3f} days")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for name, history in res['history'].items():
            if len(history) > 0:
                ax.plot(xc, history[time_idx], label=name)
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Concentration / Temperature")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Time Series at Specific Locations")
        
        loc_opts = [f"{x:.1f} m" for x in xc]
        sel_loc = st.selectbox("Select Location", loc_opts, index=0)
        loc_idx = loc_opts.index(sel_loc)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        for name, history in res['history'].items():
            if len(history) > 0:
                # Extract time series for this cell
                ts_data = [step[loc_idx] for step in history]
                ax2.plot(times, ts_data, label=name)
                
        ax2.set_xlabel("Time (Days)")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    with tab3:
        st.dataframe(pd.DataFrame(res['history']['Temperature']))
