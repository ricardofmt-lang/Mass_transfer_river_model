import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel
from default_configs import DEFAULT_CSVS
from io import StringIO

st.set_page_config(page_title="River Advection-Diffusion Model", layout="wide")

st.title("1D River Water Quality Model")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Configuration")

def load_config_file(label, key_name):
    uploaded = st.sidebar.file_uploader(label, type=["csv"], key=key_name)
    if uploaded:
        return pd.read_csv(uploaded, header=None)
    else:
        return pd.read_csv(StringIO(DEFAULT_CSVS[key_name]), header=None)

df_main = load_config_file("Main Config", "Main")
df_river = load_config_file("River Geometry", "River")
df_atmos = load_config_file("Atmosphere", "Atmosphere")
df_dis = load_config_file("Discharges", "Discharges")

st.sidebar.subheader("Constituents")
df_temp = load_config_file("Temperature", "Temperature")
df_do = load_config_file("Dissolved Oxygen", "DO")
df_bod = load_config_file("BOD", "BOD")
df_co2 = load_config_file("CO2", "CO2")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if st.button("Run Simulation", type="primary"):
    # 1. Instantiate
    model = RiverModel()
    
    # 2. Load Data
    with st.spinner("Initializing Model..."):
        model.load_main_config(df_main)
        model.load_river_config(df_river)
        model.load_atmosphere_config(df_atmos)
        model.load_discharges(df_dis)
        model.load_constituent("Temperature", df_temp)
        model.load_constituent("DO", df_do)
        model.load_constituent("BOD", df_bod)
        model.load_constituent("CO2", df_co2)

    # 3. Run
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Execute using the generator
    runner = model.run_simulation()
    for progress in runner:
        progress_bar.progress(progress)
    
    progress_bar.progress(1.0)
    status_text.success("Simulation Complete!")
    
    # 4. Save Results
    st.session_state['results'] = model.results
    st.session_state['grid'] = model.grid.xc

# =============================================================================
# VISUALIZATION
# =============================================================================

if 'results' in st.session_state:
    res = st.session_state['results']
    xc = st.session_state['grid']
    times = res['times']
    
    tab1, tab2, tab3 = st.tabs(["Longitudinal Profiles", "Time Series", "Raw Data"])
    
    with tab1:
        st.subheader("Spatial Profiles")
        if len(times) > 0:
            time_idx = st.slider("Select Time (Days)", 0, len(times)-1, len(times)-1)
            st.write(f"Time: {times[time_idx]:.3f} days")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot whatever is available in results
            for name in ["Temperature", "DO", "BOD", "CO2"]:
                if name in res and len(res[name]) > time_idx:
                    ax.plot(xc, res[name][time_idx], label=name)
            
            ax.set_xlabel("Distance (m)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
    with tab2:
        st.subheader("Time Series")
        loc_opts = [f"{x:.1f} m" for x in xc]
        sel_loc = st.selectbox("Select Location", loc_opts, index=0)
        loc_idx = loc_opts.index(sel_loc)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        for name in ["Temperature", "DO", "BOD", "CO2"]:
            if name in res and len(res[name]) > 0:
                ts_data = [step[loc_idx] for step in res[name]]
                ax2.plot(times, ts_data, label=name)
                
        ax2.set_xlabel("Time (Days)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    with tab3:
        if "Temperature" in res and len(res["Temperature"]) > 0:
             st.dataframe(pd.DataFrame(res["Temperature"]))
