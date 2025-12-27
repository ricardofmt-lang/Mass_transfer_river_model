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

main_tabs = st.tabs(["River Geometry", "Atmosphere", "Discharges", "Constituents", "Results"])

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
    
    c_subtabs = st.tabs(["Temperature", "DO", "BOD", "CO2", "Generic"])
    constituents_config = {}

    def render_constituent_tab(key, def_active, def_val, unit, is_generic=False):
        # 1. Active & Global Default
        col_a, col_b = st.columns(2)
        active = col_a.selectbox(f"Active?", ["Yes", "No"], index=0 if def_active else 1, key=f"{key}_act")
        
        # 2. Boundary Conditions
        st.markdown("##### Boundary Conditions")
        bc_c1, bc_c2 = st.columns(2)
        with bc_c1:
            st.markdown("**Left Boundary (Inflow)**")
            bc_left_type = st.selectbox("Type", ["Fixed Value", "Zero Gradient"], key=f"{key}_bc_l_type")
            bc_left_val = st.number_input(f"Value ({unit})", value=def_val, key=f"{key}_bc_l_val")
        with bc_c2:
            st.markdown("**Right Boundary (Outflow)**")
            bc_right_type = st.selectbox("Type", ["Zero Gradient", "Fixed Value"], key=f"{key}_bc_r_type")
            bc_right_val = st.number_input(f"Value ({unit})", value=def_val, key=f"{key}_bc_r_val")

        # 3. Initial Conditions Mode
        st.markdown("##### Initial Conditions")
        ic_mode = st.radio("Initialization Mode", ["Default", "Cell List", "Interval"], horizontal=True, key=f"{key}_ic_mode")
        
        init_cells = []
        init_intervals = []
        default_ic = def_val
        
        if ic_mode == "Default":
            default_ic = st.number_input(f"Global Initial Value ({unit})", value=def_val, key=f"{key}_ic_def")
        elif ic_mode == "Cell List":
            st.caption("Define value per cell index.")
            df_cell = pd.DataFrame(columns=["Cell Index", f"Value ({unit})"])
            if key == "Temperature": df_cell = pd.DataFrame([[150, 25.0]], columns=["Cell Index", f"Value ({unit})"]) # Example
            ed_cell = st.data_editor(df_cell, num_rows="dynamic", key=f"{key}_ic_cell")
            for _, row in ed_cell.iterrows():
                try: init_cells.append({"idx": int(row[0]), "val": float(row[1])})
                except: pass
        elif ic_mode == "Interval":
            st.caption("Define value for spatial ranges.")
            df_int = pd.DataFrame(columns=["Start Dist (m)", "End Dist (m)", f"Value ({unit})"])
            if key == "Temperature": df_int = pd.DataFrame([[0.0, 2000.0, 18.0]], columns=["Start Dist (m)", "End Dist (m)", f"Value ({unit})"])
            ed_int = st.data_editor(df_int, num_rows="dynamic", key=f"{key}_ic_int")
            for _, row in ed_int.iterrows():
                try: init_intervals.append({"start": float(row[0]), "end": float(row[1]), "val": float(row[2])})
                except: pass
        
        # 4. Kinetics (Generic Only)
        k_val = 0.0
        if is_generic:
            st.markdown("##### Kinetics")
            k_mode = st.selectbox("Decay Model", ["T90 (Bacteria)", "First Order Decay (k)"], key="gen_k_mode")
            if k_mode == "T90 (Bacteria)":
                t90 = st.number_input("T90 (Hours)", value=10.0, key="gen_t90")
                if t90 > 0: k_val = 2.302585 / (t90 * 3600.0)
            else:
                k_day = st.number_input("Decay Rate k (1/day)", value=0.5, key="gen_k")
                k_val = k_day / 86400.0

        return {
            "active": active == "Yes",
            "unit": unit,
            "init_mode": ic_mode, # Default, Cell, Interval
            "default_val": default_ic,
            "init_cells": init_cells,
            "init_intervals": init_intervals,
            "bc_left_type": "Fixed" if bc_left_type == "Fixed Value" else "ZeroGrad",
            "bc_left_val": bc_left_val,
            "bc_right_type": "Fixed" if bc_right_type == "Fixed Value" else "ZeroGrad",
            "bc_right_val": bc_right_val,
            "k_decay": k_val
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
        constituents_config["Generic"] = render_constituent_tab("Generic", True, 0.0, "conc", is_generic=True)

with main_tabs[4]: # Results
    st.header("Simulation Results")
    if 'results' not in st.session_state:
        st.info("Run the simulation to see results here.")
    else:
        res = st.session_state['results']
        xc = st.session_state['grid']
        times = res['times']
        
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
            if "Temperature" in res and len(res["Temperature"]) > 0:
                df_res = pd.DataFrame(res["Temperature"], index=np.round(times, 3), columns=np.round(xc, 1))
                st.write("Temperature Data (Rows=Time, Cols=Distance)")
                st.dataframe(df_res)

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
                    "cell": int(row["Cell"]) - 1, 
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
                init_mode=cfg["init_mode"],
                default_val=cfg["default_val"],
                init_cells=cfg["init_cells"],
                init_intervals=cfg["init_intervals"],
                bc_left_type=cfg["bc_left_type"],
                bc_left_val=cfg["bc_left_val"],
                bc_right_type=cfg["bc_right_type"],
                bc_right_val=cfg["bc_right_val"],
                k_decay=cfg["k_decay"]
            )
            
        # 5. Execute
        try:
            results = model.run()
            st.session_state['results'] = results
            st.session_state['grid'] = model.grid.xc
            st.toast("Simulation Finished Successfully!", icon="✅")
            # Force rerun to update the Results tab immediately
            st.rerun()
        except Exception as e:
            st.error(f"Simulation Error: {e}")
