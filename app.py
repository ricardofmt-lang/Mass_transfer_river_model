import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river_model import RiverModel

st.set_page_config(page_title="River Transport Digital Twin", layout="wide")

st.title("Channel Transport Model (Complete Replica)")

# --- Sidebar (Simulation Control) ---
st.sidebar.header("Controls")
sim_days = st.sidebar.number_input("Simulation Duration (Days)", value=1.0)
time_step = st.sidebar.number_input("Time Step (Seconds)", value=200.0)

st.sidebar.subheader("Transport Options")
adv_active = st.sidebar.checkbox("Advection", value=True)
diff_active = st.sidebar.checkbox("Diffusion", value=True)

adv_type = "upwind"
quick_up = False
quick_ratio = 2.0

if adv_active:
    adv_type = st.sidebar.selectbox("Advection Type", ["upwind", "central", "QUICK"], index=2)
    if adv_type == "QUICK":
        quick_up = st.sidebar.checkbox("Use QUICK UP (Steep Gradient)", value=True)
        if quick_up:
            quick_ratio = st.sidebar.number_input("QUICK UP Ratio", value=2.0)

time_disc = st.sidebar.selectbox("Time Discretization", ["exp", "imp", "semi"], index=1)

tab_river, tab_atmo, tab_dis, tab_props, tab_run = st.tabs(["River/Flow", "Atmosphere", "Discharges", "Properties", "Run & Results"])

# 1. River
with tab_river:
    col1, col2 = st.columns(2)
    with col1:
        length = st.number_input("Channel Length (m)", value=12000.0)
        width = st.number_input("River Width (m)", value=50.0)
        depth = st.number_input("Water Depth (m)", value=5.0)
        n_cells = st.number_input("Number of Cells", value=100, step=1)
    with col2:
        velocity = st.number_input("Flow Velocity (m/s)", value=0.1)
        slope_pct = st.number_input("River Slope (%)", value=0.01)
        diffusivity = st.number_input("Diffusivity (m2/s) (0=Auto)", value=0.0)
        if diffusivity == 0:
            diffusivity = 0.1 * abs(velocity) * width
            st.caption(f"Calculated: {diffusivity}")

# 2. Atmosphere
with tab_atmo:
    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp = st.number_input("Air Temp (°C)", value=20.0)
        wind = st.number_input("Wind Speed (m/s)", value=2.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
    with col2:
        solar = st.number_input("Solar Constant (W/m2)", value=1367.0)
        lat = st.number_input("Latitude (deg)", value=40.0)
        cloud = st.number_input("Cloud Cover (%)", value=10.0)
    with col3:
        tsr = st.number_input("Sunrise Hour", value=6.0)
        tss = st.number_input("Sunset Hour", value=20.0)

# 3. Discharges
with tab_dis:
    if 'discharges' not in st.session_state: st.session_state.discharges = []
    with st.expander("Add Discharge"):
        d_name = st.text_input("Name", "D1")
        d_loc = st.number_input("Location (m)", 0.0)
        d_flow = st.number_input("Flow Rate (m3/s)", 1.0)
        d_val_temp = st.number_input("Temp (°C)", 20.0)
        d_val_bod = st.number_input("BOD (mg/L)", 50.0)
        d_val_do = st.number_input("DO (mg/L)", 0.0)
        if st.button("Add Discharge"):
            cell_idx = int(d_loc / (length/n_cells))
            st.session_state.discharges.append({
                "name": d_name, "cell": cell_idx, "vol": d_flow,
                "vals": {"Temperature": d_val_temp, "BOD": d_val_bod, "DO": d_val_do}
            })
    st.dataframe(pd.DataFrame(st.session_state.discharges))

# Helper for Initial Conditions GUI
def render_init_ui(key_prefix):
    itype = st.selectbox("Initialization Method", ["DEFAULT", "CELL", "INTERVAL_M"], key=f"{key_prefix}_type")
    
    init_data = {"type": itype, "intervals": []}
    
    # Default Value always required as fallback
    def_val = st.number_input("Default Value", value=0.0, key=f"{key_prefix}_def")
    init_data["default_val"] = def_val
    
    if itype == "CELL":
        st.caption("Add points (Cell Index, Value)")
        c_idx = st.number_input("Cell Index", 0, int(n_cells)-1, key=f"{key_prefix}_cidx")
        c_val = st.number_input("Value", key=f"{key_prefix}_cval")
        if st.button("Add Point", key=f"{key_prefix}_btn"):
            if f"{key_prefix}_pts" not in st.session_state: st.session_state[f"{key_prefix}_pts"] = []
            st.session_state[f"{key_prefix}_pts"].append((c_idx, c_idx, c_val))
    
    elif itype == "INTERVAL_M":
        st.caption("Add Interval (Start m, End m, Value)")
        m_start = st.number_input("Start (m)", 0.0, length, key=f"{key_prefix}_mstart")
        m_end = st.number_input("End (m)", 0.0, length, key=f"{key_prefix}_mend")
        m_val = st.number_input("Value", key=f"{key_prefix}_mval")
        if st.button("Add Interval", key=f"{key_prefix}_btn_m"):
            if f"{key_prefix}_pts" not in st.session_state: st.session_state[f"{key_prefix}_pts"] = []
            st.session_state[f"{key_prefix}_pts"].append((m_start, m_end, m_val))
            
    # Show active intervals
    if f"{key_prefix}_pts" in st.session_state:
        st.write("Configured Intervals:", st.session_state[f"{key_prefix}_pts"])
        init_data["intervals"] = st.session_state[f"{key_prefix}_pts"]
        
    return init_data

# 4. Properties
with tab_props:
    # Temperature
    with st.expander("Temperature", expanded=True):
        t_act = st.checkbox("Active", value=True, key="t_act")
        st.subheader("Initial Conditions")
        t_init_cfg = render_init_ui("temp")
        st.subheader("Boundary Conditions")
        t_left = st.number_input("Left Value", value=15.0, key="t_l")
        t_right = st.number_input("Right Value", value=15.0, key="t_r")
        t_cyclic = st.checkbox("Cyclic Boundary", key="t_cyc")
        st.subheader("Fluxes")
        t_fs = st.checkbox("Free Surface Flux", value=True, key="t_fs")
    
    # BOD
    with st.expander("BOD"):
        b_act = st.checkbox("Active", value=True, key="b_act")
        st.subheader("Initial Conditions")
        b_init_cfg = render_init_ui("bod")
        st.subheader("Params")
        b_dec = st.number_input("Decay Rate (1/day)", value=0.23)
        b_ana = st.checkbox("Anaerobic Respiration", value=True)
        st.subheader("Boundary")
        b_left = st.number_input("Left Value", value=2.0, key="b_l")
        
    # DO
    with st.expander("DO"):
        d_act = st.checkbox("Active", value=True, key="d_act")
        st.subheader("Initial Conditions")
        d_init_cfg = render_init_ui("do")
        st.subheader("Boundary")
        d_left = st.number_input("Left Value", value=9.0, key="d_l")
        d_fs = st.checkbox("Reaeration", value=True, key="d_fs")

# Run
with tab_run:
    if st.button("RUN SIMULATION", type="primary"):
        def get_d(pname):
            return [{"name":d['name'], "cell":d['cell'], "volume_rate":d['vol'], "specific_value":d['vals'].get(pname,0)} for d in st.session_state.discharges]

        prop_params = {}
        if t_act:
            prop_params["Temperature"] = {
                "base": {"active":True},
                "init_config": t_init_cfg,
                "boundary": {
                    "left_value": t_left, "right_value": t_right, "cyclic": t_cyclic,
                    "free_surface_flux": True, "fs_sensible_heat": t_fs, "fs_latent_heat": t_fs, "fs_radiative_heat": t_fs
                },
                "discharges": get_d("Temperature")
            }
        
        if b_act:
            prop_params["BOD"] = {
                "base": {"active":True, "decay_rate": b_dec/86400, "max_val_logistic":1e6, "anaerobic_respiration":b_ana},
                "init_config": b_init_cfg,
                "boundary": {"left_value": b_left},
                "discharges": get_d("BOD")
            }
            
        if d_act:
            prop_params["DO"] = {
                "base": {"active":True},
                "init_config": d_init_cfg,
                "boundary": {
                    "left_value": d_left, "free_surface_flux": d_fs,
                    "gas_exchange_params": {
                        "label":"O2", "partial_pressure":0.2095, "molecular_weight":32000,
                        "henry_temps": [0,10,20,30,40], "henry_ks": [0.000067, 0.000054, 0.000044, 0.000037, 0.000030]
                    }
                },
                "discharges": get_d("DO")
            }

        grid_p = {"length": length, "river_width": width, "water_depth": depth, "nc": int(n_cells)}
        flow_p = {"velocity": velocity, "diffusivity": diffusivity, "river_slope": slope_pct/100.0}
        atm_p = {
            "temperature": air_temp, "wind_speed": wind, "humidity": humidity/100.0,
            "solar_constant": solar, "latitude": lat, "cloud_cover": cloud/100.0,
            "tsr": tsr, "tss": tss, "sky_temperature": 0, "sky_temp_imposed": False
        }
        ctrl_p = {
            "total_time": 0, "dt": time_step, "sim_duration": sim_days * 86400,
            "advection": adv_active, "diffusion": diff_active, "adv_type": adv_type,
            "quick_up": quick_up, "quick_up_ratio": quick_ratio, "time_disc": time_disc
        }
        
        model = RiverModel()
        model.initialize(grid_p, flow_p, atm_p, ctrl_p, prop_params)
        
        with st.spinner("Calculating..."):
            prog = st.progress(0)
            res = model.run(lambda x: prog.progress(x))
            
        st.success("Done!")
        
        # Plot
        x_ax = np.linspace(0, length, int(n_cells))
        fig, ax = plt.subplots()
        if "Temperature" in res: ax.plot(x_ax, res["Temperature"][-1], label="Temp", color="red")
        if "DO" in res: ax.plot(x_ax, res["DO"][-1], label="DO", color="blue")
        if "BOD" in res: ax.plot(x_ax, res["BOD"][-1], label="BOD", color="brown")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
