import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel

st.set_page_config(page_title="River Water Quality Model", layout="wide")

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("Configuration")
sim_duration = st.sidebar.number_input("Duration (Days)", 1.0)
dt = st.sidebar.number_input("Time Step (s)", 200.0)
dt_print = st.sidebar.number_input("Print Interval (s)", 3600.0)
st.sidebar.divider()
time_disc = st.sidebar.selectbox("Discretisation", ["semi", "imp", "exp"])
advection = st.sidebar.selectbox("Advection", ["Yes", "No"])
adv_type = "QUICK"
quick_ratio = 0.5
if advection == "Yes":
    adv_type = st.sidebar.selectbox("Advection Type", ["QUICK", "Upwind", "Central"])
    if adv_type == "QUICK":
        quick_ratio = st.sidebar.number_input("QUICK UP Ratio", value=4.0) # Excel default 4
diffusion = st.sidebar.selectbox("Diffusion", ["Yes", "No"])

st.sidebar.divider()
run_btn = st.sidebar.button("Run Simulation", type="primary")

# =============================================================================
# MAIN UI
# =============================================================================
st.title("1D River Water Quality Model")
main_tabs = st.tabs(["River Geometry", "Atmosphere", "Discharges", "Constituents", "Results"])

with main_tabs[0]: 
    st.header("River Geometry & Flow")
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Length (m)", value=12000.0)
        nc = st.number_input("Cells", value=300)
        width = st.number_input("Width (m)", value=100.0)
        depth = st.number_input("Depth (m)", value=0.5)
    with col2:
        slope = st.number_input("Slope (m/m)", value=0.0001, format="%.6f")
        manning = st.number_input("Manning n (storage only)", value=0.025, format="%.4f")
        Q_in = st.number_input("Discharge (m³/s)", value=12.515)
        diff_in = st.number_input("Diffusivity (m²/s)", value=1.0)

    # Calculations
    area = width * depth
    perimeter = width + 2*depth
    rh = area/perimeter if perimeter > 0 else 0
    vel = Q_in / area if area > 0 else 0
    sugg_diff = 0.01 + vel * width
    
    st.info(f"""
    **Calculated Properties:**
    - Flow Velocity: {vel:.4f} m/s
    - Hydraulic Radius: {rh:.4f} m
    - Suggested Diffusivity (0.01 + u*W): {sugg_diff:.4f} m²/s
    """)

with main_tabs[1]:
    st.header("Atmosphere")
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air Temp (°C)", 20.0)
        wind = st.number_input("Wind Speed (m/s)", 0.0)
        humidity = st.number_input("Humidity (%)", 80.0)
        h_min = st.number_input("h_min", 6.9)
        sky_temp = st.number_input("Sky Temp (°C) (-40=Calc)", -40.0)
    with col2:
        solar = st.number_input("Solar Constant (W/m²)", 1370.0)
        cloud = st.number_input("Cloud Cover (%)", 0.0)
        lat = st.number_input("Latitude", 38.0)
        sunrise = st.number_input("Sunrise (h)", 6.0)
        sunset = st.number_input("Sunset (h)", 18.0)
        
    st.markdown("### Physical Constants (Read-Only)")
    st.dataframe(pd.DataFrame({
        "Parameter": ["O2 Partial Pressure", "CO2 Partial Pressure", "MW O2", "MW CO2"],
        "Value": [0.2095, 0.000395, 32000, 44000],
        "Unit": ["bar", "bar", "mg/mol", "mg/mol"]
    }), hide_index=True)

with main_tabs[2]:
    st.header("Discharges")
    st.caption("Enter Cell OR Distance. Distance takes priority if changed (logic implied).")
    
    # Init data
    if 'dis_df' not in st.session_state:
        st.session_state['dis_df'] = pd.DataFrame({
            "Cell": [1, 60],
            "Distance (m)": [20.0, 2400.0],
            "Flow (m3/s)": [0.0, 0.0],
            "Temp (C)": [30.0, 30.0],
            "BOD": [100.0, 100.0],
            "DO": [0.0, 0.0],
            "CO2": [1.0, 1.0],
            "Generic": [100000.0, 100000.0]
        })
    
    edited = st.data_editor(st.session_state['dis_df'], num_rows="dynamic")
    
    # Sync Logic (Approximate for UI speed)
    # In a real app, use on_change callback. Here we update for next run.
    # We will assume "Cell" is the driver for the engine.
    dx = L / nc if nc > 0 else 1
    # Update Dist column for display based on Cell
    edited["Distance (m)"] = (edited["Cell"] - 0.5) * dx
    st.session_state['dis_df'] = edited

with main_tabs[3]:
    st.header("Constituents")
    c_tabs = st.tabs(["Temperature", "DO", "BOD", "CO2", "Generic"])
    configs = {}
    
    def common_inputs(key, unit, def_val):
        c1, c2 = st.columns(2)
        act = c1.selectbox(f"{key} Active", ["Yes", "No"], key=f"{key}_act")
        val = c2.number_input(f"Default {unit}", value=def_val, key=f"{key}_def")
        
        c3, c4 = st.columns(2)
        min_v = c3.number_input(f"Min {unit}", value=-1000.0 if key=="Temperature" else 0.0, key=f"{key}_min")
        max_v = c4.number_input(f"Max {unit}", value=1000.0, key=f"{key}_max")
        
        st.caption("Boundary Conditions")
        b1, b2 = st.columns(2)
        b_l_t = b1.selectbox("Left BC", ["Fixed", "ZeroGrad", "Cyclic"], key=f"{key}_bcl")
        b_l_v = b1.number_input("Left Val", value=def_val, key=f"{key}_bclv")
        b_r_t = b2.selectbox("Right BC", ["ZeroGrad", "Fixed", "Cyclic"], key=f"{key}_bcr")
        b_r_v = b2.number_input("Right Val", value=def_val, key=f"{key}_bcrv")
        
        st.caption("Initial Conditions (Exceptions)")
        df_ic = pd.DataFrame(columns=["Type", "Start/Idx", "End", "Value"])
        if key == "Temperature": 
             df_ic = pd.DataFrame([["Interval", 0, 1000, 18.0]], columns=["Type", "Start/Idx", "End", "Value"])
        df_ic_ed = st.data_editor(df_ic, num_rows="dynamic", key=f"{key}_ic_ed")
        
        return {
            "active": act=="Yes", "unit": unit, "def": val, "min": min_v, "max": max_v,
            "bclt": b_l_t, "bclv": b_l_v, "bcrt": b_r_t, "bcrv": b_r_v,
            "ic_table": df_ic_ed
        }

    with c_tabs[0]: # Temp
        cfg = common_inputs("Temperature", "ºC", 15.0)
        st.caption("Fluxes")
        c1, c2, c3, c4 = st.columns(4)
        use_surf = c1.checkbox("Surface Flux", True)
        use_sens = c2.checkbox("Sensible", True)
        use_lat = c3.checkbox("Latent", True)
        use_rad = c4.checkbox("Radiative", True)
        cfg.update({"surf": use_surf, "sens": use_sens, "lat": use_lat, "rad": use_rad})
        configs["Temperature"] = cfg

    with c_tabs[1]: # DO
        cfg = common_inputs("DO", "mg/L", 8.0)
        use_flux = st.checkbox("Include Air-Water Exchange", True, key="do_flux")
        cfg["surf"] = use_flux
        configs["DO"] = cfg

    with c_tabs[2]: # BOD
        cfg = common_inputs("BOD", "mg/L", 5.0)
        st.caption("Kinetics")
        c1, c2 = st.columns(2)
        use_log = c1.checkbox("Use Logistic Formulation", False)
        use_ana = c2.checkbox("Consider Anaerobic", False)
        
        k_decay = st.number_input("Decay Rate (1/day)", 0.3)
        half_sat = st.number_input("O2 Semi-Sat Conc (mg/L)", 0.5)
        
        k_grow = 0.0
        max_log = 0.0
        if use_log:
            k_grow = st.number_input("Logistic Growth Rate", 0.5)
            max_log = st.number_input("Max Val Logistic", 50.0)
            
        cfg.update({"log": use_log, "ana": use_ana, "k_dec": k_decay, "k_gro": k_grow, "max_log": max_log, "half": half_sat})
        configs["BOD"] = cfg

    with c_tabs[3]: # CO2
        cfg = common_inputs("CO2", "mg/L", 0.7)
        use_flux = st.checkbox("Include Air-Water Exchange", True, key="co2_flux")
        cfg["surf"] = use_flux
        configs["CO2"] = cfg

    with c_tabs[4]: # Generic
        unit_gen = st.text_input("Unit", "UFC/100ml")
        cfg = common_inputs("Generic", unit_gen, 0.0)
        mode = st.selectbox("Decay Model", ["T90 (Hours)", "Half-Life (Days)", "T-Duplicate (Days)", "Rate (1/day)"])
        val = st.number_input("Parameter Value", 10.0)
        
        # Convert to k (1/sec)
        k_sec = 0.0
        if mode == "T90 (Hours)" and val > 0: k_sec = 2.302585 / (val * 3600)
        elif mode == "Half-Life (Days)" and val > 0: k_sec = 0.693147 / (val * 86400)
        elif mode == "T-Duplicate (Days)" and val > 0: k_sec = -0.693147 / (val * 86400) # Negative for growth? Or positive growth? Usually duplicate = growth.
        elif mode == "Rate (1/day)": k_sec = val / 86400
        
        cfg["k"] = k_sec
        configs["Generic"] = cfg

# =============================================================================
# RUN
# =============================================================================
if run_btn:
    model = RiverModel()
    with st.spinner("Calculating..."):
        # Setup Grid
        model.setup_grid(L, int(nc), width, depth, slope, manning, Q_in, diff_in)
        
        # Setup Atmos
        # Handle sky temp logic (-40 input means calc)
        sky_t_val = sky_temp
        sky_imp = True
        if sky_temp == -40: 
            sky_t_val = -40
            sky_imp = False
            
        model.setup_atmos(air_temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min, sky_t_val, sky_imp)
        
        # Config
        model.config.duration_days = sim_duration
        model.config.dt = dt
        model.config.dt_print = dt_print
        model.config.time_discretisation = time_disc
        model.config.advection_active = (advection=="Yes")
        model.config.advection_type = adv_type
        model.config.quick_up_ratio = quick_ratio
        model.config.diffusion_active = (diffusion=="Yes")
        
        # Discharges
        d_list = []
        for _, r in st.session_state['dis_df'].iterrows():
            d_list.append({
                "cell": int(r["Cell"])-1, "flow": r["Flow (m3/s)"],
                "temp": r["Temp (C)"], "bod": r["BOD"], "do": r["DO"],
                "co2": r["CO2"], "generic": r["Generic"]
            })
        model.set_discharges(d_list)
        
        # Constituents
        for name, c in configs.items():
            # Parse Init table
            i_cells = []
            i_ints = []
            if not c["ic_table"].empty:
                for _, r in c["ic_table"].iterrows():
                    try:
                        if r["Type"] == "Cell": i_cells.append({"idx": int(r["Start/Idx"]), "val": float(r["Value"])})
                        elif r["Type"] == "Interval": i_ints.append({"start": float(r["Start/Idx"]), "end": float(r["End"]), "val": float(r["Value"])})
                    except: pass
            
            # Add to model
            # Map specific params
            k_d = c.get("k_dec", 0.0)
            if name == "Generic": k_d = c.get("k", 0.0)
            
            model.add_constituent(
                name=name, active=c["active"], unit=c["unit"],
                default_val=c["def"], min_val=c["min"], max_val=c["max"],
                init_mode="Mixed", init_cells=i_cells, init_intervals=i_ints,
                bc_left_type=c["bclt"], bc_left_val=c["bclv"],
                bc_right_type=c["bcrt"], bc_right_val=c["bcrv"],
                use_surface_flux=c.get("surf", False),
                use_sensible=c.get("sens", False),
                use_latent=c.get("lat", False),
                use_radiative=c.get("rad", False),
                k_decay=k_d,
                k_growth=c.get("k_gro", 0.0),
                max_logistic=c.get("max_log", 0.0),
                use_logistic=c.get("log", False),
                use_anaerobic=c.get("ana", False),
                o2_half_sat=c.get("half", 0.0)
            )
            
        res = model.run()
        st.session_state['results'] = res
        st.session_state['grid'] = model.grid.xc
        st.success("Simulation Complete")
        st.rerun()

# =============================================================================
# RESULTS
# =============================================================================
with main_tabs[4]:
    if 'results' in st.session_state:
        res = st.session_state['results']
        xc = st.session_state['grid']
        times = res['times']
        
        tab1, tab2 = st.tabs(["Profiles", "Time Series"])
        
        with tab1:
            t_idx = st.slider("Time (Days)", 0, len(times)-1, len(times)-1)
            t_val = times[t_idx]
            for name in configs.keys():
                if name in res and len(res[name]) > 0:
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(xc, res[name][t_idx])
                    ax.set_title(f"{name} at T={t_val:.3f}d")
                    ax.set_xlabel("Distance (m)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
        with tab2:
            x_sel = st.selectbox("Location", xc)
            x_idx = np.argmin(np.abs(xc - x_sel))
            for name in configs.keys():
                if name in res and len(res[name]) > 0:
                    fig, ax = plt.subplots(figsize=(8,3))
                    ts = [step[x_idx] for step in res[name]]
                    ax.plot(times, ts)
                    ax.set_title(f"{name} at X={x_sel:.1f}m")
                    ax.set_xlabel("Time (Days)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
