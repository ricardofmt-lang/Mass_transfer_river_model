import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel
import math
import json, base64, zlib

st.set_page_config(page_title="River Water Quality Model", layout="wide")

# ======================================================================
# CONFIGURATION PERSISTENCE
#
# Persist user configuration in the URL query parameters.  The
# configuration dictionary is compressed with zlib and base64 encoded.
# Users can bookmark or share the resulting URL to restore their last
# inputs.  The persistence is per-browser since it relies on the query
# string.
# ======================================================================

def _encode_config(cfg: dict) -> str:
    raw = json.dumps(cfg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(comp).decode("ascii")

def _decode_config(s: str) -> dict:
    try:
        comp = base64.urlsafe_b64decode(s.encode("ascii"))
        raw = zlib.decompress(comp)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

def load_config_from_url(default_cfg: dict) -> dict:
    params = st.query_params
    if "cfg" in params:
        cfg_dict = _decode_config(params.get("cfg"))
        # merge with defaults; values in cfg override defaults
        if isinstance(cfg_dict, dict):
            return {**default_cfg, **cfg_dict}
    return default_cfg.copy()

def save_config_to_url(cfg: dict) -> None:
    # Only update the URL if cfg is not empty
    try:
        st.query_params["cfg"] = _encode_config(cfg)
    except Exception:
        pass

# Default configuration values
DEFAULT_CONFIG = {
    'sim_duration': 2.0,
    'dt': 500.0,
    'dt_print': 2500.0,
    'L': 10000.0,
    'nc': 1000,
    'width': 5.0,
    'depth': 0.5,
    'slope': 0.0001,
    'manning': 0.025,
    'Q_in': 0.1,
    'diff_in': 4.0,
    'air_temp': 20.0,
    'wind': 0.0,
    'humidity': 80.0,
    'h_min': 6.9,
    'sky_temp': 0.0,
    'solar': 1370.0,
    'cloud': 0.0,
    'lat': 38.0,
    'sunrise': 6.0,
    'sunset': 18.0
}

# Initialize session state configuration from URL (or defaults)
if 'config' not in st.session_state:
    st.session_state['config'] = load_config_from_url(DEFAULT_CONFIG)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_step(val):
    if val == 0: return 0.1
    magnitude = 10**math.floor(math.log10(abs(val)))
    return float(magnitude)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("Configuration")
sim_duration = st.sidebar.number_input("Duration (Days)", value=st.session_state['config']['sim_duration'], step=0.1, key="sim_duration")
dt = st.sidebar.number_input("Time Step (s)", value=st.session_state['config']['dt'], step=10.0, key="dt")
# Fix: Force dt_print >= dt
dt_print_val = max(st.session_state['config']['dt_print'], dt)
dt_print = st.sidebar.number_input("Print Interval (s)", value=dt_print_val, min_value=dt, step=100.0, key="dt_print")

st.sidebar.divider()
time_disc = st.sidebar.selectbox("Discretisation", ["semi", "imp", "exp"])
advection = st.sidebar.selectbox("Advection", ["Yes", "No"])
adv_type = "QUICK"
quick_ratio = 4.0

adv_type = st.sidebar.selectbox("Advection Type / Scheme", ["QUICK", "Upwind", "Central"])
if adv_type == "QUICK":
    quick_ratio = st.sidebar.number_input("QUICK UP Ratio", value=4.0, step=0.1)

diffusion = st.sidebar.selectbox("Diffusion", ["Yes", "No"])

st.sidebar.divider()
run_btn = st.sidebar.button("Run Simulation", type="primary")

# -----------------------------------------------------------------------------
# SESSION MANAGEMENT
#
# Provide explicit Save and Reset buttons so users can persist the
# current configuration (without running a simulation) or reset back
# to the default configuration.  Saving writes the configuration to
# the URL query parameter and updates the internal session state.  Reset
# overwrites the session state with DEFAULT_CONFIG and updates the URL.
# -----------------------------------------------------------------------------
with st.sidebar.expander("Session Management", expanded=False):
    # Save current settings into session state and persist to URL
    if st.button("Save Settings"):
        # Update session_state['config'] from the current widget values
        st.session_state['config'] = {
            'sim_duration': st.session_state.get('sim_duration', DEFAULT_CONFIG['sim_duration']),
            'dt': st.session_state.get('dt', DEFAULT_CONFIG['dt']),
            'dt_print': st.session_state.get('dt_print', DEFAULT_CONFIG['dt_print']),
            'L': st.session_state.get('L', DEFAULT_CONFIG['L']),
            'nc': int(st.session_state.get('nc', DEFAULT_CONFIG['nc'])),
            'width': st.session_state.get('width', DEFAULT_CONFIG['width']),
            'depth': st.session_state.get('depth', DEFAULT_CONFIG['depth']),
            'slope': st.session_state.get('slope', DEFAULT_CONFIG['slope']),
            'manning': st.session_state.get('manning', DEFAULT_CONFIG['manning']),
            'Q_in': st.session_state.get('Q_in', DEFAULT_CONFIG['Q_in']),
            'diff_in': st.session_state.get('diff_in', DEFAULT_CONFIG['diff_in']),
            'air_temp': st.session_state.get('air_temp', DEFAULT_CONFIG['air_temp']),
            'wind': st.session_state.get('wind', DEFAULT_CONFIG['wind']),
            'humidity': st.session_state.get('humidity', DEFAULT_CONFIG['humidity']),
            'h_min': st.session_state.get('h_min', DEFAULT_CONFIG['h_min']),
            'sky_temp': st.session_state.get('sky_temp', DEFAULT_CONFIG['sky_temp']),
            'solar': st.session_state.get('solar', DEFAULT_CONFIG['solar']),
            'cloud': st.session_state.get('cloud', DEFAULT_CONFIG['cloud']),
            'lat': st.session_state.get('lat', DEFAULT_CONFIG['lat']),
            'sunrise': st.session_state.get('sunrise', DEFAULT_CONFIG['sunrise']),
            'sunset': st.session_state.get('sunset', DEFAULT_CONFIG['sunset']),
        }
        save_config_to_url(st.session_state['config'])
        st.success("Settings saved to URL")
        st.experimental_rerun()
    # Reset to default configuration and persist
    if st.button("Reset Settings"):
        st.session_state['config'] = DEFAULT_CONFIG.copy()
        # Replace widget state with defaults
        for k, v in DEFAULT_CONFIG.items():
            st.session_state[k] = v
        save_config_to_url(st.session_state['config'])
        st.success("Settings reset to default")
        st.experimental_rerun()

# =============================================================================
# MAIN UI
# =============================================================================
st.title("1D River Water Quality Model")
main_tabs = st.tabs(["River Geometry", "Atmosphere", "Discharges", "Constituents", "Results"])

with main_tabs[0]: 
    st.header("Channel Properties")
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Length (m)", value=st.session_state['config']['L'], step=100.0, key="L")
        nc = st.number_input("Cells", value=st.session_state['config']['nc'], step=10, key="nc")
        width = st.number_input("Width (m)", value=st.session_state['config']['width'], step=10.0, key="width")
        depth = st.number_input("Depth (m)", value=st.session_state['config']['depth'], step=0.1, key="depth")
    with col2:
        slope = st.number_input("Slope (m/m)", value=st.session_state['config']['slope'], format="%.6f", step=0.00001, key="slope")
        manning = st.number_input("Manning n (storage only)", value=st.session_state['config']['manning'], format="%.4f", step=0.001, key="manning")
        Q_in = st.number_input("Discharge (m³/s)", value=st.session_state['config']['Q_in'], step=1.0, key="Q_in")
        
        # Fix: Grey out Diffusivity if diffusion inactive
        diff_disabled = (diffusion == "No")
        diff_in = st.number_input("Diffusivity (m²/s)", value=st.session_state['config']['diff_in'], step=0.1, disabled=diff_disabled, key="diff_in")

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
        air_temp = st.number_input("Air Temp (°C)", value=st.session_state['config']['air_temp'], step=1.0, key="air_temp")
        wind = st.number_input("Wind Speed (m/s)", value=st.session_state['config']['wind'], step=0.1, key="wind")
        humidity = st.number_input("Humidity (%)", value=st.session_state['config']['humidity'], step=1.0, key="humidity")
        h_min = st.number_input("h_min", value=st.session_state['config']['h_min'], step=0.1, key="h_min")
        sky_temp = st.number_input("Sky Temp (°C)", value=st.session_state['config']['sky_temp'], step=1.0, key="sky_temp")
    with col2:
        solar = st.number_input("Solar Constant (W/m²)", value=st.session_state['config']['solar'], step=10.0, key="solar")
        cloud = st.number_input("Cloud Cover (%)", value=st.session_state['config']['cloud'], step=1.0, key="cloud")
        lat = st.number_input("Latitude", value=st.session_state['config']['lat'], step=1.0, key="lat")
        sunrise = st.number_input("Sunrise (h)", value=st.session_state['config']['sunrise'], step=0.5, key="sunrise")
        sunset = st.number_input("Sunset (h)", value=st.session_state['config']['sunset'], step=0.5, key="sunset")
        
    st.markdown("### Physical Constants (Read-Only)")
    st.dataframe(pd.DataFrame({
        "Parameter": ["O2 Partial Pressure", "CO2 Partial Pressure", "MW O2", "MW CO2"],
        "Value": [0.2095, 0.000395, 32000, 44000],
        "Unit": ["bar", "bar", "mg/mol", "mg/mol"]
    }), hide_index=True)

with main_tabs[2]:
    st.header("Discharges")
    if 'dis_df' not in st.session_state:
        st.session_state['dis_df'] = pd.DataFrame({
            "Cell": [1], "Distance (m)": [0.0], "Flow (m3/s)": [0.0],
            "Temp (C)": [20.0], "BOD": [0.0], "DO": [0.0], "CO2": [0.0], "Generic": [0.0]
        })
    dx = L / nc if nc > 0 else 1
    col_mode = st.radio("Edit Mode", ["By Cell", "By Distance"], horizontal=True)
    edited = st.data_editor(st.session_state['dis_df'], num_rows="dynamic")
    if col_mode == "By Cell":
        edited["Distance (m)"] = (edited["Cell"] - 0.5) * dx
    else:
        edited["Cell"] = (edited["Distance (m)"] / dx + 0.5).astype(int)
    st.session_state['dis_df'] = edited

with main_tabs[3]:
    st.header("Constituents")
    c_tabs = st.tabs(["Temperature", "DO", "BOD", "CO2", "Generic"])
    configs = {}
    
    def common_inputs(key, unit, def_val):
        c1, c2 = st.columns(2)
        act = c1.selectbox(f"{key} Active", ["Yes", "No"], key=f"{key}_act")
        is_active = (act == "Yes")
        
        # Disable everything if inactive
        val = c2.number_input(f"Default {unit}", value=def_val, step=get_step(def_val), key=f"{key}_def", disabled=not is_active)
        
        c3, c4 = st.columns(2)
        min_v = c3.number_input(f"Min {unit}", value=-1000.0 if key=="Temperature" else 0.0, step=1.0, key=f"{key}_min", disabled=not is_active)
        max_v = c4.number_input(f"Max {unit}", value=1000.0, step=1.0, key=f"{key}_max", disabled=not is_active)
        
        if is_active:
            st.markdown("##### Boundary Conditions")
            is_temp = (key == "Temperature")
            
            if is_temp:
                is_cyclic = st.checkbox("Cyclic Boundary (Inflow=Outflow)?", value=False, key=f"{key}_cyclic")
            else:
                is_cyclic = False
                
            bc_l_type, bc_r_type = "Fixed", "ZeroGrad"
            b_l_v, b_r_v = 0.0, 0.0
            
            if is_cyclic:
                bc_l_type = "Cyclic"; bc_r_type = "Cyclic"
            else:
                b1, b2 = st.columns(2)
                bc_l_type = b1.selectbox("Left BC Type", ["Fixed", "ZeroGrad"], key=f"{key}_bclt")
                b_l_v = b1.number_input("Left Val", value=def_val, step=get_step(def_val), key=f"{key}_bclv_real")
                bc_r_type = b2.selectbox("Right BC Type", ["ZeroGrad", "Fixed"], key=f"{key}_bcrt")
                b_r_v = b2.number_input("Right Val", value=def_val, step=get_step(def_val), key=f"{key}_bcrv_real")

            st.markdown("##### Initial Conditions")
            ic_mode = st.radio("Initialization", ["Default", "Cell", "Interval"], horizontal=True, key=f"{key}_ic_mode")
            df_ic = pd.DataFrame()
            if ic_mode == "Cell": df_ic = pd.DataFrame(columns=["Cell Index", "Value"])
            elif ic_mode == "Interval": df_ic = pd.DataFrame(columns=["Start (m)", "End (m)", "Value"])
            if ic_mode != "Default":
                df_ic_ed = st.data_editor(df_ic, num_rows="dynamic", key=f"{key}_ic_ed")
            else: df_ic_ed = pd.DataFrame()
        else:
            # Dummies for inactive
            bc_l_type, bc_r_type, b_l_v, b_r_v, ic_mode, df_ic_ed = "Fixed", "ZeroGrad", 0.0, 0.0, "Default", pd.DataFrame()

        return {
            "active": is_active, "unit": unit, "def": val, "min": min_v, "max": max_v,
            "bclt": bc_l_type, "bclv": b_l_v, "bcrt": bc_r_type, "bcrv": b_r_v,
            "ic_mode": ic_mode, "ic_table": df_ic_ed
        }

    with c_tabs[0]: # Temp
        cfg = common_inputs("Temperature", "ºC", 15.0)
        if cfg["active"]:
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
        if cfg["active"]:
            use_flux = st.checkbox("Include Air-Water Exchange", True, key="do_flux")
            cfg["surf"] = use_flux
        configs["DO"] = cfg

    with c_tabs[2]: # BOD
        cfg = common_inputs("BOD", "mg/L", 5.0)
        if cfg["active"]:
            st.caption("Kinetics")
            c1, c2 = st.columns(2)
            use_log = c1.checkbox("Use Logistic Formulation", False)
            use_ana = c2.checkbox("Consider Anaerobic", False)
            k_decay = st.number_input("Decay Rate (1/day)", 0.3, step=0.01)
            half_sat = st.number_input("O2 Semi-Sat Conc (mg/L)", 0.5, step=0.1)
            k_grow = 0.0; max_log = 0.0
            if use_log:
                k_grow = st.number_input("Logistic Growth Rate", 0.5, step=0.1)
                max_log = st.number_input("Max Val Logistic", 50.0, step=1.0)
            cfg.update({"log": use_log, "ana": use_ana, "k_dec": k_decay, "k_gro": k_grow, "max_log": max_log, "half": half_sat})
        configs["BOD"] = cfg

    with c_tabs[3]: # CO2
        cfg = common_inputs("CO2", "mg/L", 0.7)
        if cfg["active"]:
            use_flux = st.checkbox("Include Air-Water Exchange", True, key="co2_flux")
            cfg["surf"] = use_flux
        configs["CO2"] = cfg

    with c_tabs[4]: # Generic
        unit_gen = st.text_input("Unit", "UFC/100ml")
        cfg = common_inputs("Generic", unit_gen, 0.0)
        if cfg["active"]:
            mode = st.selectbox("Decay Model", ["T90 (Hours)", "Half-Life (Days)", "T-Duplicate (Days)", "Rate (1/day)"])
            val = st.number_input("Parameter Value", 10.0, step=1.0)
            k_sec = 0.0
            # Use VBA sign conventions: Ln(0.1) is negative, Ln(0.5) is negative, Ln(2) is positive
            if mode == "T90 (Hours)" and val > 0:
                k_sec = math.log(0.1) / (val * 3600.0)
            elif mode == "Half-Life (Days)" and val > 0:
                k_sec = math.log(0.5) / (val * 86400.0)
            elif mode == "T-Duplicate (Days)" and val > 0:
                k_sec = math.log(2.0) / (val * 86400.0)
            elif mode == "Rate (1/day)":
                # Rate (1/day) is provided as per day; convert to per second
                k_sec = val / 86400.0
            cfg["k"] = k_sec
        configs["Generic"] = cfg

# =============================================================================
# RUN
# =============================================================================
if run_btn:
    model = RiverModel()
    with st.spinner("Calculating..."):
        model.setup_grid(L, int(nc), width, depth, slope, manning, Q_in, diff_in)
        
        # Sky temperature must always be imposed; no automatic calculation
        model.setup_atmos(air_temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min, sky_temp, True)
        
        model.config.duration_days = sim_duration
        model.config.dt = dt
        model.config.dt_print = dt_print
        model.config.time_discretisation = time_disc
        model.config.advection_active = (advection=="Yes")
        model.config.advection_type = adv_type
        model.config.quick_up_ratio = quick_ratio
        model.config.diffusion_active = (diffusion=="Yes")
        
        d_list = []
        for _, r in st.session_state['dis_df'].iterrows():
            d_list.append({
                "cell": int(r["Cell"])-1, "flow": r["Flow (m3/s)"],
                "temp": r["Temp (C)"], "bod": r["BOD"], "do": r["DO"],
                "co2": r["CO2"], "generic": r["Generic"]
            })
        model.set_discharges(d_list)
        
        for name, c in configs.items():
            if not c["active"]: continue # Skip inactive in model
            
            i_cells = []
            i_ints = []
            if not c["ic_table"].empty:
                for _, r in c["ic_table"].iterrows():
                    try:
                        if c["ic_mode"] == "Cell": i_cells.append({"idx": int(r["Cell Index"]), "val": float(r["Value"])})
                        elif c["ic_mode"] == "Interval": i_ints.append({"start": float(r["Start (m)"]), "end": float(r["End (m)"]), "val": float(r["Value"])})
                    except: pass
            
            k_d = c.get("k_dec", 0.0)
            if name == "Generic": k_d = c.get("k", 0.0)
            
            model.add_constituent(
                name=name, active=True, unit=c["unit"],
                default_val=c["def"], min_val=c["min"], max_val=c["max"],
                init_mode=c["ic_mode"], init_cells=i_cells, init_intervals=i_ints,
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
        # Update stored configuration with the latest values
        st.session_state['config'] = {
            'sim_duration': sim_duration,
            'dt': dt,
            'dt_print': dt_print,
            'L': L,
            'nc': int(nc),
            'width': width,
            'depth': depth,
            'slope': slope,
            'manning': manning,
            'Q_in': Q_in,
            'diff_in': diff_in,
            'air_temp': air_temp,
            'wind': wind,
            'humidity': humidity,
            'h_min': h_min,
            'sky_temp': sky_temp,
            'solar': solar,
            'cloud': cloud,
            'lat': lat,
            'sunrise': sunrise,
            'sunset': sunset
        }
        # Persist configuration to URL
        save_config_to_url(st.session_state['config'])
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
        
        tab1, tab2, tab3 = st.tabs(["Profiles", "Time Series", "Space-Time Tables"])
        
        with tab1:
            if len(times) > 0:
                t_idx = st.slider("Time (Days)", 0, len(times)-1, len(times)-1)
                t_val = times[t_idx]
                for name in configs.keys():
                    if configs[name]["active"] and name in res and len(res[name]) > 0:
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.plot(xc, res[name][t_idx])
                        ax.set_title(f"{name} at T={t_val:.3f}d")
                        ax.set_xlabel("Distance (m)")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
        with tab2:
            if len(xc) > 0:
                x_sel = st.selectbox("Location", xc)
                x_idx = np.argmin(np.abs(xc - x_sel))
                for name in configs.keys():
                    if configs[name]["active"] and name in res and len(res[name]) > 0:
                        fig, ax = plt.subplots(figsize=(8,3))
                        ts = [step[x_idx] for step in res[name]]
                        ax.plot(times, ts)
                        ax.set_title(f"{name} at X={x_sel:.1f}m")
                        ax.set_xlabel("Time (Days)")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

        with tab3:
            # Space–Time Tables: show a table of values for a selected constituent over
            # time (rows) and space (columns), including boundary values and
            # initial conditions.  Provides downsampling controls to limit
            # table size for large simulations.
            active_names = [n for n, cfg in configs.items() if cfg["active"] and n in res]
            if active_names:
                sel_name = st.selectbox("Constituent", active_names)
                if sel_name:
                    data = res[sel_name]  # list of 1D arrays
                    n_time = len(data)
                    n_cells = len(xc)
                    # Determine boundary values based on UI config
                    cfg = configs[sel_name]
                    bc_left_type = cfg.get("bclt", "Fixed")
                    bc_left_val = cfg.get("bclv", 0.0)
                    bc_right_type = cfg.get("bcrt", "ZeroGrad")
                    bc_right_val = cfg.get("bcrv", 0.0)
                    # Sliders for downsampling
                    max_rows = st.slider("Max Rows", min_value=5, max_value=min(100, n_time), value=min(30, n_time))
                    max_cols = st.slider("Max Columns", min_value=5, max_value=min(100, n_cells + 2), value=min(30, n_cells + 2))
                    # Determine sampled row indices (always include first and last)
                    if max_rows >= n_time:
                        row_idx = list(range(n_time))
                    else:
                        row_idx = sorted({0, n_time - 1} | set(np.linspace(0, n_time - 1, max_rows, dtype=int)))
                    # Determine sampled column indices for interior cells
                    num_interior_cols = max_cols - 2
                    if num_interior_cols < 1:
                        num_interior_cols = 1
                    if num_interior_cols >= n_cells:
                        col_idx = list(range(n_cells))
                    else:
                        col_idx = sorted(set(np.linspace(0, n_cells - 1, num_interior_cols, dtype=int)))
                    # Build DataFrame rows
                    table_rows = []
                    for i in row_idx:
                        arr = data[i]
                        # compute boundary values
                        if bc_left_type == "Cyclic":
                            left_val = arr[-1]
                        elif bc_left_type == "ZeroGrad":
                            left_val = arr[0]
                        else:
                            left_val = bc_left_val
                        if bc_right_type == "Cyclic":
                            right_val = arr[0]
                        elif bc_right_type == "ZeroGrad":
                            right_val = arr[-1]
                        else:
                            right_val = bc_right_val
                        row = [left_val]
                        for j in col_idx:
                            row.append(arr[j])
                        row.append(right_val)
                        table_rows.append(row)
                    # Column labels: left boundary, interior x positions, right boundary
                    col_labels = ["x=0 (BC)"]
                    for j in col_idx:
                        col_labels.append(f"{xc[j]:.1f}")
                    col_labels.append(f"x={xc[-1] + (xc[1]-xc[0] if len(xc)>1 else 0):.1f} (BC)")
                    df_table = pd.DataFrame(table_rows, columns=col_labels, index=[f"{times[i]:.3f}d" for i in row_idx])
                    st.dataframe(df_table, use_container_width=True)
                    # Download button
                    csv = df_table.to_csv(index_label="Time (d)")
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{sel_name}_space_time.csv",
                        mime="text/csv"
                    )
