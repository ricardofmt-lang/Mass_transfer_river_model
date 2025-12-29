import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from river_core import RiverModel
import math
import json
import base64
import zlib

# =============================================================================
# PER-USER PERSISTENCE (Option A: URL query param)
# Stores a compressed JSON payload in ?cfg=... so each user/browser keeps its
# own last inputs across reloads/reopens (no server-side storage required).
# =============================================================================

def _encode_state(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(comp).decode("ascii")


def _decode_state(s: str) -> dict:
    comp = base64.urlsafe_b64decode(s.encode("ascii"))
    raw = zlib.decompress(comp)
    return json.loads(raw.decode("utf-8"))


def _state_from_url() -> dict | None:
    qp = st.query_params
    if "cfg" not in qp:
        return None
    try:
        return _decode_state(qp["cfg"])
    except Exception:
        return None


def _write_state_to_url(payload: dict) -> None:
    st.query_params["cfg"] = _encode_state(payload)

st.set_page_config(page_title="River Water Quality Model", layout="wide")

DEFAULT_CONFIG = {
    'sim_duration': 1.0, 'dt': 200.0, 'dt_print': 3600.0,
    'L': 12000.0, 'nc': 300, 'width': 100.0, 'depth': 0.5,
    'slope': 0.0001, 'manning': 0.025, 'Q_in': 12.515, 'diff_in': 1.0,
    'air_temp': 20.0, 'wind': 0.0, 'humidity': 80.0, 'h_min': 6.9,
    'sky_temp': -40.0, 'solar': 1370.0, 'cloud': 0.0, 'lat': 38.0,
    'sunrise': 6.0, 'sunset': 18.0,
}

DEFAULT_WIDGETS = {
    **DEFAULT_CONFIG,
    # Sidebar controls
    "time_disc": "semi",
    "advection": "Yes",
    "adv_type": "QUICK",
    "quick_ratio": 4.0,
    "diffusion": "Yes",
    # Generic
    "generic_unit": "UFC/100ml",
    "generic_decay_model": "T90 (Hours)",
    "generic_decay_value": 10.0,
}


def _is_jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


def _restore_persisted_state() -> None:
    """Restore persisted state from URL into st.session_state (must run before widgets)."""
    payload = _state_from_url()
    if not payload or not isinstance(payload, dict):
        return

    widgets = payload.get("widgets", {})
    if isinstance(widgets, dict):
        for k, v in widgets.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # Restore tables
    tables = payload.get("tables", {})
    if isinstance(tables, dict):
        for k, rows in tables.items():
            try:
                if k not in st.session_state and isinstance(rows, list):
                    st.session_state[k] = pd.DataFrame(rows)
            except Exception:
                pass

    # Special case: discharges
    if "dis_df" in payload and "dis_df" not in st.session_state:
        try:
            st.session_state["dis_df"] = pd.DataFrame(payload["dis_df"])
        except Exception:
            pass


def _collect_persistable_state() -> dict:
    """Collect a compact, JSON-safe snapshot of user inputs (excluding results)."""
    widgets = {}
    tables = {}

    for k, v in st.session_state.items():
        if k in {"results", "grid"}:
            continue

        # DataFrames: store only for known editors
        if isinstance(v, pd.DataFrame):
            if k == "dis_df" or k.endswith("_ic_ed"):
                tables[k] = v.to_dict("records")
            continue
        if isinstance(v, np.ndarray):
            continue

        # JSON-safe primitives
        if isinstance(v, (str, int, float, bool, type(None), list, dict)) and _is_jsonable(v):
            widgets[k] = v

    payload = {"v": 1, "widgets": widgets, "tables": tables}

    # Keep dis_df also at top-level for backwards compatibility
    if "dis_df" in tables:
        payload["dis_df"] = tables["dis_df"]

    return payload


# Restore per-user state from URL before creating any widgets
_restore_persisted_state()

# Seed missing widget defaults
for _k, _v in DEFAULT_WIDGETS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Keep the older config dict (used as initial widget defaults in several places)
if 'config' not in st.session_state:
    st.session_state['config'] = DEFAULT_CONFIG.copy()
for _k in DEFAULT_CONFIG.keys():
    # Prefer current widget state if present
    if _k in st.session_state:
        st.session_state['config'][_k] = st.session_state[_k]

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
time_disc = st.sidebar.selectbox("Discretisation", ["semi", "imp", "exp"], key="time_disc")
advection = st.sidebar.selectbox("Advection", ["Yes", "No"], key="advection")
adv_type = "QUICK"
quick_ratio = 4.0

adv_type = st.sidebar.selectbox("Advection Type / Scheme", ["QUICK", "Upwind", "Central"], key="adv_type")
if adv_type == "QUICK":
    quick_ratio = st.sidebar.number_input("QUICK UP Ratio", value=float(st.session_state.get("quick_ratio", 4.0)), step=0.1, key="quick_ratio")

diffusion = st.sidebar.selectbox("Diffusion", ["Yes", "No"], key="diffusion")

st.sidebar.divider()
col_save, col_reset = st.sidebar.columns(2)
save_btn = col_save.button("Save settings")
reset_btn = col_reset.button("Reset link")

if save_btn:
    _write_state_to_url(_collect_persistable_state())
    st.sidebar.success("Saved")

if reset_btn:
    st.query_params.clear()
    # Clearing session_state entirely can break widgets mid-run; instead, reload with defaults.
    for k in list(st.session_state.keys()):
        if k not in {"config"}:
            st.session_state.pop(k, None)
    st.session_state['config'] = DEFAULT_CONFIG.copy()
    st.rerun()

run_btn = st.sidebar.button("Run Simulation", type="primary")

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
        sky_temp = st.number_input("Sky Temp (°C) (-40=Calc)", value=st.session_state['config']['sky_temp'], step=1.0, key="sky_temp")
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
            use_surf = c1.checkbox("Surface Flux", True, key="Temperature_surf")
            use_sens = c2.checkbox("Sensible", True, key="Temperature_sens")
            use_lat = c3.checkbox("Latent", True, key="Temperature_lat")
            use_rad = c4.checkbox("Radiative", True, key="Temperature_rad")
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
            use_log = c1.checkbox("Use Logistic Formulation", False, key="BOD_use_log")
            use_ana = c2.checkbox("Consider Anaerobic", False, key="BOD_use_ana")
            k_decay = st.number_input("Decay Rate (1/day)", 0.3, step=0.01, key="BOD_k_decay")
            half_sat = st.number_input("O2 Semi-Sat Conc (mg/L)", 0.5, step=0.1, key="BOD_half_sat")
            k_grow = 0.0; max_log = 0.0
            if use_log:
                k_grow = st.number_input("Logistic Growth Rate", 0.5, step=0.1, key="BOD_k_grow")
                max_log = st.number_input("Max Val Logistic", 50.0, step=1.0, key="BOD_max_log")
            cfg.update({"log": use_log, "ana": use_ana, "k_dec": k_decay, "k_gro": k_grow, "max_log": max_log, "half": half_sat})
        configs["BOD"] = cfg

    with c_tabs[3]: # CO2
        cfg = common_inputs("CO2", "mg/L", 0.7)
        if cfg["active"]:
            use_flux = st.checkbox("Include Air-Water Exchange", True, key="co2_flux")
            cfg["surf"] = use_flux
        configs["CO2"] = cfg

    with c_tabs[4]: # Generic
        unit_gen = st.text_input("Unit", "UFC/100ml", key="generic_unit")
        cfg = common_inputs("Generic", unit_gen, 0.0)
        if cfg["active"]:
            mode = st.selectbox(
                "Decay Model",
                ["T90 (Hours)", "Half-Life (Days)", "T-Duplicate (Days)", "Rate (1/day)"],
                key="generic_decay_model",
            )
            val = st.number_input("Parameter Value", 10.0, step=1.0, key="generic_decay_value")
            k_sec = 0.0
            if mode == "T90 (Hours)" and val > 0: k_sec = 2.302585 / (val * 3600)
            elif mode == "Half-Life (Days)" and val > 0: k_sec = 0.693147 / (val * 86400)
            elif mode == "T-Duplicate (Days)" and val > 0: k_sec = -0.693147 / (val * 86400) 
            elif mode == "Rate (1/day)": k_sec = val / 86400
            cfg["k"] = k_sec
        configs["Generic"] = cfg

# =============================================================================
# RUN
# =============================================================================
if run_btn:
    model = RiverModel()
    with st.spinner("Calculating..."):
        model.setup_grid(L, int(nc), width, depth, slope, manning, Q_in, diff_in)
        
        sky_t_val = sky_temp
        sky_imp = True
        if sky_temp == -40: 
            sky_t_val = -40
            sky_imp = False
            
        model.setup_atmos(air_temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min, sky_t_val, sky_imp)
        
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
        # Persist latest user inputs automatically whenever a simulation is run.
        # This is per-user (stored in the URL query parameter).
        try:
            _write_state_to_url(_collect_persistable_state())
        except Exception:
            pass
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
                        y = np.asarray(res[name][t_idx])
                        if len(y) != len(xc):
                            st.error(f"Plot error for {name}: profile length {len(y)} does not match grid length {len(xc)}")
                            continue
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.plot(xc, y)
                        ax.set_title(f"{name} at T={t_val:.3f}d")
                        ax.set_xlabel("Distance (m)")
                        ax.set_ylabel(configs[name].get("unit", ""))
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
        with tab2:
            if len(xc) > 0:
                sel_mode = st.radio("Select location by", ["Cell index", "Distance"], horizontal=True, key="ts_loc_mode")
                if sel_mode == "Cell index":
                    cell_1based = st.slider("Cell index", 1, len(xc), 1, key="ts_cell_index")
                    x_idx = int(cell_1based) - 1
                    x_sel = float(xc[x_idx])
                else:
                    step = float(xc[1] - xc[0]) if len(xc) > 1 else 1.0
                    dist_in = st.number_input(
                        "Location (m)",
                        value=float(xc[len(xc)//2]),
                        min_value=float(xc[0]),
                        max_value=float(xc[-1]),
                        step=step,
                        key="ts_distance",
                    )
                    x_idx = int(np.argmin(np.abs(np.asarray(xc) - dist_in)))
                    x_sel = float(xc[x_idx])
                for name in configs.keys():
                    if configs[name]["active"] and name in res and len(res[name]) > 0:
                        fig, ax = plt.subplots(figsize=(8,3))
                        # Ensure we sample consistent array shapes
                        series = []
                        for step_arr in res[name]:
                            step_arr = np.asarray(step_arr)
                            if x_idx >= len(step_arr):
                                series = None
                                break
                            series.append(float(step_arr[x_idx]))
                        if series is None:
                            st.error(f"Plot error for {name}: time-series index {x_idx} out of bounds")
                            continue
                        ts = series
                        ax.plot(times, ts)
                        ax.set_title(f"{name} at X={x_sel:.1f}m")
                        ax.set_xlabel("Time (Days)")
                        ax.set_ylabel(configs[name].get("unit", ""))
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

        with tab3:
            active_names = [
                n for n in configs.keys()
                if configs[n].get("active") and n in res and isinstance(res.get(n), list) and len(res[n]) > 0
            ]

            if not active_names:
                st.info("No active constituents were simulated; enable at least one constituent to view space-time tables.")
            else:
                name = st.selectbox("Constituent", active_names, key="st_table_const")
                mat = np.vstack([np.asarray(v) for v in res[name]])  # (nt, nc)

                c1, c2, c3 = st.columns(3)
                max_cols = int(c1.number_input("Max columns", value=120, min_value=10, max_value=2000, step=10, key="st_table_max_cols"))
                max_rows = int(c2.number_input("Max rows", value=200, min_value=10, max_value=5000, step=10, key="st_table_max_rows"))
                dist_unit = c3.radio("Distance unit", ["m", "km"], horizontal=True, key="st_table_dist_unit")

                col_step = max(1, int(np.ceil(mat.shape[1] / max_cols)))
                row_step = max(1, int(np.ceil(mat.shape[0] / max_rows)))

                mat_ds = mat[::row_step, ::col_step]
                times_ds = np.asarray(times)[::row_step]
                xc_ds = np.asarray(xc)[::col_step]

                if dist_unit == "km":
                    cols = [f"{x/1000:.3f}" for x in xc_ds]
                    col_label = "Distance (km)"
                else:
                    cols = [f"{x:.1f}" for x in xc_ds]
                    col_label = "Distance (m)"

                df = pd.DataFrame(mat_ds, index=[f"{t:.6f}" for t in times_ds], columns=cols)
                df.index.name = "Time (days)"

                st.caption(
                    f"Table is downsampled for readability: every {row_step}th time step and every {col_step}th cell. "
                    f"Columns are {col_label}."
                )
                st.dataframe(df, use_container_width=True, height=520)

                st.download_button(
                    "Download CSV",
                    df.to_csv().encode("utf-8"),
                    file_name=f"{name}_space_time_table.csv",
                    mime="text/csv",
                )
