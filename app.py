import math
from typing import Dict, List

import numpy as np
import streamlit as st

from sim_engine import (
    BoundaryCondition,
    Discharge,
    Flow,
    Grid,
    PropertyConfig,
    Simulation,
    SimulationConfig,
)
from visualization import (
    make_spatial_profile_figure,
    make_space_time_figure,
    make_time_series_figure,
    property_to_csv,
    results_to_excel,
    river_topview_animation,
    river_topview_frame,
)


st.set_page_config(
    page_title="1D Mass Transfer River Model",
    layout="wide",
)


if "sim_data" not in st.session_state:
    st.session_state["sim_data"] = None


# -------------------------------------------------------------------------
# Helper for “standard” dispersion like in the VBA model
# -------------------------------------------------------------------------


def parse_diffusivity(input_str: str, velocity: float, width: float) -> float:
    s = input_str.strip().lower()
    if s in {"std", "standard", "auto"}:
        # same idea as “standard” option: proportional to U * width
        return 0.1 * abs(velocity) * max(width, 1e-6)
    try:
        return float(s)
    except ValueError:
        raise ValueError(
            "Dispersion must be a number or 'standard' / 'std'."
        )


# -------------------------------------------------------------------------
# Sidebar – configuration
# -------------------------------------------------------------------------


st.sidebar.title("Model setup")

st.sidebar.subheader("1. River geometry")
length = st.sidebar.number_input(
    "River length (m)",
    min_value=10.0,
    value=10_000.0,
    step=100.0,
)
width = st.sidebar.number_input(
    "Average width (m)",
    min_value=0.1,
    value=10.0,
)
depth = st.sidebar.number_input(
    "Average depth (m)",
    min_value=0.1,
    value=2.0,
)
slope = st.sidebar.number_input(
    "Bed slope (m/m)",
    min_value=0.0,
    value=0.001,
    format="%.5f",
)

nc = st.sidebar.number_input(
    "Number of spatial cells",
    min_value=5,
    max_value=500,
    value=50,
    step=1,
)

st.sidebar.subheader("2. Flow & mixing")
velocity = st.sidebar.number_input(
    "Mean velocity U (m/s)",
    value=0.5,
    format="%.3f",
)
diffusivity_input = st.sidebar.text_input(
    "Longitudinal dispersion D (m²/s) or 'standard'",
    value="standard",
)
advection_on = st.sidebar.checkbox("Include advection", value=True)
diffusion_on = st.sidebar.checkbox("Include dispersion", value=True)

st.sidebar.subheader("3. Numerical controls")
dt = st.sidebar.number_input(
    "Time step Δt (s)",
    min_value=0.001,
    value=60.0,
)
duration_h = st.sidebar.number_input(
    "Simulation duration (h)",
    min_value=0.01,
    value=24.0,
)
duration = duration_h * 3600.0

output_every_min = st.sidebar.number_input(
    "Store results every (min)",
    min_value=0.1,
    value=10.0,
)
output_interval = output_every_min * 60.0

advection_scheme = st.sidebar.selectbox(
    "Advection scheme",
    ["upwind", "central", "quick"],
    index=0,
)
time_scheme = st.sidebar.selectbox(
    "Time discretisation",
    ["explicit", "semi-implicit", "implicit"],
    index=0,
)

st.sidebar.subheader("4. Properties (DO, BOD, CO₂, T)")

property_names = ["Temperature", "DO", "BOD", "CO2"]

prop_settings: Dict[str, Dict] = {}

# Default values roughly in line with the teaching examples
default_units = {
    "Temperature": "°C",
    "DO": "mg/L",
    "BOD": "mg/L",
    "CO2": "mg/L",
}
default_init = {
    "Temperature": 20.0,
    "DO": 8.0,
    "BOD": 2.0,
    "CO2": 0.0,
}
default_left = {
    "Temperature": 20.0,
    "DO": 8.0,
    "BOD": 2.0,
    "CO2": 0.0,
}
default_right = {
    "Temperature": 20.0,
    "DO": 8.0,
    "BOD": 2.0,
    "CO2": 0.0,
}

for name in property_names:
    with st.sidebar.expander(f"{name} settings", expanded=(name in ["DO", "BOD"])):
        active = st.checkbox(
            f"Include {name}",
            value=True if name in ["Temperature", "DO", "BOD"] else False,
            key=f"{name}_active",
        )
        units = st.text_input(
            "Units",
            value=default_units[name],
            key=f"{name}_units",
        )
        init_val = st.number_input(
            "Initial value in river",
            value=default_init[name],
            key=f"{name}_init",
        )
        left_bc_val = st.number_input(
            "Upstream boundary value",
            value=default_left[name],
            key=f"{name}_left_bc",
        )
        right_bc_val = st.number_input(
            "Downstream boundary value",
            value=default_right[name],
            key=f"{name}_right_bc",
        )
        left_bc_type = st.selectbox(
            "Left boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key=f"{name}_left_type",
        )
        right_bc_type = st.selectbox(
            "Right boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key=f"{name}_right_type",
        )

        min_val = st.number_input(
            "Minimum value (optional)",
            value=0.0 if name in ["DO", "BOD", "CO2"] else -1000.0,
            key=f"{name}_min_val",
        )
        max_val = st.number_input(
            "Maximum value (0 = no limit)",
            value=0.0,
            key=f"{name}_max_val",
        )

        decay_rate_per_day = 0.0
        reaer_per_day = 0.0
        eq_conc = 0.0
        oxygen_per_bod = 0.0

        if name == "BOD":
            decay_rate_per_day = st.number_input(
                "BOD decay rate k_d (1/day)",
                value=0.1,
                key="BOD_decay_per_day",
            )
        elif name == "DO":
            eq_conc = st.number_input(
                "Equilibrium DO (mg/L)",
                value=8.0,
                key="DO_eq",
            )
            reaer_per_day = st.number_input(
                "Reaeration rate k₂ (1/day)",
                value=0.5,
                key="DO_k2",
            )
            oxygen_per_bod = st.number_input(
                "O₂ required per BOD (mgO₂/mgBOD)",
                value=1.5,
                key="DO_o2_per_bod",
            )
        elif name == "CO2":
            eq_conc = st.number_input(
                "Equilibrium CO₂ (mg/L)",
                value=0.0,
                key="CO2_eq",
            )
            reaer_per_day = st.number_input(
                "Gas exchange rate (1/day)",
                value=0.0,
                key="CO2_k2",
            )

        prop_settings[name] = dict(
            active=active,
            units=units,
            init_val=init_val,
            left_bc_type=left_bc_type,
            left_bc_val=left_bc_val,
            right_bc_type=right_bc_type,
            right_bc_val=right_bc_val,
            min_val=min_val,
            max_val=max_val,
            decay_rate_per_day=decay_rate_per_day,
            reaer_per_day=reaer_per_day,
            eq_conc=eq_conc,
            oxygen_per_bod=oxygen_per_bod,
        )

st.sidebar.subheader("5. Point discharges (optional)")
with st.sidebar.expander("Configure discharges"):
    n_discharges = st.number_input(
        "Number of discharges",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        key="n_discharges",
    )

    discharge_configs: List[Discharge] = []
    for i in range(n_discharges):
        st.markdown(f"**Discharge #{i + 1}**")
        x_d = st.number_input(
            f"Location x (m) for discharge #{i + 1}",
            min_value=0.0,
            max_value=float(length),
            value=float(length) * (i + 1) / (n_discharges + 1) if n_discharges > 0 else 0.0,
            key=f"d{i}_x",
        )
        q_d = st.number_input(
            f"Flow Q (m³/s) for discharge #{i + 1}",
            min_value=0.0,
            value=1.0,
            key=f"d{i}_q",
        )

        concs: Dict[str, float] = {}
        for pname in property_names:
            concs[pname] = st.number_input(
                f"{pname} at discharge #{i + 1}",
                value=default_init[pname],
                key=f"d{i}_{pname}_conc",
            )

        discharge_configs.append(
            Discharge(x=x_d, flow=q_d, concentrations=concs)
        )

# -------------------------------------------------------------------------
# Build and run simulation
# -------------------------------------------------------------------------


run_button = st.sidebar.button("Run simulation", type="primary")


def build_and_run_simulation():
    diffusivity = parse_diffusivity(diffusivity_input, velocity, width)

    grid = Grid(length=length, width=width, depth=depth, nc=int(nc))

    flow = Flow(
        velocity=velocity,
        diffusivity=diffusivity,
        slope=slope,
        advection_on=advection_on,
        diffusion_on=diffusion_on,
    )

    sim_cfg = SimulationConfig(
        dt=dt,
        duration=duration,
        output_interval=output_interval,
        advection_scheme=advection_scheme,
        time_scheme=time_scheme,
    )

    properties: Dict[str, PropertyConfig] = {}
    initial_profiles: Dict[str, np.ndarray] = {}

    for name in property_names:
        cfg = prop_settings[name]
        if not cfg["active"]:
            continue

        bc = BoundaryCondition(
            left_type=cfg["left_bc_type"],
            left_value=cfg["left_bc_val"],
            right_type=cfg["right_bc_type"],
            right_value=cfg["right_bc_val"],
        )

        decay_rate = cfg["decay_rate_per_day"] / 86400.0
        reaer_rate = cfg["reaer_per_day"] / 86400.0

        p = PropertyConfig(
            name=name,
            units=cfg["units"],
            active=True,
            decay_rate=decay_rate,
            reaeration_rate=reaer_rate,
            equilibrium_conc=cfg["eq_conc"],
            oxygen_per_bod=cfg["oxygen_per_bod"] if name == "DO" else 0.0,
            min_value=cfg["min_val"],
            max_value=cfg["max_val"] if cfg["max_val"] > 0.0 else None,
            boundary=bc,
        )
        properties[name] = p

        initial_profiles[name] = np.full(grid.nc, cfg["init_val"], dtype=float)

    discharges = discharge_configs

    sim = Simulation(
        grid=grid,
        flow=flow,
        sim_cfg=sim_cfg,
        properties=properties,
        initial_profiles=initial_profiles,
        discharges=discharges,
    )
    times, results = sim.run()

    st.session_state["sim_data"] = dict(
        grid=grid,
        flow=flow,
        cfg=sim_cfg,
        properties=properties,
        times=times,
        results=results,
    )


if run_button:
    try:
        build_and_run_simulation()
    except Exception as e:
        st.error(f"Error during simulation: {e}")

# -------------------------------------------------------------------------
# Main layout – results
# -------------------------------------------------------------------------


st.title("1D Mass Transfer River Model")

sim_data = st.session_state["sim_data"]

if sim_data is None:
    st.info("Configure the model on the left and click **Run simulation**.")
    st.stop()

grid: Grid = sim_data["grid"]
properties: Dict[str, PropertyConfig] = sim_data["properties"]
times: np.ndarray = sim_data["times"]
results: Dict[str, np.ndarray] = sim_data["results"]

active_prop_names = [n for n, cfg in properties.items() if cfg.active]

if not active_prop_names:
    st.warning("No active properties. Activate at least one in the sidebar.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["1D plots", "River map (top view)", "Downloads"])

# ---------------------- Tab 1: 1D plots ---------------------------------

with tab1:
    st.subheader("1D views")

    prop_name_1d = st.selectbox(
        "Property",
        active_prop_names,
        key="prop_1d",
    )
    prop_cfg = properties[prop_name_1d]
    arr = results[prop_name_1d]  # shape (n_times, nc)

    plot_type = st.radio(
        "Plot type",
        ["Profile at fixed time", "Time series at fixed position", "Space–time map"],
        key="plot_type_1d",
        horizontal=True,
    )

    if plot_type == "Profile at fixed time":
        t_idx = st.slider(
            "Time step index",
            min_value=0,
            max_value=len(times) - 1,
            value=len(times) - 1,
            key="t_idx_profile",
        )
        fig = make_spatial_profile_figure(
            grid.x,
            arr[t_idx, :],
            float(times[t_idx]),
            prop_name_1d,
            prop_cfg.units,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Time series at fixed position":
        x_loc = st.slider(
            "Location along river (m)",
            min_value=float(grid.x[0]),
            max_value=float(grid.x[-1]),
            value=float(grid.x[len(grid.x) // 2]),
            key="x_loc_ts",
        )
        idx = int(np.argmin(np.abs(grid.x - x_loc)))
        fig = make_time_series_figure(
            times,
            arr[:, idx],
            float(grid.x[idx]),
            prop_name_1d,
            prop_cfg.units,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Space–time map
        fig = make_space_time_figure(
            times,
            grid.x,
            arr,
            prop_name_1d,
            prop_cfg.units,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Tab 2: Top-view map -----------------------------

with tab2:
    st.subheader("Top view of river")

    prop_name_map = st.selectbox(
        "Property",
        active_prop_names,
        key="prop_map",
    )
    prop_cfg_map = properties[prop_name_map]
    arr_map = results[prop_name_map]

    mode_map = st.radio(
        "Mode",
        ["Static frame", "Animated"],
        key="mode_map",
        horizontal=True,
    )

    if mode_map == "Static frame":
        t_idx_map = st.slider(
            "Time step index",
            min_value=0,
            max_value=len(times) - 1,
            value=len(times) - 1,
            key="t_idx_map",
        )
        fig = river_topview_frame(
            x=grid.x,
            width=grid.width,
            values=arr_map[t_idx_map, :],
            name=prop_name_map,
            units=prop_cfg_map.units,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        speed = st.slider(
            "Animation speed factor (1 = normal)",
            min_value=0.25,
            max_value=4.0,
            value=1.0,
            step=0.25,
            key="map_speed",
        )
        frame_duration_ms = int(300 / speed)
        fig = river_topview_animation(
            x=grid.x,
            width=grid.width,
            values_2d=arr_map,
            times=times,
            name=prop_name_map,
            units=prop_cfg_map.units,
            frame_duration_ms=frame_duration_ms,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Tab 3: Downloads --------------------------------

with tab3:
    st.subheader("Export results")

    excel_bytes = results_to_excel(
        x=grid.x,
        times=times,
        results=results,
    )
    st.download_button(
        "Download all results as Excel",
        data=excel_bytes,
        file_name="river_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    prop_name_csv = st.selectbox(
        "Property for CSV export",
        active_prop_names,
        key="prop_csv",
    )
    csv_bytes = property_to_csv(
        x=grid.x,
        times=times,
        values_2d=results[prop_name_csv],
    )
    st.download_button(
        f"Download {prop_name_csv} as CSV (time, x, value)",
        data=csv_bytes,
        file_name=f"{prop_name_csv}_results.csv",
        mime="text/csv",
    )
