import math
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sim_engine import (
    Atmosphere,
    BoundaryCondition,
    Discharge,
    Flow,
    GasExchangeConfig,
    Grid,
    PropertyConfig,
    Simulation,
    SimulationConfig,
)

from visualization import (
    make_space_time_figure,
    make_spatial_profile_figure,
    make_time_series_figure,
    river_topview_animation,
    river_topview_frame,
    property_to_csv,
    results_to_excel,
)


st.set_page_config(
    page_title="1D River Transport Model - Beta version",
    layout="wide",
)

HENRY_TEMPS = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
HENRY_O2 = np.array([0.002181, 0.001913, 0.001696, 0.001524, 0.001384, 0.001263])
HENRY_CO2 = np.array([0.076425, 0.063532, 0.053270, 0.045463, 0.039170, 0.033363])

from dataclasses import dataclass

@dataclass
class Diagnostics:
    courant: float
    diffusion_number: float
    grid_reynolds: float
    res_time_river_days: float
    res_time_cell_seconds: float
    estimated_diffusivity: float


def compute_diagnostics(grid: Grid, flow: Flow, cfg: SimulationConfig) -> Diagnostics:
    """
    Same logic as the Excel sheet:

    Courant  = u * dt / dx
    Diff     = K * dt / dx^2
    Re_grid  = u * dx / K
    Residence times from length / u
    Estimated K ~ 0.1 * U * width
    """
    dt = cfg.dt
    u = flow.velocity
    dx = grid.dx

    courant = flow.courant_number(grid, dt)
    diffusion_number = flow.diffusion_number(grid, dt)

    if flow.diffusivity > 0.0:
        grid_re = abs(u) * dx / flow.diffusivity
    else:
        grid_re = 0.0

    if u != 0.0:
        res_time_cell = dx / abs(u)
        res_time_river = grid.length / abs(u) / 86400.0  # seconds → days
    else:
        res_time_cell = 0.0
        res_time_river = 0.0

    # “Standard” dispersion: K ~ 0.1 * U * width
    estimated_diff = 0.1 * abs(u) * grid.width

    return Diagnostics(
        courant=courant,
        diffusion_number=diffusion_number,
        grid_reynolds=grid_re,
        res_time_river_days=res_time_river,
        res_time_cell_seconds=res_time_cell,
        estimated_diffusivity=estimated_diff,
    )


# ---------------------------------------------------------------------
# SMALL UTILS
# ---------------------------------------------------------------------


def parse_diffusivity(option: str, velocity: float, width: float) -> float:
    """
    Reproduce Excel logic for “standard” eddy diffusivity:
    K ~ 0.1 * U * width. If user enters a value, use it directly.
    """
    option = option.lower()
    if option == "standard":
        return 0.1 * abs(velocity) * width
    try:
        return float(option)
    except ValueError:
        return 0.0


def build_initial_profile(
    x: np.ndarray,
    default_value: float,
    intervals_df: pd.DataFrame | None,
    min_value: float | None,
    max_value: float | None,
) -> np.ndarray:
    """
    Initialise property along x with a default value and optional
    intervals (using x in metres, as in the Generic sheet of Excel).
    """
    x = np.asarray(x, dtype=float)
    C0 = np.full_like(x, default_value, dtype=float)

    if intervals_df is not None and not intervals_df.empty:
        df = intervals_df.copy()
        # ensure needed columns exist
        for col in ["x_start_m", "x_end_m", "value"]:
            if col not in df.columns:
                return C0

        df = df.dropna(subset=["x_start_m", "x_end_m", "value"])
        for _, row in df.iterrows():
            try:
                xs = float(row["x_start_m"])
                xe = float(row["x_end_m"])
                val = float(row["value"])
            except Exception:
                continue

            if xe < xs:
                xs, xe = xe, xs

            mask = (x >= xs) & (x <= xe)
            C0[mask] = val

    # clip to min/max if active
    if min_value is not None:
        C0 = np.maximum(C0, min_value)
    if max_value is not None and max_value > 0.0:
        C0 = np.minimum(C0, max_value)

    return C0


def generic_decay_and_growth(
    k_direct_per_day: float,
    t90_hours: float,
    half_life_days: float,
    t_duplicate_days: float,
    rt_factor: float,
) -> float:
    """
    Convert Generic/E. coli decay/growth parameters to a net first-order
    rate k (1/s) in dC/dt = -k C format (k>0 decay, k<0 growth).

    Follows the formulas in the VBA:
      - T90(Hours):   C(T90) = 0.1 C0
      - HalfLife:     C(Thalf) = 0.5 C0
      - TDuplicate:   C(Tdup) = 2 C0
      - Day-1:        direct 1/day
      - RT:           (RT - 1)/7/86400 (epidemic analogy)
    """
    # precedence similar to the description: if a direct rate is given,
    # use it; otherwise use the times.
    # All are converted to a “Decayrate” in VBA sense (1/s, can be
    # positive or negative in dC/dt = Decayrate*C). We then map to k>0
    # for decay in dC/dt = -k C.
    decayrate = 0.0

    if k_direct_per_day != 0.0:
        decayrate = k_direct_per_day / 86400.0
    elif t90_hours != 0.0:
        decayrate = math.log(0.1) / (t90_hours * 3600.0)
    elif half_life_days != 0.0:
        decayrate = math.log(0.5) / (half_life_days * 86400.0)
    elif t_duplicate_days != 0.0:
        decayrate = math.log(2.0) / (t_duplicate_days * 86400.0)
    elif rt_factor != 0.0:
        decayrate = (rt_factor - 1.0) / 7.0 / 86400.0

    # VBA uses dC/dt = Decayrate * C, with Decayrate possibly negative
    # for decay. Our engine uses dC/dt = -k C with k>0 for decay.
    k = -decayrate
    return k


# ---------------------------------------------------------------------
# SIDEBAR – INPUTS
# ---------------------------------------------------------------------

st.title("1D River Transport Model")
st.write(
    "Configure the river, choose numerical schemes and properties, "
    "then run the simulation and explore the results."
)

# ---------------------- 1. River geometry & flow ---------------------

st.sidebar.header("1. River geometry and flow")

length = st.sidebar.number_input(
    "River length (m)",
    min_value=10.0,
    max_value=1e6,
    value=1000.0,
    step=10.0,
)
width = st.sidebar.number_input(
    "Average width (m)",
    min_value=0.1,
    max_value=1e5,
    value=50.0,
    step=0.5,
)
depth = st.sidebar.number_input(
    "Average depth (m)",
    min_value=0.1,
    max_value=1e4,
    value=3.0,
    step=0.1,
)

nc = st.sidebar.number_input(
    "Number of cells",
    min_value=5,
    max_value=2000,
    value=100,
    step=1,
)

velocity = st.sidebar.number_input(
    "Mean flow velocity (m/s)",
    min_value=0.0,
    max_value=10.0,
    value=0.5,
    step=0.05,
)

slope = st.sidebar.number_input(
    "River bed slope (m/m)",
    min_value=0.0,
    max_value=0.1,
    value=0.001,
    step=0.0005,
)

diffusivity_option = st.sidebar.text_input(
    "Diffusivity (m²/s) or 'standard'",
    value="standard",
)

# ------------------- 2. Time & printing controls --------------------

st.sidebar.header("2. Time and printing controls")

sim_duration_days = st.sidebar.number_input(
    "Simulation duration (days)",
    min_value=0.001,
    max_value=365.0,
    value=1.0,
    step=0.1,
)
dt_seconds = st.sidebar.number_input(
    "Time step Δt (s)",
    min_value=1.0,
    max_value=86400.0,
    value=300.0,
    step=10.0,
)
dt_print_seconds = st.sidebar.number_input(
    "Output interval Δt_print (s)",
    min_value=1.0,
    max_value=86400.0,
    value=3600.0,
    step=60.0,
)

# ------------------- 3. Numerical controls --------------------------

st.sidebar.header("3. Numerical schemes")

advection_scheme_choice = st.sidebar.selectbox(
    "Advection scheme",
    ["upwind", "central", "quick", "quick_up"],
    index=0,
)

quick_up_ratio = 3.0
if advection_scheme_choice in ("quick", "quick_up"):
    quick_up_ratio = st.sidebar.number_input(
        "QUICK_UP ratio (for QUICK schemes)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
    )

quick_up_enabled = advection_scheme_choice == "quick_up"
advection_scheme = (
    "quick" if advection_scheme_choice in ("quick", "quick_up") else advection_scheme_choice
)

time_scheme = st.sidebar.selectbox(
    "Time scheme",
    ["explicit", "implicit", "semi-implicit"],
    index=0,
)

# ------------------- 4. Properties (T, DO, BOD, CO2) ----------------

st.sidebar.header("4. Main properties")

# Base properties as in the original Excel model
property_names_base = ["Temperature", "DO", "BOD", "CO2"]

default_units = {
    "Temperature": "°C",
    "DO": "mg/L",
    "BOD": "mg/L",
    "CO2": "mg/L",
}

# default boundary values
default_left = {
    "Temperature": 20.0,
    "DO": 8.0,
    "BOD": 4.0,
    "CO2": 5.0,
}
default_right = {
    "Temperature": 20.0,
    "DO": 8.0,
    "BOD": 4.0,
    "CO2": 5.0,
}

prop_settings: Dict[str, Dict] = {}

for name in property_names_base:
    with st.sidebar.expander(f"{name} settings", expanded=(name == "DO")):
        active = st.checkbox(f"Include {name}", value=True, key=f"{name}_active")

        units = st.text_input(
            "Units",
            value=default_units[name],
            key=f"{name}_units",
        )

        init_val = st.number_input(
            f"Initial value in river ({units})",
            value=default_left[name],
            key=f"{name}_init",
        )

        # boundaries (Dirichlet or Neumann)
        bc_left_type = st.selectbox(
            "Left boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key=f"{name}_bc_left_type",
        )
        bc_left_val = st.number_input(
            "Left boundary value",
            value=default_left[name],
            key=f"{name}_bc_left_val",
        )

        bc_right_type = st.selectbox(
            "Right boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key=f"{name}_bc_right_type",
        )
        bc_right_val = st.number_input(
            "Right boundary value",
            value=default_right[name],
            key=f"{name}_bc_right_val",
        )

        # min/max
        min_val = st.number_input(
            "Minimum allowed value",
            value=0.0 if name in ("DO", "BOD", "CO2") else -1e9,
            key=f"{name}_min_val",
        )
        max_val = st.number_input(
            "Maximum allowed value (0 = no limit)",
            value=0.0,
            key=f"{name}_max_val",
        )
        if max_val <= 0.0:
            max_val = None

        # initialisation mode and intervals
        init_mode = st.selectbox(
            "Initialisation mode",
            ["Uniform (default)", "Uniform + intervals"],
            index=0,
            key=f"{name}_init_mode",
        )

        intervals_df = None
        if init_mode == "Uniform + intervals":
            st.caption(
                "Define intervals along the river (x in metres) where the "
                "initial value differs from the default."
            )
            intervals_df = st.data_editor(
                pd.DataFrame(columns=["x_start_m", "x_end_m", "value"]),
                num_rows="dynamic",
                key=f"{name}_intervals",
            )

        # reaction parameters
        decay_rate_per_day = 0.0
        growth_rate_per_day = 0.0
        logistic_max = 0.0
        reaer_rate_per_day = 0.0
        eq_conc = 0.0
        oxygen_per_bod = 0.0

        if name == "BOD":
            decay_rate_per_day = st.number_input(
                "BOD decay rate (1/day)",
                value=0.1,
                key="BOD_decay_per_day",
            )
            growth_rate_per_day = st.number_input(
                "BOD growth rate (1/day, 0 = none)",
                value=0.0,
                key="BOD_growth_per_day",
            )
            logistic_max = st.number_input(
                "BOD carrying capacity (mg/L, 0 = none)",
                value=0.0,
                key="BOD_logistic_max",
            )
        elif name == "DO":
            reaer_rate_per_day = st.number_input(
                "DO reaeration rate (1/day)",
                value=0.5,
                key="DO_reaer_per_day",
            )
            eq_conc = st.number_input(
                "DO equilibrium concentration (mg/L)",
                value=8.0,
                key="DO_eq_conc",
            )
            oxygen_per_bod = st.number_input(
                "Stoichiometric O2 per BOD (mgO2/mgBOD)",
                value=1.0,
                key="DO_oxygen_per_bod",
            )
        elif name == "CO2":
            reaer_rate_per_day = st.number_input(
                "CO2 exchange rate (1/day)",
                value=0.1,
                key="CO2_reaer_per_day",
            )
            eq_conc = st.number_input(
                "CO2 equilibrium concentration (mg/L)",
                value=5.0,
                key="CO2_eq_conc",
            )

        prop_settings[name] = {
            "active": active,
            "units": units,
            "init_val": init_val,
            "left_bc_type": bc_left_type,
            "left_bc_val": bc_left_val,
            "right_bc_type": bc_right_type,
            "right_bc_val": bc_right_val,
            "min_val": min_val if min_val > -1e8 else None,
            "max_val": max_val,
            "init_mode": init_mode,
            "intervals_df": intervals_df,
            "decay_rate_per_day": decay_rate_per_day,
            "growth_rate_per_day": growth_rate_per_day,
            "logistic_max": logistic_max,
            "reaer_rate_per_day": reaer_rate_per_day,
            "eq_conc": eq_conc,
            "oxygen_per_bod": oxygen_per_bod,
        }


# ------------------- 5. Atmosphere ----------------------------------

st.sidebar.header("5. Atmosphere (free-surface fluxes)")

atm_use = st.sidebar.checkbox(
    "Include atmosphere (heat + gas exchange at free surface)?",
    value=True,
)

if atm_use:
    air_T = st.sidebar.number_input(
        "Air temperature (°C)",
        min_value=-40.0,
        max_value=60.0,
        value=20.0,
        step=0.5,
    )
    rel_h_pct = st.sidebar.slider(
        "Relative humidity (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
    )
    rel_h = rel_h_pct / 100.0

    cloud_cover = st.sidebar.slider(
        "Cloud cover (0–1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    solar_const = st.sidebar.number_input(
        "Solar constant at top of atmosphere (W/m²)",
        min_value=200.0,
        max_value=1500.0,
        value=1360.0,
        step=10.0,
    )

    latitude = st.sidebar.number_input(
        "Latitude (deg)",
        min_value=-90.0,
        max_value=90.0,
        value=38.0,
        step=0.5,
    )

    h_min = st.sidebar.number_input(
        "Minimum heat transfer coefficient h_min (W/m²/K)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )

    wind_speed = st.sidebar.number_input(
        "Wind speed at surface (m/s)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1,
    )

    sunrise_hour = st.sidebar.number_input(
        "Sunrise hour (0–24)",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.25,
    )

    sunset_hour = st.sidebar.number_input(
        "Sunset hour (0–24)",
        min_value=0.0,
        max_value=24.0,
        value=18.0,
        step=0.25,
    )

    sky_mode = st.sidebar.selectbox(
        "Sky temperature",
        ["use correlation", "impose value"],
        index=0,
    )

    sky_temperature = None
    sky_temperature_imposed = False
    if sky_mode == "impose value":
        sky_temperature = st.sidebar.number_input(
            "Sky temperature (°C)",
            min_value=-80.0,
            max_value=60.0,
            value=10.0,
            step=0.5,
        )
        sky_temperature_imposed = True
else:
    # sensible defaults if atmosphere is off
    air_T = 20.0
    rel_h = 0.7
    cloud_cover = 0.5
    solar_const = 1360.0
    latitude = 38.0
    h_min = 5.0
    wind_speed = 1.0
    sunrise_hour = 6.0
    sunset_hour = 18.0
    sky_temperature = None
    sky_temperature_imposed = False

# ------------------- 6. Generic property (E. coli) -------------------

st.sidebar.header("6. Generic property (e.g. E. coli)")

generic_active = st.sidebar.checkbox(
    "Include Generic property (E. coli)",
    value=False,
    key="Generic_active",
)

generic_settings: Dict | None = None

if generic_active:
    with st.sidebar.expander("Generic (Faecal bacteria) settings", expanded=True):
        generic_units = st.text_input(
            "Units",
            value="UFC/100mL",
            key="Generic_units",
        )

        generic_default = st.number_input(
            "Default/initial value in river",
            value=0.0,
            key="Generic_init",
        )

        gen_min_val = st.number_input(
            "Minimum allowed value",
            value=0.0,
            key="Generic_min_val",
        )
        gen_max_val = st.number_input(
            "Maximum allowed value (0 = no limit)",
            value=0.0,
            key="Generic_max_val",
        )
        if gen_max_val <= 0.0:
            gen_max_val = None

        gen_init_mode = st.selectbox(
            "Initialisation mode",
            ["Uniform (default)", "Uniform + intervals (x in metres)"],
            index=1,
            key="Generic_init_mode",
        )

        gen_intervals_df = None
        if gen_init_mode.startswith("Uniform +"):
            st.caption(
                "Example: cells between x=100–200 m and 600–700 m set to "
                "10,000 UFC/100mL."
            )
            gen_intervals_df = st.data_editor(
                pd.DataFrame(columns=["x_start_m", "x_end_m", "value"]),
                num_rows="dynamic",
                key="Generic_intervals",
            )

        # boundaries (both usually 0)
        gen_left_type = st.selectbox(
            "Left boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key="Generic_bc_left_type",
        )
        gen_left_val = st.number_input(
            "Left boundary value",
            value=0.0,
            key="Generic_bc_left_val",
        )
        gen_right_type = st.selectbox(
            "Right boundary type",
            ["dirichlet", "neumann"],
            index=0,
            key="Generic_bc_right_type",
        )
        gen_right_val = st.number_input(
            "Right boundary value",
            value=0.0,
            key="Generic_bc_right_val",
        )

        st.markdown("**Decay / growth rate (choose one option):**")
        gen_t90_h = st.number_input(
            "T90 (hours) – time for 90% reduction (0 = ignore)",
            value=10.0,
            key="Generic_t90_h",
        )
        gen_half_life_d = st.number_input(
            "Half-life (days, 0 = ignore)",
            value=0.0,
            key="Generic_half_life_d",
        )
        gen_tdup_d = st.number_input(
            "TDuplicate (days, 0 = ignore – >0 implies growth)",
            value=0.0,
            key="Generic_tdup_d",
        )
        gen_k_day = st.number_input(
            "Direct rate (Day⁻¹, 0 = ignore, >0 decay, <0 growth)",
            value=0.0,
            key="Generic_k_day",
        )
        gen_rt_factor = st.number_input(
            "RT factor (0 = ignore)",
            value=0.0,
            key="Generic_rt_factor",
        )

        generic_settings = {
            "active": generic_active,
            "units": generic_units,
            "init_val": generic_default,
            "min_val": gen_min_val if gen_min_val > -1e8 else None,
            "max_val": gen_max_val,
            "init_mode": gen_init_mode,
            "intervals_df": gen_intervals_df,
            "left_bc_type": gen_left_type,
            "left_bc_val": gen_left_val,
            "right_bc_type": gen_right_type,
            "right_bc_val": gen_right_val,
            "t90_h": gen_t90_h,
            "half_life_d": gen_half_life_d,
            "tdup_d": gen_tdup_d,
            "k_day": gen_k_day,
            "rt_factor": gen_rt_factor,
        }

# ------------------- 7. Point discharges -----------------------------

st.sidebar.header("7. Point discharges (optional)")

n_discharges = st.sidebar.number_input(
    "Number of point discharges",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

discharge_settings: List[Dict] = []
# Discharges can affect any active property, including Generic
discharge_properties = property_names_base + (["Generic"] if generic_active else [])

for i in range(n_discharges):
    with st.sidebar.expander(f"Discharge #{i + 1}", expanded=False):
        x_d = st.number_input(
            "Position from upstream (m)",
            min_value=0.0,
            max_value=float(length),
            value=float(length) / 2.0,
            key=f"disc_{i}_x",
        )
        q_d = st.number_input(
            "Flow rate (m³/s)",
            min_value=0.0,
            max_value=1e3,
            value=1.0,
            key=f"disc_{i}_q",
        )

        concs: Dict[str, float] = {}
        for pname in discharge_properties:
            concs[pname] = st.number_input(
                f"{pname} concentration",
                value=(
                    prop_settings.get(pname, {}).get("init_val", 0.0)
                    if pname in prop_settings
                    else (generic_settings or {}).get("init_val", 0.0)
                ),
                key=f"disc_{i}_{pname}_conc",
            )

        discharge_settings.append(
            {"x": x_d, "q": q_d, "concentrations": concs}
        )

# ------------------- 8. Run button ----------------------------------


def build_and_run_simulation():
    # Grid and flow
    grid = Grid(
        length=length,
        width=width,
        depth=depth,
        nc=nc,
    )

    diffusivity = parse_diffusivity(diffusivity_option, velocity, width)
    flow = Flow(
        velocity=velocity,
        diffusivity=diffusivity,
        slope=slope,
        advection_on=(velocity != 0.0),
        diffusion_on=(diffusivity > 0.0),
    )

    # Atmosphere object (for free-surface fluxes)
    atmosphere: Optional[Atmosphere] = None
    if atm_use:
        atmosphere = Atmosphere(
            temperature=air_T,
            humidity=rel_h,
            cloud_cover=cloud_cover,
            solar_constant=solar_const,
            latitude_deg=latitude,
            h_min=h_min,
            wind_speed=wind_speed,
            sky_temperature=sky_temperature,
            sky_temperature_imposed=sky_temperature_imposed,
            sunrise_hour=sunrise_hour,
            sunset_hour=sunset_hour,
        )

    # Simulation config
    cfg = SimulationConfig(
        dt=dt_seconds,
        duration=sim_duration_days * 24.0 * 3600.0,
        output_interval=dt_print_seconds,
        advection_scheme=advection_scheme,
        time_scheme=time_scheme,
        quick_up_enabled=quick_up_enabled,
        quick_up_ratio=quick_up_ratio,
        atmosphere=atmosphere,
    )


    # Property configs
    properties: Dict[str, PropertyConfig] = {}
    initial_profiles: Dict[str, np.ndarray] = {}

    for name in property_names_base:
        cfg_p = prop_settings[name]
        if not cfg_p["active"]:
            continue

        bc = BoundaryCondition(
            left_type=cfg_p["left_bc_type"],
            left_value=cfg_p["left_bc_val"],
            right_type=cfg_p["right_bc_type"],
            right_value=cfg_p["right_bc_val"],
        )

        decay_rate = cfg_p["decay_rate_per_day"] / 86400.0
        growth_rate = cfg_p["growth_rate_per_day"] / 86400.0
        reaer_rate = cfg_p["reaer_rate_per_day"] / 86400.0

        p = PropertyConfig(
            name=name,
            units=cfg_p["units"],
            active=True,
            decay_rate=decay_rate,
            growth_rate=growth_rate,
            logistic_max=cfg_p["logistic_max"],
            reaeration_rate=reaer_rate,
            equilibrium_conc=cfg_p["eq_conc"],
            oxygen_per_bod=cfg_p["oxygen_per_bod"],
            min_value=cfg_p["min_val"],
            max_value=cfg_p["max_val"],
            boundary=bc,
        )
        properties[name] = p

        # Free-surface coupling with atmosphere
        if name == "Temperature":
            p.enable_free_surface_heat_flux = atm_use
            # keep individual components on; can be refined later if needed
            p.enable_sensible_heat_flux = True
            p.enable_latent_heat_flux = True
            p.enable_radiative_heat_flux = True

        if name == "DO":
            p.enable_gas_exchange = atm_use
            if atm_use:
                p.gas_exchange = GasExchangeConfig(
                    henry_temps=HENRY_TEMPS.copy(),
                    henry_constants=HENRY_O2.copy(),
                    partial_pressure_atm=0.2095,       # ~21% O2
                    molecular_mass_mg_per_mol=32_000.0,  # 32 g/mol
                )

        if name == "CO2":
            p.enable_gas_exchange = atm_use
            if atm_use:
                p.gas_exchange = GasExchangeConfig(
                    henry_temps=HENRY_TEMPS.copy(),
                    henry_constants=HENRY_CO2.copy(),
                    partial_pressure_atm=0.0004,       # ~400 ppm CO2
                    molecular_mass_mg_per_mol=44_000.0,  # 44 g/mol
                )

        
        # initial profile with optional intervals
        initial_profiles[name] = build_initial_profile(
            x=grid.x,
            default_value=cfg_p["init_val"],
            intervals_df=cfg_p["intervals_df"],
            min_value=cfg_p["min_val"],
            max_value=cfg_p["max_val"],
        )

    # Generic (E. coli) property
    if generic_settings and generic_settings.get("active", False):
        g = generic_settings

        bc_g = BoundaryCondition(
            left_type=g["left_bc_type"],
            left_value=g["left_bc_val"],
            right_type=g["right_bc_type"],
            right_value=g["right_bc_val"],
        )

        k_generic = generic_decay_and_growth(
            k_direct_per_day=g["k_day"],
            t90_hours=g["t90_h"],
            half_life_days=g["half_life_d"],
            t_duplicate_days=g["tdup_d"],
            rt_factor=g["rt_factor"],
        )

        p_generic = PropertyConfig(
            name="Generic",
            units=g["units"],
            active=True,
            decay_rate=k_generic,
            growth_rate=0.0,
            logistic_max=0.0,
            reaeration_rate=0.0,
            equilibrium_conc=0.0,
            oxygen_per_bod=0.0,
            min_value=g["min_val"],
            max_value=g["max_val"],
            boundary=bc_g,
        )
        properties["Generic"] = p_generic

        initial_profiles["Generic"] = build_initial_profile(
            x=grid.x,
            default_value=g["init_val"],
            intervals_df=g["intervals_df"],
            min_value=g["min_val"],
            max_value=g["max_val"],
        )

    # Discharges
    discharges: List[Discharge] = []
    for ds in discharge_settings:
        if ds["q"] <= 0.0:
            continue
        d_conc: Dict[str, float] = {}
        for pname, val in ds["concentrations"].items():
            if pname in properties:
                d_conc[pname] = float(val)
        if not d_conc:
            continue
        discharges.append(
            Discharge(
                x=float(ds["x"]),
                flow=float(ds["q"]),
                concentrations=d_conc,
            )
        )

    # Build and run simulation
    sim = Simulation(
        grid=grid,
        flow=flow,
        properties=properties,
        config=cfg,
        discharges=discharges,
    )
    results = sim.run(initial_profiles=initial_profiles)
    times = sim.times

    st.session_state["sim_data"] = {
        "grid": grid,
        "flow": flow,
        "cfg": cfg,
        "properties": properties,
        "times": times,
        "results": results,
    }


run_button = st.sidebar.button("Run simulation")

if run_button:
    build_and_run_simulation()

sim_data = st.session_state.get("sim_data", None)

# ---------------------------------------------------------------------
# MAIN LAYOUT – RESULTS
# ---------------------------------------------------------------------

if sim_data is None:
    st.info("Configure the model in the sidebar and click **Run simulation**.")
else:
    grid: Grid = sim_data["grid"]
    flow: Flow = sim_data["flow"]
    sim_cfg: SimulationConfig = sim_data["cfg"]
    properties: Dict[str, PropertyConfig] = sim_data["properties"]
    times: np.ndarray = sim_data["times"]
    results: Dict[str, np.ndarray] = sim_data["results"]

    active_prop_names = [n for n, cfg in properties.items() if cfg.active]

    st.subheader("Simulation results")

    # Diagnostics (Courant, diffusion number, residence times, etc.)
    diagnostics = compute_diagnostics(grid, flow, sim_cfg)
    with st.expander("Diagnostics (Courant, diffusion number, residence time, etc.)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Courant number", f"{diagnostics.courant:.3f}")
            st.metric("Diffusion number", f"{diagnostics.diffusion_number:.3f}")
            st.metric("Grid Reynolds number", f"{diagnostics.grid_reynolds:.3f}")

        with col2:
            st.metric(
                "Residence time in river (days)",
                f"{diagnostics.res_time_river_days:.3f}",
            )
            st.metric(
                "Residence time in one cell (s)",
                f"{diagnostics.res_time_cell_seconds:.3f}",
            )
            st.metric(
                "Estimated diffusivity (m²/s)",
                f"{diagnostics.estimated_diffusivity:.3e}",
            )

    tab1, tab2, tab3 = st.tabs(
        ["Profiles / time series", "Top-view map", "Downloads"]
    )

    # ---------------------- Tab 1: 1D plots --------------------------

    with tab1:
        st.markdown("### 1D space/time plots")

        colA, colB = st.columns(2)

        # spatial profile
        with colA:
            prop_name_1d = st.selectbox(
                "Property for profiles",
                active_prop_names,
                key="prop_1d",
            )
            prop_cfg = properties[prop_name_1d]
            arr = results[prop_name_1d]  # shape (nt, nc)

            t_idx = st.slider(
                "Time index for profile",
                min_value=0,
                max_value=len(times) - 1,
                value=len(times) - 1,
                key="t_idx_profile",
            )
            fig_profile = make_spatial_profile_figure(
                x=grid.x,
                values=arr[t_idx, :],
                name=prop_name_1d,
                units=prop_cfg.units,
            )
            st.plotly_chart(fig_profile, use_container_width=True)

        # time series
        with colB:
            x_sel = st.slider(
                "Location x for time series (m)",
                min_value=float(grid.x[0]),
                max_value=float(grid.x[-1]),
                value=float(grid.x[len(grid.x) // 2]),
                key="x_ts",
            )
            # find nearest cell
            idx = int(np.argmin(np.abs(grid.x - x_sel)))
            fig_ts = make_time_series_figure(
                times=times,
                values=arr[:, idx],
                x_location=grid.x[idx],
                name=prop_name_1d,
                units=prop_cfg.units,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("### Space–time diagram")
        fig_st = make_space_time_figure(
            x=grid.x,
            times=times,
            values_2d=arr,
            name=prop_name_1d,
            units=prop_cfg.units,
        )
        st.plotly_chart(fig_st, use_container_width=True)

    # ---------------------- Tab 2: Top-view map ----------------------

    with tab2:
        st.markdown("### Top-view map of river (2D)")

        prop_name_map = st.selectbox(
            "Property for map",
            active_prop_names,
            key="prop_map",
        )
        prop_cfg_map = properties[prop_name_map]
        arr_map = results[prop_name_map]  # (nt, nc)

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

    # ---------------------- Tab 3: Downloads -------------------------

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
