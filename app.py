import streamlit as st
import numpy as np
import pandas as pd

from sim_engine import (
    Grid,
    Flow,
    PropertyConfig,
    BoundaryCondition,
    Discharge,
    SimulationConfig,
    Simulation,
)

from visualization import (
    make_excel_workbook,
    profiles_over_space_figure,
    timeseries_figure,
    curtain_figure,
    river_surface_figure,
)

st.set_page_config(page_title="River Mass Transfer Model", layout="wide")

st.title("1D River Mass Transfer Model (Advection–Diffusion)")


# ----------------------- Sidebar: grid & numerics ----------------------------


st.sidebar.header("Grid and Flow")
length = st.sidebar.number_input("River length (m)", value=1000.0, min_value=1.0)
nc = st.sidebar.slider("Number of cells", min_value=10, max_value=500, value=100, step=10)
width = st.sidebar.number_input("Average width (m)", value=10.0, min_value=0.1)
depth = st.sidebar.number_input("Average depth (m)", value=2.0, min_value=0.1)

velocity = st.sidebar.number_input("Mean velocity (m/s)", value=0.5)
diffusivity = st.sidebar.number_input("Longitudinal diffusivity (m²/s)", value=1.0, min_value=0.0)
diffusion_on = st.sidebar.checkbox("Include diffusion", value=True)

st.sidebar.header("Time and Numerics")
dt = st.sidebar.number_input("Time step Δt (s)", value=60.0, min_value=1e-6, format="%.6f")
duration = st.sidebar.number_input("Total simulation time (s)", value=3600.0, min_value=dt)
output_interval = st.sidebar.number_input("Output interval (s)", value=600.0, min_value=dt)

adv_scheme = st.sidebar.selectbox("Advection scheme", ["upwind", "central", "quick"])
time_scheme = st.sidebar.selectbox("Time discretization", ["explicit", "implicit", "semi-implicit"])


# ----------------------- Properties configuration ---------------------------


st.header("Properties configuration")

default_props = ["Generic", "Temperature", "DO", "BOD", "CO2"]
prop_cfg: dict[str, PropertyConfig] = {}
init_profiles: dict[str, np.ndarray] = {}

# grid preview for initial conditions
grid_preview = Grid(length=length, width=width, depth=depth, nc=nc)
x_centers = grid_preview.x

with st.expander("Global notes / help", expanded=False):
    st.markdown(
        "- All properties use the same **1D grid** and **flow conditions**.\n"
        "- Initial profiles can be `Uniform`, `Gaussian pulse`, or a `Step`.\n"
        "- Decay, reaeration, equilibrium concentration, and BOD–DO coupling are all configurable.\n"
        "- The 2D plots reproduce the typical profiles and space–time plots shown in the course text.:contentReference[oaicite:1]{index=1}"
    )

for pname in default_props:
    with st.expander(f"{pname} settings", expanded=(pname == "Generic")):
        active = st.checkbox(
            f"Activate {pname}",
            value=(pname in ["Generic", "Temperature"]),
            key=f"active_{pname}",
        )
        units = st.text_input(
            f"{pname} units",
            value="mg/L" if pname != "Temperature" else "°C",
            key=f"units_{pname}",
        )

        if not active:
            cfg = PropertyConfig(name=pname, active=False, units=units)
            prop_cfg[pname] = cfg
            continue

        # kinetics
        col1, col2, col3 = st.columns(3)
        with col1:
            decay_rate = st.number_input(
                f"{pname} decay rate k (1/s)",
                value=0.0 if pname != "BOD" else 1.0 / (30 * 24 * 3600),
                format="%.3e",
                key=f"decay_{pname}",
            )
        with col2:
            reaer_rate = st.number_input(
                f"{pname} reaeration / equilibration rate (1/s)",
                value=0.0 if pname not in ["DO", "CO2"] else 1.0 / (2 * 24 * 3600),
                format="%.3e",
                key=f"reaer_{pname}",
            )
        with col3:
            eq_conc = st.number_input(
                f"{pname} equilibrium conc. (for reaeration)",
                value=8.0 if pname == "DO" else 0.0,
                key=f"eq_{pname}",
            )

        oxygen_per_bod = 0.0
        if pname == "DO":
            oxygen_per_bod = st.number_input(
                "O₂ consumption per BOD unit (mg O₂ / mg BOD)",
                value=1.0,
                key="o2_per_bod",
            )

        # bounds
        colb1, colb2 = st.columns(2)
        with colb1:
            min_val = st.number_input(
                f"{pname} minimum (optional)",
                value=0.0,
                key=f"min_{pname}",
            )
        with colb2:
            max_val = st.number_input(
                f"{pname} maximum (optional, 0 = ignore)",
                value=0.0,
                key=f"max_{pname}",
            )
        if max_val <= 0:
            max_val = None

        # boundary conditions
        st.subheader(f"{pname} boundary conditions")
        colL, colR = st.columns(2)
        with colL:
            left_type = st.selectbox(
                f"{pname} left BC type",
                ["dirichlet", "neumann"],
                key=f"bcLtype_{pname}",
            )
            left_val = st.number_input(
                f"{pname} left BC value",
                value=1.0 if pname == "Generic" else 0.0,
                key=f"bcLval_{pname}",
            )
        with colR:
            right_type = st.selectbox(
                f"{pname} right BC type",
                ["dirichlet", "neumann"],
                index=0,
                key=f"bcRtype_{pname}",
            )
            right_val = st.number_input(
                f"{pname} right BC value",
                value=0.0,
                key=f"bcRval_{pname}",
            )

        bc = BoundaryCondition(
            left_type=left_type,
            left_value=left_val,
            right_type=right_type,
            right_value=right_val,
        )

        # initial profile
        st.subheader(f"{pname} initial condition")
        ic_type = st.selectbox(
            f"{pname} initial shape",
            ["Uniform", "Gaussian pulse", "Step from upstream"],
            key=f"ic_type_{pname}",
        )

        if ic_type == "Uniform":
            base = st.number_input(
                f"{pname} uniform value",
                value=0.0,
                key=f"ic_uniform_{pname}",
            )
            profile = np.ones_like(x_centers) * base
        elif ic_type == "Gaussian pulse":
            base = st.number_input(
                f"{pname} background value",
                value=0.0,
                key=f"ic_bg_{pname}",
            )
            peak = st.number_input(
                f"{pname} peak value",
                value=1.0,
                key=f"ic_peak_{pname}",
            )
            center = st.number_input(
                f"{pname} pulse center (m)",
                value=length / 2,
                min_value=0.0,
                max_value=length,
                key=f"ic_center_{pname}",
            )
            width_ic = st.number_input(
                f"{pname} pulse width (m)",
                value=length / 10,
                min_value=1e-6,
                key=f"ic_width_{pname}",
            )
            profile = base + peak * np.exp(-(x_centers - center) ** 2 / (2 * width_ic**2))
        else:  # Step from upstream
            up_val = st.number_input(
                f"{pname} upstream (left) initial value",
                value=1.0,
                key=f"ic_up_{pname}",
            )
            down_val = st.number_input(
                f"{pname} downstream (right) initial value",
                value=0.0,
                key=f"ic_down_{pname}",
            )
            step_pos = st.number_input(
                f"{pname} step position (m)",
                value=length / 2,
                min_value=0.0,
                max_value=length,
                key=f"ic_step_{pname}",
            )
            profile = np.where(x_centers <= step_pos, up_val, down_val)

        if min_val is not None:
            profile = np.maximum(profile, min_val)
        if max_val is not None:
            profile = np.minimum(profile, max_val)

        cfg = PropertyConfig(
            name=pname,
            units=units,
            active=active,
            decay_rate=decay_rate,
            reaeration_rate=reaer_rate,
            equilibrium_conc=eq_conc,
            oxygen_per_bod=oxygen_per_bod if pname == "DO" else 0.0,
            min_value=min_val,
            max_value=max_val,
            boundary=bc,
        )

        prop_cfg[pname] = cfg
        init_profiles[pname] = profile


# ----------------------- Discharges configuration ---------------------------


st.header("Lateral discharges")

num_dis = st.number_input(
    "Number of lateral discharges",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
)
discharges: list[Discharge] = []

if num_dis > 0:
    for k in range(num_dis):
        with st.expander(f"Discharge #{k + 1}", expanded=False):
            x_dis = st.number_input(
                f"Location of discharge #{k + 1} (m from upstream)",
                min_value=0.0,
                max_value=length,
                value=length / 2,
                key=f"x_dis_{k}",
            )
            q_dis = st.number_input(
                f"Flow rate of discharge #{k + 1} (m³/s)",
                min_value=0.0,
                value=1.0,
                key=f"q_dis_{k}",
            )
            concs: dict[str, float] = {}
            for pname in default_props:
                if prop_cfg[pname].active:
                    conc_val = st.number_input(
                        f"{pname} concentration in discharge #{k + 1}",
                        value=1.0 if pname == "Generic" else 0.0,
                        key=f"dis_{k}_{pname}",
                    )
                    concs[pname] = conc_val
            discharges.append(
                Discharge(x=x_dis, flow=q_dis, concentrations=concs)
            )


# ----------------------- Run simulation -------------------------------------


if st.button("Run simulation"):
    active_props = {k: v for k, v in prop_cfg.items() if v.active}

    if not active_props:
        st.error("Please activate at least one property.")
    else:
        grid = Grid(length=length, width=width, depth=depth, nc=nc)
        flow = Flow(
            velocity=velocity,
            diffusivity=diffusivity,
            diffusion_on=diffusion_on,
        )
        sim_cfg = SimulationConfig(
            dt=dt,
            duration=duration,
            output_interval=output_interval,
            advection_scheme=adv_scheme,
            time_scheme=time_scheme,
        )

        init = {name: init_profiles[name] for name in active_props.keys()}

        try:
            sim = Simulation(
                grid=grid,
                flow=flow,
                sim_cfg=sim_cfg,
                properties=active_props,
                initial_profiles=init,
                discharges=discharges,
            )
            times, results = sim.run()
        except Exception as e:
            st.error(f"Error in simulation: {e}")
        else:
            st.success("Simulation finished.")
            st.write(f"Number of output times: {len(times)}")

            # -------------------------------------------------------------
            # 3 PARTS: (1) DOWNLOADS, (2) 2D PLOTS, (3) 3D VISUALISATIONS
            # -------------------------------------------------------------
            tab_2d, tab_3d, tab_dl = st.tabs(["2D plots", "3D view", "Downloads"])

            # -------- (2) 2D PLOTS --------------------------------------
            with tab_2d:
                st.subheader("2D visualisations")

                prop_names_run = list(results.keys())
                prop_2d = st.selectbox(
                    "Property for 2D plots", prop_names_run, key="prop_2d"
                )
                arr2d = results[prop_2d]
                units = active_props[prop_2d].units

                plot_type = st.selectbox(
                    "Type of 2D plot",
                    ["Profiles at selected times", "Time series at position", "Space–time curtain"],
                    key="plot_type_2d",
                )

                if plot_type == "Profiles at selected times":
                    st.markdown("Select up to 4 time indices to compare profiles.")
                    max_idx = len(times) - 1
                    indices = st.multiselect(
                        "Output time indices",
                        options=list(range(len(times))),
                        default=[0, max_idx],
                        help="These are indices into the output list (0 = initial).",
                    )
                    if not indices:
                        indices = [0, max_idx]
                    fig_profiles = profiles_over_space_figure(
                        grid.x, times, arr2d, prop_2d, units, indices
                    )
                    st.plotly_chart(fig_profiles, use_container_width=True)

                elif plot_type == "Time series at position":
                    x_loc = st.number_input(
                        "Spatial location x (m)",
                        value=float(length / 2),
                        min_value=0.0,
                        max_value=float(length),
                    )
                    fig_ts, x_near = timeseries_figure(
                        times, arr2d, prop_2d, units, x_loc, grid.x
                    )
                    st.write(f"Using nearest grid cell at x ≈ {x_near:.2f} m.")
                    st.plotly_chart(fig_ts, use_container_width=True)

                else:  # curtain
                    fig_curtain = curtain_figure(grid.x, times, arr2d, prop_2d, units)
                    st.plotly_chart(fig_curtain, use_container_width=True)

            # -------- (3) 3D VIEW ---------------------------------------
            with tab_3d:
                st.subheader("3D river visualisation")

                prop_names_run = list(results.keys())
                prop_3d = st.selectbox(
                    "Property for 3D view", prop_names_run, key="prop_3d"
                )
                arr3d = results[prop_3d]
                units3d = active_props[prop_3d].units

                t_idx = st.slider(
                    "Select output time index for 3D view",
                    min_value=0,
                    max_value=len(times) - 1,
                    value=len(times) - 1,
                )
                fig3d = river_surface_figure(
                    grid.x, grid.width, arr3d, times, t_idx, prop_3d, units3d
                )
                st.plotly_chart(fig3d, use_container_width=True)

            # -------- (1) DOWNLOADS ------------------------------------
            with tab_dl:
                st.subheader("Download results")

                # CSV per property (students can import into Excel)
                for pname, arr in results.items():
                    units_p = active_props[pname].units
                    df = pd.DataFrame(arr, columns=grid.x)
                    df.insert(0, "time (s)", times)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {pname} as CSV",
                        data=csv,
                        file_name=f"{pname}_results.csv",
                        mime="text/csv",
                    )

                # One Excel workbook with all properties (multi-sheet)
                excel_buf = make_excel_workbook(times, grid.x, results, active_props)
                st.download_button(
                    label="Download all properties as Excel workbook",
                    data=excel_buf,
                    file_name="river_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
