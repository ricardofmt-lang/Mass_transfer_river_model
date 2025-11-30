# app.py
import streamlit as st
from sim_engine import Grid, FlowProperties, Atmosphere, BoundaryCondition, Discharge, Property, Simulation
from visualization import plot_line, animate_profile
import pandas as pd
import numpy as np

st.title("1D River Transport Simulation")

# --- Input section ---
st.sidebar.header("River Geometry")
length = st.sidebar.number_input("Channel Length (m)", value=1000.0)
width = st.sidebar.number_input("Width (m)", value=50.0)
depth = st.sidebar.number_input("Depth (m)", value=2.0)
NC = st.sidebar.slider("Number of Cells", min_value=10, max_value=500, value=100, step=10)

st.sidebar.header("Flow Conditions")
velocity = st.sidebar.number_input("Velocity (m/s)", value=0.5)
diffusivity = st.sidebar.number_input("Diffusivity (m²/s)", value=1.0)
slope = st.sidebar.number_input("Channel Slope", value=0.001)

st.sidebar.header("Atmospheric Conditions")
air_temp = st.sidebar.number_input("Air Temperature (°C)", value=20.0)
windspeed = st.sidebar.number_input("Wind Speed (m/s)", value=3.0)
humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
solar = st.sidebar.number_input("Solar Radiation (W/m²)", value=300.0)
cloud = st.sidebar.number_input("Cloud Cover (%)", value=20.0)

st.sidebar.header("Simulation Options")
props = st.multiselect("Properties to simulate", 
                       options=["Generic", "Temperature", "DO", "BOD", "CO2"],
                       default=["Generic","Temperature"])
adv_scheme = st.selectbox("Advection scheme", ["upwind", "central", "QUICK"])
time_scheme = st.selectbox("Time discretization", ["explicit", "implicit", "semi-implicit"])
dt = st.number_input("Time step (s)", value=60.0)
duration = st.number_input("Simulation duration (s)", value=3600.0)

# Boundary conditions input (simple example)
st.sidebar.header("Boundary Conditions")
left_val = st.sidebar.number_input("Left boundary value", value=1.0)
right_val = st.sidebar.number_input("Right boundary value", value=0.0)

# --- Run simulation button ---
if st.sidebar.button("Run Simulation"):
    # Initialize grid and flow
    grid = Grid(length, width, depth, NC)
    flow = FlowProperties(velocity, diffusivity, slope)
    flow.compute_numbers(dt, grid.dx)
    atmo = Atmosphere(air_temp, windspeed, humidity, solar, None, cloud, gas_data={})
    
    # Set up simulation
    sim = Simulation(grid, flow, atmo, dt, duration, time_scheme=time_scheme, adv_scheme=adv_scheme)
    
    # Add selected properties with default initial profiles
    for pname in props:
        initial = np.ones(NC) * st.sidebar.number_input(f"Initial {pname}", value=1.0)
        decay = 0.0
        if pname == "BOD": 
            decay = 1.0/(30*24*3600)  # example T90 ~ 30 days
        sim.add_property(Property(pname, initial, decay_rate=decay))
        sim.properties[pname].boundary = BoundaryCondition(0, left_val, 0, right_val)
    
    # (Optionally) add a lateral discharge example
    # If user had a section for discharges, we would parse it here.

    # Run
    results = sim.run()  # dict of name -> list of arrays

    # Display or export results
    # Convert to DataFrame for easy CSV export
    for name, data_list in results.items():
        df = pd.DataFrame(data_list, columns=sim.grid.x)
        st.download_button(f"Download {name} CSV", df.to_csv(index=False), file_name=f"{name}_output.csv")

    # Visualization below...
