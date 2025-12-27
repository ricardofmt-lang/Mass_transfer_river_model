import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Grid:
    length: float = 0.0
    nc: int = 0
    dx: float = 0.0
    xc: np.ndarray = field(default_factory=lambda: np.array([]))
    area_vertical: float = 0.0
    water_depth: float = 0.0
    river_width: float = 0.0
    river_slope: float = 0.0
    manning_coef: float = 0.0

@dataclass
class FlowProperties:
    velocity: float = 0.0
    diffusivity: float = 0.0
    discharge: float = 0.0

@dataclass
class Atmosphere:
    air_temp: float = 20.0
    wind_speed: float = 0.0
    humidity: float = 80.0
    h_min: float = 6.9
    solar_constant: float = 1370.0
    latitude: float = 38.0
    sky_temp: float = -40.0
    sky_temp_imposed: bool = False
    cloud_cover: float = 0.0
    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0
    p_o2: float = 0.2095
    p_co2: float = 0.000395
    henry_table_temps: np.ndarray = field(default_factory=lambda: np.array([]))
    henry_table_o2: np.ndarray = field(default_factory=lambda: np.array([]))
    henry_table_co2: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class Constituent:
    name: str
    active: bool
    unit: str
    values: np.ndarray
    old_values: np.ndarray
    boundary_left: float = 0.0  # Dirichlet BC at x=0
    # Specific params
    k_decay: float = 0.0      # For Generic (from T90) or BOD
    reaeration_rate: float = 0.0 

@dataclass
class SimulationConfig:
    duration_days: float = 1.0
    dt: float = 200.0
    dt_print: float = 3600.0
    time_discretisation: str = "semi"
    advection_active: bool = True
    advection_type: str = "QUICK"
    diffusion_active: bool = True

# =============================================================================
# CORE ENGINE
# =============================================================================

class RiverModel:
    def __init__(self):
        self.grid = Grid()
        self.flow = FlowProperties()
        self.atmos = Atmosphere()
        self.config = SimulationConfig()
        self.constituents: Dict[str, Constituent] = {}
        self.discharges = []
        self.current_time = 0.0
        self.results = {"times": []}

    def setup_grid(self, length, nc, width, depth, slope, manning):
        self.grid.length = length
        self.grid.nc = nc
        self.grid.dx = length / nc
        self.grid.xc = np.linspace(self.grid.dx/2, length - self.grid.dx/2, nc)
        self.grid.river_width = width
        self.grid.water_depth = depth
        self.grid.river_slope = slope
        self.grid.manning_coef = manning
        
        # Geometry & Flow
        self.grid.area_vertical = width * depth
        wet_perimeter = width + 2 * depth
        rh = self.grid.area_vertical / wet_perimeter if wet_perimeter > 0 else 0.1
        
        # Manning
        if manning > 0:
            self.flow.velocity = (1.0 / manning) * (rh**(2/3)) * (slope**0.5)
        else:
            self.flow.velocity = 0.0
            
        self.flow.discharge = self.flow.velocity * self.grid.area_vertical
        self.flow.diffusivity = 0.01 + self.flow.velocity * width

    def setup_atmos(self, temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min=6.9, sky_temp=-40, sky_imposed=False):
        self.atmos.air_temp = temp
        self.atmos.wind_speed = wind
        self.atmos.humidity = humidity
        self.atmos.solar_constant = solar
        self.atmos.latitude = lat
        self.atmos.cloud_cover = cloud
        self.atmos.sunrise_hour = sunrise
        self.atmos.sunset_hour = sunset
        self.atmos.h_min = h_min
        self.atmos.sky_temp = sky_temp
        self.atmos.sky_temp_imposed = sky_imposed
        
        # Defaults for Henry's (Temperature, O2, CO2)
        self.atmos.henry_table_temps = np.array([0, 5, 10, 15, 20, 25, 30])
        self.atmos.henry_table_o2 = np.array([0.00218, 0.00191, 0.00170, 0.00152, 0.00138, 0.00126, 0.00116])
        self.atmos.henry_table_co2 = np.array([0.0764, 0.0635, 0.0533, 0.0455, 0.0392, 0.0334, 0.0299])

    def add_constituent(self, name, active, unit, default_val, left_boundary_val, t90=0.0, special_inits=None):
        """
        Adds a constituent to the simulation.
        special_inits: List of dicts {'start_x': float, 'end_x': float, 'value': float}
        """
        vals = np.full(self.grid.nc, default_val)
        
        # Apply special initial conditions (Intervals)
        if special_inits:
            for item in special_inits:
                x1 = item['start_x']
                x2 = item['end_x']
                v = item['value']
                mask = (self.grid.xc >= x1) & (self.grid.xc <= x2)
                vals[mask] = v

        # Calculate decay k for Generic if T90 provided (T90 in hours -> k in 1/s)
        # k = 2.3026 / (T90 * 3600)
        k = 0.0
        if name == "Generic" and t90 > 0:
            k = 2.302585 / (t90 * 3600.0)

        c = Constituent(
            name=name,
            active=active,
            unit=unit,
            values=vals,
            old_values=vals.copy(),
            boundary_left=left_boundary_val,
            k_decay=k
        )
        self.constituents[name] = c
        self.results[name] = []

    def set_discharges(self, discharge_list):
        """
        discharge_list: List of dicts with keys: cell, flow, temp, bod, do, co2, generic
        """
        self.discharges = []
        for d in discharge_list:
            # Ensure 0-based index
            if "cell" in d and d["cell"] >= 0 and d["cell"] < self.grid.nc:
                self.discharges.append(d)

    # -------------------------------------------------------------------------
    # PHYSICS
    # -------------------------------------------------------------------------

    def calculate_henry_constant(self, temp_c, gas_type="O2"):
        vals = self.atmos.henry_table_o2 if gas_type == "O2" else self.atmos.henry_table_co2
        return np.interp(temp_c, self.atmos.henry_table_temps, vals)

    def calculate_saturation(self, temp_c, gas_type):
        p_atm = (self.atmos.p_o2 if gas_type == "O2" else self.atmos.p_co2) / 1.01325
        kh = self.calculate_henry_constant(temp_c, gas_type)
        mol_l = p_atm * kh
        mw = 32000.0 if gas_type == "O2" else 44000.0
        return mol_l * (mw / 1000.0)

    def calculate_heat_fluxes(self, water_temp, time_sec):
        sigma = 5.67e-8
        kelvin = 273.15
        T_w_k = water_temp + kelvin
        T_a_k = self.atmos.air_temp + kelvin
        
        # Solar
        hour = (time_sec / 3600.0) % 24
        Q_sn = 0.0
        if self.atmos.sunrise_hour < hour < self.atmos.sunset_hour:
            day_len = self.atmos.sunset_hour - self.atmos.sunrise_hour
            norm_time = (hour - self.atmos.sunrise_hour) / day_len
            Q_max = self.atmos.solar_constant * (1 - 0.65 * (self.atmos.cloud_cover/100)**2)
            Q_sn = Q_max * math.sin(math.pi * norm_time)
        
        if self.atmos.sky_temp_imposed:
            T_sky = self.atmos.sky_temp + kelvin
        else:
            T_sky = 0.0552 * (T_a_k**1.5)
        Q_an = 0.97 * sigma * (T_sky**4)
        Q_br = 0.97 * sigma * (T_w_k**4)
        
        es_a = 6.11 * 10**((7.5 * self.atmos.air_temp)/(237.3 + self.atmos.air_temp))
        es_w = 6.11 * 10**((7.5 * water_temp)/(237.3 + water_temp))
        ea = es_a * (self.atmos.humidity / 100.0)
        h_conv = self.atmos.h_min + 3.0 * self.atmos.wind_speed 
        Q_h = h_conv * (water_temp - self.atmos.air_temp)
        Q_e = 3.0 * self.atmos.wind_speed * (es_w - ea)
        if Q_e < 0 and (es_w < ea): Q_e = 0
        
        Q_net = (0.9 * Q_sn) + Q_an - Q_br - Q_e - Q_h
        return Q_net / (self.grid.water_depth * 4186000.0)

    def apply_discharges(self, dt_split):
        vol_cell = self.grid.area_vertical * self.grid.dx
        for d in self.discharges:
            idx = d["cell"]
            q = d["flow"]
            if q <= 0: continue
            rate = q / vol_cell
            
            # Heat
            if "Temperature" in self.constituents:
                T_c = self.constituents["Temperature"].values[idx]
                self.constituents["Temperature"].values[idx] += rate * (d["temp"] - T_c) * dt_split
            
            # Others
            for name in ["BOD", "DO", "CO2", "Generic"]:
                if name in self.constituents:
                    C_c = self.constituents[name].values[idx]
                    # Map dict keys to names
                    key = name.lower() 
                    val_in = d.get(key, 0.0)
                    self.constituents[name].values[idx] += rate * (val_in - C_c) * dt_split

    def solve_transport(self, dt):
        u = self.flow.velocity
        D = self.flow.diffusivity
        dx = self.grid.dx
        nc = self.grid.nc
        
        for name, prop in self.constituents.items():
            if not prop.active: continue
            C = prop.values
            C_new = np.zeros_like(C)
            
            # Arrays for TDMA
            a = np.zeros(nc)
            b = np.zeros(nc)
            c_diag = np.zeros(nc) # 'c' of the diag, not speed
            d_rhs = np.zeros(nc)
            
            theta = 0.5 if self.config.time_discretisation == "semi" else 1.0
            if self.config.time_discretisation == "exp": theta = 0.0 # Not fully implemented for exp
            
            # Coefficients
            alpha = u * dt / (2*dx)
            gamma = D * dt / (dx**2)
            
            for i in range(nc):
                if i == 0: 
                    # Left Boundary: Fixed Value (Dirichlet)
                    b[i] = 1.0
                    d_rhs[i] = prop.boundary_left
                elif i == nc - 1:
                    # Right Boundary: Zero Gradient (Neumann) -> C_N - C_N-1 = 0
                    a[i] = -1.0
                    b[i] = 1.0
                    d_rhs[i] = 0.0
                else:
                    # Internal Nodes (Central Difference)
                    a[i] = theta * (-alpha - gamma)
                    b[i] = 1 + theta * (2*gamma)
                    c_diag[i] = theta * (alpha - gamma)
                    
                    # Explicit Part (RHS)
                    adv_ex = -u * (C[i+1] - C[i-1]) / (2*dx)
                    diff_ex = D * (C[i+1] - 2*C[i] + C[i-1]) / (dx**2)
                    d_rhs[i] = C[i] + (1-theta) * dt * (adv_ex + diff_ex)
            
            # TDMA Solver
            for i in range(1, nc):
                m = a[i] / b[i-1]
                b[i] -= m * c_diag[i-1]
                d_rhs[i] -= m * d_rhs[i-1]
            C_new[nc-1] = d_rhs[nc-1] / b[nc-1]
            for i in range(nc-2, -1, -1):
                C_new[i] = (d_rhs[i] - c_diag[i] * C_new[i+1]) / b[i]
            
            prop.values = C_new

    def apply_kinetics(self, dt):
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod = self.constituents["BOD"].values if "BOD" in self.constituents else None
        do = self.constituents["DO"].values if "DO" in self.constituents else None
        co2 = self.constituents["CO2"].values if "CO2" in self.constituents else None
        generic = self.constituents["Generic"].values if "Generic" in self.constituents else None
        
        # 1. Temperature Source
        if "Temperature" in self.constituents:
            for i in range(self.grid.nc):
                self.constituents["Temperature"].values[i] += self.calculate_heat_fluxes(temp[i], self.current_time) * dt

        # 2. Generic Decay (E. coli)
        if generic is not None:
            k = self.constituents["Generic"].k_decay
            if k > 0:
                for i in range(self.grid.nc):
                    # dC/dt = -kC
                    generic[i] -= k * generic[i] * dt

        # 3. BOD & DO Coupling
        if bod is not None and do is not None:
            k1_20 = 0.3
            for i in range(self.grid.nc):
                k1 = k1_20 * (1.047**(temp[i] - 20)) / 86400.0
                dL = k1 * bod[i] * dt
                bod[i] -= dL
                do[i] -= dL
                if co2 is not None: co2[i] += dL * (44.0/32.0)
        
        # 4. Reaeration
        if do is not None:
            for i in range(self.grid.nc):
                # O'Connor Dobbins
                k2 = 3.93 * (self.flow.velocity**0.5 / self.grid.water_depth**1.5) * (1.024**(temp[i] - 20)) / 86400.0
                cs = self.calculate_saturation(temp[i], "O2")
                do[i] += k2 * (cs - do[i]) * dt

        if co2 is not None:
            for i in range(self.grid.nc):
                k2 = 3.93 * (self.flow.velocity**0.5 / self.grid.water_depth**1.5) * (1.024**(temp[i] - 20)) / 86400.0
                cs = self.calculate_saturation(temp[i], "CO2")
                co2[i] += k2 * (cs - co2[i]) * dt

    def step(self):
        dt = self.config.dt
        self.apply_discharges(0.5 * dt)
        self.solve_transport(dt)
        self.apply_discharges(0.5 * dt)
        self.apply_kinetics(dt)
        self.current_time += dt

    def run_simulation(self):
        total_steps = int((self.config.duration_days * 86400) / self.config.dt)
        print_interval = int(self.config.dt_print / self.config.dt)
        if print_interval < 1: print_interval = 1
        
        self.results["times"] = []
        for name in self.constituents:
            self.results[name] = []
            
        for step_n in range(total_steps):
            self.step()
            
            if step_n % print_interval == 0:
                self.results["times"].append(self.current_time / 86400.0)
                for name, prop in self.constituents.items():
                    self.results[name].append(prop.values.copy())
            
            yield step_n / total_steps
            
    def run(self, callback=None):
        runner = self.run_simulation()
        for progress in runner:
            if callback:
                callback(progress)
        return self.results
