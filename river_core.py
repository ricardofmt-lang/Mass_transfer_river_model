import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal

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
    # Boundary Conditions
    bc_left_type: str = "Fixed" # Fixed, ZeroGrad, Cyclic
    bc_left_val: float = 0.0
    bc_right_type: str = "ZeroGrad"
    bc_right_val: float = 0.0
    # Kinetics
    k_decay: float = 0.0      
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

    def add_constituent(self, name, active, unit, 
                        init_mode="Default", default_val=0.0, 
                        init_cells=None, init_intervals=None,
                        bc_left_type="Fixed", bc_left_val=0.0,
                        bc_right_type="ZeroGrad", bc_right_val=0.0,
                        k_decay=0.0):
        
        # 1. Initialize Array
        vals = np.full(self.grid.nc, default_val)
        
        # 2. Apply IC Mode
        if init_mode == "Cell" and init_cells is not None:
            # init_cells list of {idx, val}
            for item in init_cells:
                idx = item['idx']
                if 0 <= idx < self.grid.nc:
                    vals[idx] = item['val']
        elif init_mode == "Interval" and init_intervals is not None:
            # init_intervals list of {start, end, val}
            for item in init_intervals:
                x1, x2, v = item['start'], item['end'], item['val']
                mask = (self.grid.xc >= x1) & (self.grid.xc <= x2)
                vals[mask] = v

        c = Constituent(
            name=name,
            active=active,
            unit=unit,
            values=vals,
            old_values=vals.copy(),
            bc_left_type=bc_left_type,
            bc_left_val=bc_left_val,
            bc_right_type=bc_right_type,
            bc_right_val=bc_right_val,
            k_decay=k_decay
        )
        self.constituents[name] = c
        self.results[name] = []

    def set_discharges(self, discharge_list):
        self.discharges = []
        for d in discharge_list:
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
            
            if "Temperature" in self.constituents:
                T_c = self.constituents["Temperature"].values[idx]
                self.constituents["Temperature"].values[idx] += rate * (d["temp"] - T_c) * dt_split
                
            for name in ["BOD", "DO", "CO2", "Generic"]:
                if name in self.constituents:
                    C_c = self.constituents[name].values[idx]
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
            
            # TDMA Arrays
            a = np.zeros(nc)
            b = np.zeros(nc)
            c_diag = np.zeros(nc)
            d_rhs = np.zeros(nc)
            
            theta = 0.5 if self.config.time_discretisation == "semi" else 1.0
            alpha = u * dt / (2*dx)
            gamma = D * dt / (dx**2)
            
            # --- LEFT BOUNDARY (i=0) ---
            if prop.bc_left_type == "Fixed":
                # C[0] = Val
                b[0] = 1.0
                d_rhs[0] = prop.bc_left_val
            elif prop.bc_left_type == "ZeroGrad":
                # C[0] - C[1] = 0
                b[0] = 1.0
                c_diag[0] = -1.0
                d_rhs[0] = 0.0
                
            # --- RIGHT BOUNDARY (i=nc-1) ---
            if prop.bc_right_type == "Fixed":
                # C[N] = Val
                b[nc-1] = 1.0
                d_rhs[nc-1] = prop.bc_right_val
            elif prop.bc_right_type == "ZeroGrad":
                # C[N] - C[N-1] = 0
                a[nc-1] = -1.0
                b[nc-1] = 1.0
                d_rhs[nc-1] = 0.0
            
            # --- INTERNAL NODES ---
            for i in range(1, nc-1):
                # Central Differences Crank-Nicolson
                a[i] = theta * (-alpha - gamma)
                b[i] = 1 + theta * (2*gamma)
                c_diag[i] = theta * (alpha - gamma)
                
                # Explicit Part (n)
                adv_ex = -u * (C[i+1] - C[i-1]) / (2*dx)
                diff_ex = D * (C[i+1] - 2*C[i] + C[i-1]) / (dx**2)
                d_rhs[i] = C[i] + (1-theta) * dt * (adv_ex + diff_ex)
            
            # --- TDMA SOLVER ---
            C_new = np.zeros(nc)
            # Forward Elimination
            # We start from 1 because 0 is the start of the tridiagonal system
            # But specific BC handling might require careful indexing.
            # Standard TDMA: b[i]x[i] + c[i]x[i+1] + a[i]x[i-1] = d[i]
            
            # Since we modified b[0] and c_diag[0], we can proceed standard way
            c_prime = np.zeros(nc)
            d_prime = np.zeros(nc)
            
            # i=0
            c_prime[0] = c_diag[0] / b[0]
            d_prime[0] = d_rhs[0] / b[0]
            
            for i in range(1, nc):
                temp = b[i] - a[i] * c_prime[i-1]
                if temp == 0: temp = 1e-10 # Avoid zero division
                c_prime[i] = c_diag[i] / temp
                d_prime[i] = (d_rhs[i] - a[i] * d_prime[i-1]) / temp
                
            # Back Substitution
            C_new[nc-1] = d_prime[nc-1]
            for i in range(nc-2, -1, -1):
                C_new[i] = d_prime[i] - c_prime[i] * C_new[i+1]
                
            prop.values = C_new

    def apply_kinetics(self, dt):
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod = self.constituents["BOD"].values if "BOD" in self.constituents else None
        do = self.constituents["DO"].values if "DO" in self.constituents else None
        co2 = self.constituents["CO2"].values if "CO2" in self.constituents else None
        generic = self.constituents["Generic"].values if "Generic" in self.constituents else None
        
        if "Temperature" in self.constituents:
            for i in range(self.grid.nc):
                self.constituents["Temperature"].values[i] += self.calculate_heat_fluxes(temp[i], self.current_time) * dt

        # Generic Decay (First Order)
        if generic is not None:
            k = self.constituents["Generic"].k_decay
            if k > 0:
                for i in range(self.grid.nc):
                    generic[i] -= k * generic[i] * dt

        if bod is not None and do is not None:
            k1_20 = 0.3
            for i in range(self.grid.nc):
                k1 = k1_20 * (1.047**(temp[i] - 20)) / 86400.0
                dL = k1 * bod[i] * dt
                bod[i] -= dL
                do[i] -= dL
                if co2 is not None: co2[i] += dL * (44.0/32.0)
                
        if do is not None:
            for i in range(self.grid.nc):
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
