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
    area_horizontal: float = 0.0
    water_depth: float = 0.0
    river_width: float = 0.0
    river_slope: float = 0.0
    manning_coef: float = 0.0

@dataclass
class FlowProperties:
    velocity: float = 0.0
    diffusivity: float = 0.0
    discharge: float = 0.0
    courant_nr: float = 0.0
    diffusion_nr: float = 0.0
    grid_reynolds_nr: float = 0.0

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
    decay_rate: float = 0.0
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
        # Results storage: Dictionary where keys are constituent names, values are lists of arrays
        self.results = {"times": []}

    # -------------------------------------------------------------------------
    # PARSING LOGIC
    # -------------------------------------------------------------------------
    
    def parse_vertical_params(self, df, key_col_idx, val_col_idx):
        params = {}
        keys = df.iloc[:, key_col_idx].astype(str).str.strip()
        vals = df.iloc[:, val_col_idx]
        for i, key in enumerate(keys):
            if key in ["nan", "None", ""]: continue
            clean_key = key.replace(":", "")
            try:
                params[clean_key] = float(vals.iloc[i])
            except:
                params[clean_key] = vals.iloc[i]
        return params

    def load_main_config(self, df):
        p = self.parse_vertical_params(df, 0, 1)
        self.config.duration_days = float(p.get("SimulationDuration(Days)", 1))
        self.config.dt = float(p.get("TimeStep(Seconds)", 200))
        self.config.dt_print = float(p.get("Dtprint(seconds)", 3600))
        self.config.time_discretisation = str(p.get("TimeDiscretisation", "semi")).lower()
        self.config.advection_active = str(p.get("Advection", "Yes")).lower() == "yes"
        self.config.advection_type = str(p.get("AdvectionType", "QUICK"))
        self.config.diffusion_active = str(p.get("Diffusion", "Yes")).lower() == "yes"

    def load_river_config(self, df):
        p = self.parse_vertical_params(df, 0, 1)
        self.grid.length = float(p.get("ChannelLength", 12000))
        self.grid.nc = int(p.get("NumberOfCells", 300))
        self.grid.dx = self.grid.length / self.grid.nc
        self.grid.xc = np.linspace(self.grid.dx/2, self.grid.length - self.grid.dx/2, self.grid.nc)
        
        self.grid.river_width = float(p.get("RiverWidth", 100))
        self.grid.water_depth = float(p.get("WaterDepth", 0.5))
        self.grid.river_slope = float(p.get("RiverSlope", 0.0001))
        self.grid.manning_coef = float(p.get("ManningCoef(n)", 0.025))
        
        # Geometry & Flow
        self.grid.area_vertical = self.grid.river_width * self.grid.water_depth
        wet_perimeter = self.grid.river_width + 2 * self.grid.water_depth
        rh = self.grid.area_vertical / wet_perimeter
        
        self.flow.velocity = (1.0 / self.grid.manning_coef) * (rh**(2/3)) * (self.grid.river_slope**0.5)
        self.flow.discharge = self.flow.velocity * self.grid.area_vertical
        self.flow.diffusivity = 0.01 + self.flow.velocity * self.grid.river_width

    def load_atmosphere_config(self, df):
        p = self.parse_vertical_params(df, 0, 1)
        self.atmos.air_temp = float(p.get("AirTemperature", 20))
        self.atmos.wind_speed = float(p.get("WindSpeed", 0))
        self.atmos.humidity = float(p.get("AirHumidity", 80))
        self.atmos.h_min = float(p.get("h_min", 6.9))
        self.atmos.solar_constant = float(p.get("SolarConstant", 1370))
        self.atmos.latitude = float(p.get("Latitude", 38))
        self.atmos.cloud_cover = float(p.get("CloudCover", 0))
        
        sky_t = p.get("SkyTemperature", -40)
        self.atmos.sky_temp_imposed = (sky_t != -40 and not pd.isna(sky_t))
        self.atmos.sky_temp = float(sky_t) if self.atmos.sky_temp_imposed else -40.0
            
        self.atmos.sunrise_hour = float(p.get("SunRizeHour", 6))
        self.atmos.sunset_hour = float(p.get("SunSetHour", 18))
        self.atmos.p_o2 = float(p.get("O2PartialPressure", 0.2095))
        self.atmos.p_co2 = float(p.get("CO2PartialPressure", 0.000395))

        # Henry's Table
        start_row = -1
        for i, val in enumerate(df.iloc[:,0].astype(str)):
            if "HenryConstants" in val:
                start_row = i + 2
                break
        if start_row > 0:
            try:
                table = df.iloc[start_row:, 0:3].dropna().astype(float).values
                self.atmos.henry_table_temps = table[:, 0]
                self.atmos.henry_table_o2 = table[:, 1]
                self.atmos.henry_table_co2 = table[:, 2]
            except: pass

    def load_discharges(self, df):
        self.discharges = []
        def get_row_values(key):
            for i, val in enumerate(df.iloc[:,0].astype(str)):
                if key in val: return df.iloc[i, 1:].values
            return None

        locs = get_row_values("DischargeCells")
        flows = get_row_values("DischargeFlowRates")
        temps = get_row_values("DischargeTemperatures")
        bods = get_row_values("DischargeConcentrations_BOD")
        dos = get_row_values("DischargeConcentrations_DO")
        co2s = get_row_values("DischargeConcentrations_CO2")
        generics = get_row_values("DischargeGeneric")

        if locs is not None:
            for i in range(len(locs)):
                try:
                    cell_idx = int(locs[i]) - 1
                    if cell_idx < 0: continue
                    d = {
                        "cell": cell_idx,
                        "flow": float(flows[i]) if flows is not None else 0.0,
                        "temp": float(temps[i]) if temps is not None else 0.0,
                        "bod": float(bods[i]) if bods is not None else 0.0,
                        "do": float(dos[i]) if dos is not None else 0.0,
                        "co2": float(co2s[i]) if co2s is not None else 0.0,
                        "generic": float(generics[i]) if generics is not None else 0.0
                    }
                    self.discharges.append(d)
                except: continue

    def load_constituent(self, name, df):
        p = self.parse_vertical_params(df, 0, 1)
        active = str(p.get("PropertyActive", "No")).lower() == "yes"
        unit = str(p.get("PropertyUnits", "-"))
        default_val = float(p.get("DefaultValue", 0.0))
        
        c = Constituent(
            name=name,
            active=active,
            unit=unit,
            values=np.full(self.grid.nc, default_val),
            old_values=np.full(self.grid.nc, default_val)
        )
        
        # Cell Init
        col0 = df.iloc[:,0].astype(str)
        try:
            cell_start = -1
            for i, val in enumerate(col0):
                if "CellNumber" in val: 
                    cell_start = i + 1
                    break
            if cell_start > 0:
                for i in range(cell_start, len(df)):
                    try:
                        c_idx = int(df.iloc[i, 0]) - 1
                        c_val = float(df.iloc[i, 1])
                        if 0 <= c_idx < self.grid.nc:
                            c.values[c_idx] = c_val
                            c.old_values[c_idx] = c_val
                    except: pass
        except: pass

        # Interval Init
        try:
            interval_start = -1
            for i, val in enumerate(col0):
                if "intervalValues" in val:
                    interval_start = i + 1
                    break
            if interval_start > 0:
                for i in range(interval_start, len(df)):
                    try:
                        row = df.iloc[i, :].astype(str).values
                        x1, x2, val = float(row[0]), float(row[1]), float(row[2])
                        mask = (self.grid.xc >= x1) & (self.grid.xc <= x2)
                        c.values[mask] = val
                        c.old_values[mask] = val
                    except: pass
        except: pass
            
        self.constituents[name] = c
        self.results[name] = [] # Init result storage

    # -------------------------------------------------------------------------
    # PHYSICS
    # -------------------------------------------------------------------------

    def calculate_henry_constant(self, temp_c, gas_type="O2"):
        if len(self.atmos.henry_table_temps) == 0:
            return 0.001 if gas_type=="O2" else 0.03
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
            
        # Atmos Radiation
        if self.atmos.sky_temp_imposed:
            T_sky = self.atmos.sky_temp + kelvin
        else:
            T_sky = 0.0552 * (T_a_k**1.5)
        Q_an = 0.97 * sigma * (T_sky**4)
        
        # Back Radiation
        Q_br = 0.97 * sigma * (T_w_k**4)
        
        # Evap & Sensible
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
            if idx >= self.grid.nc: continue
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
                    if key == "generic": key = "generic"
                    self.constituents[name].values[idx] += rate * (d.get(key,0) - C_c) * dt_split

    def solve_transport(self, dt):
        u = self.flow.velocity
        D = self.flow.diffusivity
        dx = self.grid.dx
        nc = self.grid.nc
        
        for name, prop in self.constituents.items():
            if not prop.active: continue
            C = prop.values
            C_new = np.zeros_like(C)
            
            # Semi-Implicit TDMA
            a, b, c_diag, d_rhs = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
            theta = 0.5
            
            for i in range(nc):
                if i == 0: # Left BC
                    b[i] = 1.0
                    d_rhs[i] = prop.values[0]
                elif i == nc - 1: # Right BC (Zero Gradient)
                    a[i], b[i], d_rhs[i] = -1.0, 1.0, 0.0
                else:
                    # Central Differences
                    alpha = u * dt / (2*dx)
                    gamma = D * dt / (dx**2)
                    
                    a[i] = theta * (-alpha - gamma)
                    b[i] = 1 + theta * (2*gamma)
                    c_diag[i] = theta * (alpha - gamma)
                    
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
        
        # Heat
        if "Temperature" in self.constituents:
            for i in range(self.grid.nc):
                self.constituents["Temperature"].values[i] += self.calculate_heat_fluxes(temp[i], self.current_time) * dt
                
        # Bio
        if bod is not None and do is not None:
            k1_20 = 0.3
            for i in range(self.grid.nc):
                k1 = k1_20 * (1.047**(temp[i] - 20)) / 86400.0
                dL = k1 * bod[i] * dt
                bod[i] -= dL
                do[i] -= dL
                if co2 is not None: co2[i] += dL * (44.0/32.0)
                
        # Reaeration
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

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------

    def run_simulation(self):
        """Generator for step-by-step execution (UI friendly)"""
        total_steps = int((self.config.duration_days * 86400) / self.config.dt)
        print_interval = int(self.config.dt_print / self.config.dt)
        
        # Init Results keys
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
        """Compatibility method: runs full simulation, optionally calling callback(progress)"""
        runner = self.run_simulation()
        for progress in runner:
            if callback:
                callback(progress)
        return self.results
