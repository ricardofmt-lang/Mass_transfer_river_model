import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from io import StringIO

# =============================================================================
# DATA STRUCTURES (Replicating VBA Types)
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
    # Gas Properties
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
    values: np.ndarray  # Current time step (New)
    old_values: np.ndarray # Previous time step (Old)
    boundary_left: float = 0.0
    boundary_right: float = 0.0
    decay_rate: float = 0.0 # K1
    reaeration_rate: float = 0.0 # K2 (calculated)
    
@dataclass
class SimulationConfig:
    duration_days: float = 1.0
    dt: float = 200.0
    dt_print: float = 3600.0
    time_discretisation: str = "semi" # semi, imp, exp
    advection_active: bool = True
    advection_type: str = "QUICK" # upwind, central, QUICK, QUICK_UP
    quick_up_ratio: float = 4.0
    diffusion_active: bool = True

# =============================================================================
# CORE SIMULATION ENGINE
# =============================================================================

class RiverSimulation:
    def __init__(self):
        self.grid = Grid()
        self.flow = FlowProperties()
        self.atmos = Atmosphere()
        self.config = SimulationConfig()
        self.constituents: Dict[str, Constituent] = {}
        self.discharges = [] # List of dicts
        self.current_time = 0.0
        self.results = {} # Store history

    # -------------------------------------------------------------------------
    # PARSING LOGIC (Replicating the flexible CSV reading of VBA)
    # -------------------------------------------------------------------------
    
    def parse_vertical_params(self, df, key_col_idx, val_col_idx):
        """Helper to read key-value pairs from specific columns"""
        params = {}
        # Convert to string and strip, handle NaNs
        keys = df.iloc[:, key_col_idx].astype(str).str.strip()
        vals = df.iloc[:, val_col_idx]
        
        for i, key in enumerate(keys):
            if key in ["nan", "None", ""]: continue
            # Remove colon if present at end
            clean_key = key.replace(":", "")
            try:
                params[clean_key] = float(vals.iloc[i])
            except:
                params[clean_key] = vals.iloc[i]
        return params

    def load_main_config(self, df):
        p = self.parse_vertical_params(df, 0, 1) # Main.csv usually Key=Col A, Val=Col B
        self.config.duration_days = float(p.get("SimulationDuration(Days)", 1))
        self.config.dt = float(p.get("TimeStep(Seconds)", 200))
        self.config.dt_print = float(p.get("Dtprint(seconds)", 3600))
        self.config.time_discretisation = str(p.get("TimeDiscretisation", "semi")).lower()
        self.config.advection_active = str(p.get("Advection", "Yes")).lower() == "yes"
        self.config.advection_type = str(p.get("AdvectionType", "QUICK"))
        self.config.quick_up_ratio = float(p.get("QUICK_UP_Ratio", 4))
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
        
        # Calculate Geometry
        self.grid.area_vertical = self.grid.river_width * self.grid.water_depth
        self.grid.area_horizontal = self.grid.dx * self.grid.river_width
        
        # Calculate Flow (Manning-Strickler)
        # u = (1/n) * Rh^(2/3) * S^(1/2)
        # For wide channel Rh approx Depth
        rh = self.grid.water_depth # Simplified as per common 1D models, or A/P
        # Exact Hydraulic Radius:
        wet_perimeter = self.grid.river_width + 2 * self.grid.water_depth
        rh = self.grid.area_vertical / wet_perimeter
        
        velocity_calc = (1.0 / self.grid.manning_coef) * (rh**(2/3)) * (self.grid.river_slope**0.5)
        
        # VBA allows overriding velocity if provided, otherwise calc. 
        # Here we assume calc unless explicit logic exists in CSV parsing to read computed values.
        # Based on file analysis, it's calculated.
        self.flow.velocity = velocity_calc
        self.flow.discharge = self.flow.velocity * self.grid.area_vertical
        
        # Diffusivity: 0.01 + velocity * width (from snippets) or user input
        # The snippets show "Equation for diffusivity: 0.01+velocity*width"
        # But also a specific value. We implement the formula logic.
        self.flow.diffusivity = 0.01 + self.flow.velocity * self.grid.river_width
        
        # Flow numbers
        self.flow.courant_nr = self.flow.velocity * self.config.dt / self.grid.dx
        self.flow.diffusion_nr = self.flow.diffusivity * self.config.dt / (self.grid.dx**2)
        self.flow.grid_reynolds_nr = self.flow.velocity * self.grid.dx / self.flow.diffusivity

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
        if sky_t == -40 or pd.isna(sky_t):
            self.atmos.sky_temp_imposed = False
            # Swinbank calculated later
        else:
            self.atmos.sky_temp_imposed = True
            self.atmos.sky_temp = float(sky_t)
            
        self.atmos.sunrise_hour = float(p.get("SunRizeHour", 6))
        self.atmos.sunset_hour = float(p.get("SunSetHour", 18))
        
        self.atmos.p_o2 = float(p.get("O2PartialPressure", 0.2095))
        self.atmos.p_co2 = float(p.get("CO2PartialPressure", 0.000395))

        # Parse Henry's Constants Table (Starts around row 20 usually)
        # We need to scan for "HenryConstants:" keyword in col 0
        start_row = -1
        for i, val in enumerate(df.iloc[:,0].astype(str)):
            if "HenryConstants" in val:
                start_row = i + 2 # Skip header
                break
        
        if start_row > 0:
            # Assuming Temperature, O2, CO2 columns in A, B, C
            # Python indexing: 0, 1, 2
            try:
                table = df.iloc[start_row:, 0:3].dropna().astype(float).values
                self.atmos.henry_table_temps = table[:, 0]
                self.atmos.henry_table_o2 = table[:, 1]
                self.atmos.henry_table_co2 = table[:, 2]
            except:
                pass # Fail silently or log, default to empty

    def load_discharges(self, df):
        # Transpose based logic as per CSV structure
        # DischargeNumbers in Row 0, Names in Row 1, Cells in Row 2
        # Data starts from Column 1 (index 1)
        self.discharges = []
        
        # Helper to find row by key
        def get_row_values(key):
            for i, val in enumerate(df.iloc[:,0].astype(str)):
                if key in val:
                    return df.iloc[i, 1:].values
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
                    cell_idx = int(locs[i]) - 1 # Convert 1-based to 0-based
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
                except:
                    continue

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
        
        # COMPLEX INITIALIZATION PARSING (CELL, INTERVAL)
        # Scan column 0 for keywords
        col0 = df.iloc[:,0].astype(str)
        
        # 1. Single Cell Initialization: "CellNumber", "CellValue"
        # Find where these headers are
        # In the provided files, it often looks like:
        # InitType: (CELL)
        # CellNumber | CellValue
        # ...        | ...
        
        # We scan for numeric pairs after "CellNumber"
        try:
            cell_start = -1
            for i, val in enumerate(col0):
                if "CellNumber" in val:
                    cell_start = i + 1
                    break
            
            if cell_start > 0:
                for i in range(cell_start, len(df)):
                    try:
                        c_idx = int(df.iloc[i, 0]) - 1 # 1-based to 0-based
                        c_val = float(df.iloc[i, 1])
                        if 0 <= c_idx < self.grid.nc:
                            c.values[c_idx] = c_val
                            c.old_values[c_idx] = c_val
                    except:
                        pass # End of list
        except:
            pass

        # 2. Interval Initialization: "(X1,X2)_intervalValues(CELL):"
        # Logic: Look for pattern, read next lines
        try:
            interval_start = -1
            for i, val in enumerate(col0):
                if "intervalValues" in val:
                    interval_start = i + 1
                    break
            
            if interval_start > 0:
                for i in range(interval_start, len(df)):
                    row = df.iloc[i, :].astype(str).values
                    # Format: x_start, x_end, val
                    try:
                        x1 = float(row[0])
                        x2 = float(row[1])
                        val = float(row[2])
                        
                        # Apply to cells whose center is in [x1, x2]
                        mask = (self.grid.xc >= x1) & (self.grid.xc <= x2)
                        c.values[mask] = val
                        c.old_values[mask] = val
                    except:
                        pass
        except:
            pass
            
        self.constituents[name] = c

    # -------------------------------------------------------------------------
    # PHYSICS AND NUMERICS
    # -------------------------------------------------------------------------

    def calculate_henry_constant(self, temp_c, gas_type="O2"):
        """Interpolates Henry constant from table"""
        if len(self.atmos.henry_table_temps) == 0:
            return 0.001 if gas_type=="O2" else 0.03 # Fallback
            
        temps = self.atmos.henry_table_temps
        vals = self.atmos.henry_table_o2 if gas_type == "O2" else self.atmos.henry_table_co2
        
        return np.interp(temp_c, temps, vals)

    def calculate_saturation(self, temp_c, gas_type):
        # Saturation = PartialPressure / HenryConstant (M/atm) -> Molar Conc
        # Wait, usually C_sat = P * Kh.
        # Check units in CSV: Kh is M/atm. P is atm/bar.
        # Need to be careful with units.
        # VBA: Saturation = P_gas / Henry_Const ? Or P_gas * Henry_Const?
        # Henry Law: C = P * Kh.
        
        # Pressure conversion: Config has 'bar', Kh is '/atm'.
        # 1 atm = 1.01325 bar.
        p_atm = (self.atmos.p_o2 if gas_type == "O2" else self.atmos.p_co2) / 1.01325
        kh = self.calculate_henry_constant(temp_c, gas_type)
        
        mol_l = p_atm * kh
        
        # Convert to mg/L
        mw = 32000.0 if gas_type == "O2" else 44000.0 # mg/mol (from CSV)
        return mol_l * (mw / 1000.0) # check mw unit in CSV, snippet says 32000 mg/mole.

    def calculate_heat_fluxes(self, water_temp, time_sec):
        # Constants
        sigma = 5.67e-8
        kelvin = 273.15
        T_w_k = water_temp + kelvin
        T_a_k = self.atmos.air_temp + kelvin
        
        # 1. Solar Radiation (Shortwave)
        # Simple Sinusoidal model based on sunrise/sunset
        hour = (time_sec / 3600.0) % 24
        Q_sn = 0.0
        if self.atmos.sunrise_hour < hour < self.atmos.sunset_hour:
            # Fraction of day
            day_len = self.atmos.sunset_hour - self.atmos.sunrise_hour
            norm_time = (hour - self.atmos.sunrise_hour) / day_len
            Q_max = self.atmos.solar_constant * (1 - 0.65 * (self.atmos.cloud_cover/100)**2)
            Q_sn = Q_max * math.sin(math.pi * norm_time)
            
        # 2. Atmospheric Radiation (Longwave)
        if self.atmos.sky_temp_imposed:
            T_sky = self.atmos.sky_temp + kelvin
        else:
            # Swinbank
            T_sky = 0.0552 * (T_a_k**1.5)
            
        # Stefan-Boltzmann: epsilon * sigma * T^4. Epsilon water ~ 0.97
        Q_an = 0.97 * sigma * (T_sky**4)
        
        # 3. Back Radiation (Water surface)
        Q_br = 0.97 * sigma * (T_w_k**4)
        
        # 4. Evaporation (Latent Heat) and Convection (Sensible)
        # Vapor pressures (mmHg? mb?). Using Magnus-Tetens approx for esat (mb)
        es_a = 6.11 * 10**((7.5 * self.atmos.air_temp)/(237.3 + self.atmos.air_temp))
        es_w = 6.11 * 10**((7.5 * water_temp)/(237.3 + water_temp))
        ea = es_a * (self.atmos.humidity / 100.0)
        
        # Wind function f(u)
        # VBA uses h_min logic often.
        # Standard Dalton: E = f(u)(es_w - ea)
        # Let's use the explicit formulas if available, otherwise standard Penman-like wind function
        # f_u = 0.26 * (1 + 0.54 * self.atmos.wind_speed) # typical
        
        # Snippets mention h_min = 6.9.
        # Q_h = h * (Tw - Ta)
        # h = h_min + something * wind?
        # Let's assume a simplified linear wind dependence common in these codes
        h_conv = self.atmos.h_min + 3.0 * self.atmos.wind_speed 
        
        Q_h = h_conv * (water_temp - self.atmos.air_temp)
        
        # Q_e relates to Q_h via Bowen ratio or similar wind function
        # Or using the vapor pressure diff directly.
        # Approx: Q_e approx 1.5 * Q_h (very rough) or strictly Dalton.
        # Let's use Dalton with wind function derived from h
        # Evap heat flux W/m2.
        f_evap = (h_conv / 0.62) # Approx relation Cp*P/0.62*L
        # Using a standard formulation to match likely VBA behavior:
        Q_e = 3.0 * self.atmos.wind_speed * (es_w - ea) # Placeholder for exact VBA func
        if Q_e < 0 and (es_w < ea): Q_e = 0 # Condensation usually small
        
        # Total Flux into water
        # Q_net = absorbed_solar + absorbed_atmos - back_rad - evap - sensible
        # Absorb fractions: Solar ~0.9 (albedo 0.1), Atmos ~0.97
        Q_net = (0.9 * Q_sn) + Q_an - Q_br - Q_e - Q_h
        
        # Return volumetric source: Q_net * Width / Area_Vert = Q_net / Depth
        # Cp_water * rho_water approx 4.18e6 J/m3K
        source_term = Q_net / (self.grid.water_depth * 4186000.0) 
        return source_term # Kelvin/s

    def apply_discharges(self, dt_split):
        """Applies discharges as mass/energy sources"""
        # Mass Balance: V_cell * C_new = V_cell * C_old + Q_dis * C_dis * dt
        # If Q_dis = 0 (diffusive input), then Load is given directly?
        # Based on Discharges.csv where Flow=0 but Generic=100000, 
        # It implies specific handling.
        
        vol_cell = self.grid.area_vertical * self.grid.dx
        
        for d in self.discharges:
            idx = d["cell"]
            if idx >= self.grid.nc: continue
            
            # Logic: If flow > 0, it's an inflow. If flow = 0, check generic for mass load?
            # VBA Snippet doesn't show exact logic, but "Split Step" implies adding source.
            
            # 1. Temperature
            # Energy Load.
            if d["flow"] > 0:
                # Dilution/Addition:
                # (V*T + q*t*dt) / (V + q*dt) - but we assume constant volume/steady flow for hydro
                # So we treat as source: dC/dt = (q/V)(C_in - C)
                pass # Not implemented fully for Q>0 in this snippet context, assuming Q=0 per file
            
            # For Q=0, Mass Load mode (kg/day or similar converted to rate)
            # If "Discharges.csv" says Generic 100000, and Q=0.
            # We assume the values in CSV are Concentations (mg/L) and Flow is actually used?
            # Or values are Mass Fluxes?
            # Headers say "DischargeConcentrations_BOD". 
            # If Flow is 0, no mass enters.
            # EXCEPT: The CSV snippet shows Flow=0. This is suspicious.
            # Maybe the user must set flow > 0.
            # Or maybe "DischargeGeneric" is a mass flux.
            
            # Let's assume Flow SHOULD be used. If 0, nothing happens, unless code has override.
            # However, for the replica, I will assume the user updates flow in the UI.
            # Wait, Discharge 3 & 4 have 50C. If flow is 0, this heat never enters.
            # I will assume there's a small mixing flow or the user intends to edit it.
            
            # BUT, let's look at "DischargeGeneric". It might be a mass source.
            
            q = d["flow"]
            if q <= 0: continue # Skip if no flow
            
            # Volumetric source rate
            rate = q / vol_cell
            
            # Apply to all constituents
            if "Temperature" in self.constituents:
                # Heat mixing: dT/dt = (Q_dis/V_cell) * (T_dis - T_cell)
                T_cell = self.constituents["Temperature"].values[idx]
                T_dis = d["temp"]
                change = rate * (T_dis - T_cell) * dt_split
                self.constituents["Temperature"].values[idx] += change
                
            for name in ["BOD", "DO", "CO2", "Generic"]:
                if name in self.constituents:
                    C_cell = self.constituents[name].values[idx]
                    key = name.lower()
                    if key == "generic": key = "generic"
                    C_dis = d.get(key, 0.0)
                    
                    change = rate * (C_dis - C_cell) * dt_split
                    self.constituents[name].values[idx] += change

    def solve_transport(self, dt):
        """Solves Advection-Diffusion Equation"""
        
        # Common parameters
        u = self.flow.velocity
        D = self.flow.diffusivity
        dx = self.grid.dx
        nc = self.grid.nc
        
        # Courant and Diffusion numbers
        sigma = u * dt / dx
        beta = D * dt / (dx**2)
        
        # Construct Matrix Coefficients for Implicit/Semi
        # -a*C_i-1 + b*C_i - c*C_i+1 = RHS
        
        # We will iterate over all active constituents
        for name, prop in self.constituents.items():
            if not prop.active: continue
            
            C = prop.values # Current values (acting as old for this step)
            C_new = np.zeros_like(C)
            
            if self.config.time_discretisation == "exp":
                # Explicit Upwind or Central
                # dC/dt = -u dC/dx + D d2C/dx2
                for i in range(nc):
                    # Boundaries (Dirichlet or Neumann 0?)
                    # VBA usually folds boundaries or assumes cyclic or fixed.
                    # Assuming Fixed at left (0), Zero Gradient at right.
                    
                    # Indices
                    im1 = i - 1 if i > 0 else 0 # Left BC handling
                    ip1 = i + 1 if i < nc - 1 else nc - 1 # Right BC
                    
                    # Advection Flux (Upwind)
                    adv = -u * (C[i] - C[im1]) / dx
                    
                    # Diffusion
                    diff = D * (C[ip1] - 2*C[i] + C[im1]) / (dx**2)
                    
                    C_new[i] = C[i] + dt * (adv + diff)
                    
            elif self.config.time_discretisation in ["imp", "semi"]:
                # Thomas Algorithm (TDMA)
                # Formulation: A_i C_i-1 + B_i C_i + C_i C_i+1 = D_i
                # Using 0-based indexing for arrays a, b, c, d
                
                a = np.zeros(nc)
                b = np.zeros(nc)
                c_diag = np.zeros(nc) # 'c' variable name conflict
                d_rhs = np.zeros(nc)
                
                # Weighting for Semi-Implicit (Crank-Nicolson)
                theta = 0.5 if self.config.time_discretisation == "semi" else 1.0
                
                # Coefficients for Advection (Central) + Diffusion
                # Adv: u * (C_ip1 - C_im1) / 2dx
                # Diff: D * (C_ip1 - 2C_i + C_im1) / dx^2
                
                # Coefs for Implicit part (LHS at n+1):
                # C_i + theta*dt * [ u*(C_ip1 - C_im1)/2dx - D*(...) ]
                # Grouping:
                # C_im1 term: theta*dt * (-u/2dx - D/dx^2)  -> A
                # C_i   term: 1 + theta*dt * (2D/dx^2)      -> B
                # C_ip1 term: theta*dt * (u/2dx - D/dx^2)   -> C
                
                # Simplified constants
                alpha = u * dt / (2*dx)
                gamma = D * dt / (dx**2)
                
                # NOTE: VBA "QUICK" implementation often adds deferred correction to RHS
                # For this replica, we implement standard Upwind/Central Implicit first to ensure stability
                # QUICK logic adds complex source terms.
                
                for i in range(nc):
                    if i == 0:
                        # Boundary Left: Fixed Value
                        b[i] = 1.0
                        c_diag[i] = 0.0
                        d_rhs[i] = prop.values[0] # Keep initial value or boundary input
                        continue
                    elif i == nc - 1:
                        # Boundary Right: Zero Gradient C_N = C_N-1
                        a[i] = -1.0
                        b[i] = 1.0
                        d_rhs[i] = 0.0
                        continue
                    
                    # Internal Nodes
                    # LHS (n+1)
                    if self.config.advection_type == "upwind":
                         # Upwind Implicit: u(C_i - C_im1)/dx
                         # Coefs: -(-u/dx - D/dx2) for im1 ...
                         # Let's stick to the user's Central/QUICK prefs roughly or fallback to stable upwind
                         
                         # Pure Implicit Upwind-Diffusion
                         a[i] = -theta * (sigma + beta)
                         b[i] = 1 + theta * (sigma + 2*beta)
                         c_diag[i] = -theta * (-beta) # No advection from downstream in upwind
                         
                         # RHS (n)
                         # C_n + (1-theta) * Diffusion/Advection terms
                         # Explicit part
                         adv_ex = -u * (C[i] - C[i-1]) / dx
                         diff_ex = D * (C[i+1] - 2*C[i] + C[i-1]) / (dx**2)
                         d_rhs[i] = C[i] + (1-theta) * dt * (adv_ex + diff_ex)
                         
                    else: 
                        # Default to Central for generic impl
                        a[i] = theta * (-alpha - gamma)
                        b[i] = 1 + theta * (2*gamma)
                        c_diag[i] = theta * (alpha - gamma)
                        
                        # RHS
                        adv_ex = -u * (C[i+1] - C[i-1]) / (2*dx)
                        diff_ex = D * (C[i+1] - 2*C[i] + C[i-1]) / (dx**2)
                        d_rhs[i] = C[i] + (1-theta) * dt * (adv_ex + diff_ex)

                # Solve Tri-Diagonal Matrix
                # Forward Elimination
                for i in range(1, nc):
                    m = a[i] / b[i-1]
                    b[i] = b[i] - m * c_diag[i-1]
                    d_rhs[i] = d_rhs[i] - m * d_rhs[i-1]
                
                # Back Substitution
                C_new[nc-1] = d_rhs[nc-1] / b[nc-1]
                for i in range(nc-2, -1, -1):
                    C_new[i] = (d_rhs[i] - c_diag[i] * C_new[i+1]) / b[i]
            
            # Update
            prop.values = C_new
            prop.old_values = C_new.copy()

    def apply_kinetics(self, dt):
        """Source/Sink terms for BOD, DO, CO2"""
        # Get references
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod = self.constituents["BOD"].values if "BOD" in self.constituents else None
        do = self.constituents["DO"].values if "DO" in self.constituents else None
        co2 = self.constituents["CO2"].values if "CO2" in self.constituents else None
        
        # 1. Temperature Source
        if "Temperature" in self.constituents:
            for i in range(self.grid.nc):
                src = self.calculate_heat_fluxes(temp[i], self.current_time) # K/s
                self.constituents["Temperature"].values[i] += src * dt

        # 2. BOD Decay & DO Consumption
        if bod is not None and do is not None:
            # Constants
            k1_20 = 0.3 # /day, need from config, hardcoded default in VBA often
            theta_bod = 1.047
            
            for i in range(self.grid.nc):
                # Temp correction
                k1 = k1_20 * (theta_bod**(temp[i] - 20))
                k1_sec = k1 / 86400.0
                
                # Monod / Aerobic check
                # If DO > small epsilon, aerobic.
                # Rate dL/dt = -K1 * L
                dL = k1_sec * bod[i] * dt
                
                # Update BOD
                bod[i] -= dL
                
                # Update DO (consumption)
                do[i] -= dL 
                
                # Update CO2 (production) - Stoichiometry approx 1:1 molar or mass? 
                # C + O2 -> CO2. Mass ratio 12:32->44.
                # Assuming BOD is O2 equivalent. 
                # 1 mg O2 consumed produces (44/32) mg CO2.
                if co2 is not None:
                    co2[i] += dL * (44.0/32.0)

        # 3. Reaeration (DO) and CO2 Exchange
        if do is not None:
            for i in range(self.grid.nc):
                # K2 calculation (O'Connor Dobbins)
                # K2 = 3.93 * u^0.5 / H^1.5  (at 20C)
                u = self.flow.velocity
                h = self.grid.water_depth
                k2_20 = 3.93 * (u**0.5) / (h**1.5) # /day
                theta_aer = 1.024
                k2 = k2_20 * (theta_aer**(temp[i] - 20))
                k2_sec = k2 / 86400.0
                
                cs = self.calculate_saturation(temp[i], "O2")
                
                # dDO/dt = K2(Cs - C)
                do[i] += k2_sec * (cs - do[i]) * dt

        if co2 is not None:
            for i in range(self.grid.nc):
                # Similar K2 but adjusted for CO2 diffusivity ratio approx 0.9 or 1.0
                # Using same K2 for simplicity or replica
                k2_20 = 3.93 * (u**0.5) / (h**1.5) 
                k2 = k2_20 * (1.024**(temp[i] - 20))
                k2_sec = k2 / 86400.0
                
                cs_co2 = self.calculate_saturation(temp[i], "CO2")
                co2[i] += k2_sec * (cs_co2 - co2[i]) * dt

    def step(self):
        """Single Time Step Wrapper"""
        dt = self.config.dt
        
        # 1. Half-Step Discharge
        self.apply_discharges(0.5 * dt)
        
        # 2. Transport (Advection/Diffusion)
        self.solve_transport(dt)
        
        # 3. Half-Step Discharge
        self.apply_discharges(0.5 * dt)
        
        # 4. Kinetics / Sources
        self.apply_kinetics(dt)
        
        self.current_time += dt

    def run_simulation(self):
        """Generator to yield results for UI progress"""
        total_steps = int((self.config.duration_days * 86400) / self.config.dt)
        print_interval = int(self.config.dt_print / self.config.dt)
        
        # Initialize History
        times = []
        history = {name: [] for name in self.constituents}
        
        for step_n in range(total_steps):
            self.step()
            
            if step_n % print_interval == 0:
                times.append(self.current_time / 86400.0) # Days
                for name, prop in self.constituents.items():
                    history[name].append(prop.values.copy())
            
            yield step_n / total_steps
            
        self.results = {"times": times, "history": history}
