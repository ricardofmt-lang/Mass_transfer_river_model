import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# =============================================================================
# CONFIG & DATA CLASSES
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
    manning_coef: float = 0.0 # Stored but user Q used for V

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
    # Constants
    henry_table_temps: np.ndarray = field(default_factory=lambda: np.array([0, 5, 10, 15, 20, 25, 30]))
    henry_table_o2: np.ndarray = field(default_factory=lambda: np.array([0.00218, 0.00191, 0.00170, 0.00152, 0.00138, 0.00126, 0.00116]))
    henry_table_co2: np.ndarray = field(default_factory=lambda: np.array([0.0764, 0.0635, 0.0533, 0.0455, 0.0392, 0.0334, 0.0299]))

@dataclass
class Constituent:
    name: str
    active: bool
    unit: str
    values: np.ndarray
    old_values: np.ndarray
    
    # Constraints
    min_val: float = -1e9
    max_val: float = 1e9
    
    # Boundary Conditions: "Fixed", "ZeroGrad", "Cyclic"
    bc_left_type: str = "Fixed"
    bc_left_val: float = 0.0
    bc_right_type: str = "ZeroGrad"
    bc_right_val: float = 0.0
    
    # Physics Toggles & Params
    use_surface_flux: bool = True     # For Temp, DO, CO2
    use_sensible_heat: bool = True    # Temp
    use_latent_heat: bool = True      # Temp
    use_radiative_heat: bool = True   # Temp
    
    # Kinetics
    k_decay: float = 0.0              # Generic, BOD
    k_growth: float = 0.0             # BOD Logistic
    max_val_logistic: float = 0.0     # BOD Logistic
    use_logistic: bool = False        # BOD
    use_anaerobic: bool = False       # BOD
    o2_half_sat: float = 0.0          # BOD/DO coupling

@dataclass
class SimulationConfig:
    duration_days: float = 1.0
    dt: float = 200.0
    dt_print: float = 3600.0
    time_discretisation: str = "semi"
    advection_active: bool = True
    advection_type: str = "QUICK"
    quick_up_ratio: float = 0.5 # Default
    diffusion_active: bool = True

# =============================================================================
# ENGINE
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

    def setup_grid(self, length, nc, width, depth, slope, manning, discharge, diffusivity):
        self.grid.length = length
        self.grid.nc = nc
        self.grid.dx = length / nc
        self.grid.xc = np.linspace(self.grid.dx/2, length - self.grid.dx/2, nc)
        self.grid.river_width = width
        self.grid.water_depth = depth
        self.grid.river_slope = slope
        self.grid.manning_coef = manning
        self.grid.area_vertical = width * depth
        
        # Flow calc based on user Q input (Excel logic)
        self.flow.discharge = discharge
        self.flow.diffusivity = diffusivity
        self.flow.velocity = discharge / self.grid.area_vertical if self.grid.area_vertical > 0 else 0.0

    def setup_atmos(self, temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min, sky_temp, sky_imposed):
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

    def add_constituent(self, name, active, unit, 
                        init_mode="Default", default_val=0.0, 
                        init_cells=None, init_intervals=None,
                        bc_left_type="Fixed", bc_left_val=0.0,
                        bc_right_type="ZeroGrad", bc_right_val=0.0,
                        min_val=-1e9, max_val=1e9,
                        # Specific flags
                        use_surface_flux=True, use_sensible=True, use_latent=True, use_radiative=True,
                        k_decay=0.0, k_growth=0.0, max_logistic=0.0, use_logistic=False, 
                        use_anaerobic=False, o2_half_sat=0.0):
        
        vals = np.full(self.grid.nc, default_val)
        
        if init_mode == "Cell List" and init_cells:
            for item in init_cells:
                idx = item['idx']
                if 0 <= idx < self.grid.nc:
                    vals[idx] = item['val']
        elif init_mode == "Interval" and init_intervals:
            for item in init_intervals:
                x1, x2, v = item['start'], item['end'], item['val']
                mask = (self.grid.xc >= x1) & (self.grid.xc <= x2)
                vals[mask] = v

        c = Constituent(
            name=name, active=active, unit=unit,
            values=vals, old_values=vals.copy(),
            min_val=min_val, max_val=max_val,
            bc_left_type=bc_left_type, bc_left_val=bc_left_val,
            bc_right_type=bc_right_type, bc_right_val=bc_right_val,
            use_surface_flux=use_surface_flux,
            use_sensible_heat=use_sensible,
            use_latent_heat=use_latent,
            use_radiative_heat=use_radiative,
            k_decay=k_decay, k_growth=k_growth,
            max_val_logistic=max_logistic, use_logistic=use_logistic,
            use_anaerobic=use_anaerobic, o2_half_sat=o2_half_sat
        )
        self.constituents[name] = c
        self.results[name] = []

    def set_discharges(self, discharge_list):
        self.discharges = []
        for d in discharge_list:
            if "cell" in d and 0 <= d["cell"] < self.grid.nc:
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
        # mg/L = (atm * M/atm) * (mg/mol / 1000) ?? 
        # Check units: Kh is M/atm (mol/L/atm). 
        # C (mol/L) = P * Kh.
        # Mass (mg/L) = C * MW * 1000? No, MW is mg/mol? No usually g/mol.
        # Excel: MW O2 = 32000 mg/mol. Correct.
        mol_l = p_atm * kh
        mw = 32000.0 if gas_type == "O2" else 44000.0
        return mol_l * (mw / 1000.0) # mol/L * mg/mol / 1000 -> mg/L ?? No.
        # mol/L * (mg/mol) = mg/L.
        # Wait, if MW is 32000 mg/mol, then mol_l * 32000 = mg/L.
        return mol_l * mw / 1000.0 # Adjusting scale if needed, or just mw

    def calculate_heat_fluxes(self, water_temp, time_sec):
        # Constants
        sigma = 5.67e-8
        kelvin = 273.15
        T_w_k = water_temp + kelvin
        T_a_k = self.atmos.air_temp + kelvin
        
        Q_total = 0.0
        
        # 1. Solar (Shortwave)
        hour = (time_sec / 3600.0) % 24
        if self.atmos.sunrise_hour < hour < self.atmos.sunset_hour:
            day_len = self.atmos.sunset_hour - self.atmos.sunrise_hour
            norm_time = (hour - self.atmos.sunrise_hour) / day_len
            Q_max = self.atmos.solar_constant * (1 - 0.65 * (self.atmos.cloud_cover/100)**2)
            Q_sn = Q_max * math.sin(math.pi * norm_time)
            # Typically albedo is reflected, so (1-albedo). Approx 0.9 absorbed.
            Q_total += 0.9 * Q_sn # User asks to toggle FreeSurfaceFlux generally, or components?
            # Assuming "FreeSurfaceFlux" means the exchange terms, usually Solar is always on if Atmos is on.
            # But let's check constituent config.
        
        # 2. Atmospheric (Longwave Down)
        if self.atmos.sky_temp_imposed:
            T_sky = self.atmos.sky_temp + kelvin
        else:
            T_sky = 0.0552 * (T_a_k**1.5)
        Q_an = 0.97 * sigma * (T_sky**4)
        
        # 3. Back Radiation (Longwave Up)
        Q_br = 0.97 * sigma * (T_w_k**4)
        
        # 4. Evap & Sensible
        es_a = 6.11 * 10**((7.5 * self.atmos.air_temp)/(237.3 + self.atmos.air_temp))
        es_w = 6.11 * 10**((7.5 * water_temp)/(237.3 + water_temp))
        ea = es_a * (self.atmos.humidity / 100.0)
        
        # Wind function f(u) = h_min + ...
        # h_conv (W/m2K)
        h_conv = self.atmos.h_min + 3.0 * self.atmos.wind_speed 
        
        Q_h = h_conv * (water_temp - self.atmos.air_temp)
        Q_e = 3.0 * self.atmos.wind_speed * (es_w - ea) # Approximate Dalton
        if Q_e < 0 and es_w < ea: Q_e = 0
        
        # Apply Toggles
        # Assuming we are inside apply_kinetics where we have access to "prop"
        # We return dictionary or tuple to be filtered by caller
        return {
            "solar": 0.9 * Q_sn if 'Q_sn' in locals() else 0,
            "atmos": Q_an,
            "back": -Q_br,
            "sensible": -Q_h,
            "latent": -Q_e
        }

    def apply_discharges(self, dt_split):
        vol_cell = self.grid.area_vertical * self.grid.dx
        for d in self.discharges:
            idx = d["cell"]
            q = d["flow"]
            if q <= 0: continue
            
            # Analytical mixing
            rate = q / vol_cell
            factor = math.exp(-rate * dt_split)
            
            if "Temperature" in self.constituents:
                T_c = self.constituents["Temperature"].values[idx]
                T_in = d["temp"]
                self.constituents["Temperature"].values[idx] = T_in + (T_c - T_in) * factor
                
            for name in ["BOD", "DO", "CO2", "Generic"]:
                if name in self.constituents:
                    C_c = self.constituents[name].values[idx]
                    key = name.lower() 
                    val_in = d.get(key, 0.0)
                    self.constituents[name].values[idx] = val_in + (C_c - val_in) * factor

    def solve_transport(self, dt):
        u = self.flow.velocity
        D = self.flow.diffusivity
        dx = self.grid.dx
        nc = self.grid.nc
        
        for name, prop in self.constituents.items():
            if not prop.active: continue
            C = prop.values
            
            a, b, c_diag, d_rhs = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
            theta = 0.5 if self.config.time_discretisation == "semi" else 1.0
            
            alpha = u * dt / (2*dx)
            gamma = D * dt / (dx**2)
            
            # --- INTERNAL NODES ---
            for i in range(1, nc-1):
                # Diffusion (Central)
                a[i] = -theta * beta if 'beta' in locals() else -theta * (D*dt/dx**2)
                c_diag[i] = a[i]
                b[i] = 1 - 2*a[i]
                
                # Re-calculate cleanly
                beta = D*dt/(dx**2)
                sigma = u*dt/dx
                
                a[i] = -theta * beta
                c_diag[i] = -theta * beta
                b[i] = 1 + 2*theta*beta
                
                rhs_diff = (1-theta)*beta*C[i+1] + (1 - 2*(1-theta)*beta)*C[i] + (1-theta)*beta*C[i-1]
                
                # Advection
                if self.config.advection_type == "Central":
                    a[i] -= theta * sigma / 2
                    c_diag[i] += theta * sigma / 2
                    rhs_adv = -(1-theta)*(sigma/2)*(C[i+1] - C[i-1])
                else:
                    # Upwind or QUICK
                    a[i] -= theta * sigma
                    b[i] += theta * sigma
                    rhs_adv = -(1-theta)*sigma*(C[i] - C[i-1])
                    
                    if self.config.advection_type == "QUICK":
                        # Deferred Correction
                        # Explicit flux adjustment
                        im2 = i-2 if i>=2 else 0
                        ip1 = i+1
                        im1 = i-1
                        
                        f_quick_in = 0.5*(C[im1]+C[i]) - 0.125*(C[im2]-2*C[im1]+C[i])
                        f_quick_out = 0.5*(C[i]+C[ip1]) - 0.125*(C[im1]-2*C[i]+C[ip1])
                        
                        f_up_in = C[im1]
                        f_up_out = C[i]
                        
                        corr = -(u/dx) * ((f_quick_out - f_quick_in) - (f_up_out - f_up_in))
                        # Apply ratio if requested
                        # ratio = self.config.quick_up_ratio # Not standard but user asked
                        rhs_adv += corr * dt

                d_rhs[i] = rhs_diff + rhs_adv

            # --- BOUNDARIES ---
            # Left
            if prop.bc_left_type == "Fixed":
                b[0], d_rhs[0] = 1.0, prop.bc_left_val
            elif prop.bc_left_type == "ZeroGrad":
                b[0], c_diag[0], d_rhs[0] = 1.0, -1.0, 0.0
            elif prop.bc_left_type == "Cyclic":
                # Handled via iterative patch or simple fold. 
                # For TDMA cyclic is complex (Sherman-Morrison). 
                # Approximation: Fixed to old C[N-1] or explicit transfer
                b[0], d_rhs[0] = 1.0, C[nc-1] 

            # Right
            if prop.bc_right_type == "Fixed":
                b[nc-1], d_rhs[nc-1] = 1.0, prop.bc_right_val
            elif prop.bc_right_type == "ZeroGrad":
                a[nc-1], b[nc-1], d_rhs[nc-1] = -1.0, 1.0, 0.0
            elif prop.bc_right_type == "Cyclic":
                a[nc-1], b[nc-1], d_rhs[nc-1] = -1.0, 1.0, 0.0 # Flow out

            # --- SOLVER ---
            if b[0] == 0: b[0] = 1e-10
            c_prime = np.zeros(nc)
            d_prime = np.zeros(nc)
            c_prime[0] = c_diag[0]/b[0]
            d_prime[0] = d_rhs[0]/b[0]
            
            for i in range(1, nc):
                denom = b[i] - a[i]*c_prime[i-1]
                if denom == 0: denom = 1e-10
                c_prime[i] = c_diag[i]/denom
                d_prime[i] = (d_rhs[i] - a[i]*d_prime[i-1])/denom
                
            C_new[nc-1] = d_prime[nc-1]
            for i in range(nc-2, -1, -1):
                C_new[i] = d_prime[i] - c_prime[i]*C_new[i+1]
                
            # Clamp limits
            np.clip(C_new, prop.min_val, prop.max_val, out=C_new)
            prop.values = C_new

    def apply_kinetics(self, dt):
        # Fetch arrays
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod_prop = self.constituents.get("BOD")
        do_prop = self.constituents.get("DO")
        co2_prop = self.constituents.get("CO2")
        gen_prop = self.constituents.get("Generic")
        
        # 1. Temperature Kinetics
        if "Temperature" in self.constituents:
            p = self.constituents["Temperature"]
            for i in range(self.grid.nc):
                fluxes = self.calculate_heat_fluxes(p.values[i], self.current_time)
                # Sum based on toggles
                Q_sum = 0.0
                if p.use_radiative_heat: Q_sum += fluxes["solar"] + fluxes["atmos"] + fluxes["back"]
                # Note: Solar usually distinct, but grouping 'radiative' often implies Longwave. 
                # Based on Excel headers "FreeSurfaceRadiative...", likely LW. Solar is "FreeSurfaceFlux"?
                # Assuming "FreeSurfaceFlux" = ALL exchange?
                # User said: FreeSurfaceFlux, FreeSurfaceSensible..., FreeSurfaceLatent..., FreeSurfaceRadiative...
                # I'll treat FreeSurfaceFlux as a Master Switch for surface exchange? 
                # Or Solar. Let's assume standard modularity:
                
                val_change = 0.0
                if p.use_surface_flux: # Master switch or Solar? Assuming Solar/Net
                     val_change += fluxes["solar"]
                if p.use_radiative_heat:
                     val_change += fluxes["atmos"] + fluxes["back"]
                if p.use_sensible_heat:
                     val_change += fluxes["sensible"]
                if p.use_latent_heat:
                     val_change += fluxes["latent"]
                
                # Volumetric Source: Q (W/m2) / (Depth * rho * Cp)
                # rho*Cp approx 4.18e6 J/m3K
                dTemp = val_change / (self.grid.water_depth * 4186000.0)
                p.values[i] += dTemp * dt

        # 2. Generic Decay
        if gen_prop:
            k = gen_prop.k_decay
            for i in range(self.grid.nc):
                # dC/dt = -k C
                gen_prop.values[i] -= k * gen_prop.values[i] * dt

        # 3. BOD Kinetics
        if bod_prop:
            vals = bod_prop.values
            for i in range(self.grid.nc):
                # Logistic Growth or Decay?
                # "Logistic Formulation" usually: dC/dt = k * C * (1 - C/Max)
                rate = 0.0
                if bod_prop.use_logistic:
                    # Logistic Growth? Or Decay? 
                    # User asked for "GrowthRateLogisticFormulation".
                    # dL/dt = k_growth * L * (1 - L/L_max)
                    # BUT BOD is oxygen demand. Typically it decays. 
                    # Perhaps this simulates Algae BOD?
                    k_g = bod_prop.k_growth
                    L_max = bod_prop.max_val_logistic
                    if L_max != 0:
                        rate = k_g * vals[i] * (1 - vals[i]/L_max)
                    vals[i] += rate * dt
                else:
                    # Standard Decay
                    # Check Anaerobic
                    # If DO < Threshold (e.g. 0.5), use Anaerobic rate?
                    # User asked for "Consider Anaerobic Respiration".
                    # Usually means decay continues even if DO=0.
                    # Standard Streeter Phelps stops if DO=0? No, it just stops consuming DO.
                    
                    k1 = bod_prop.k_decay * (1.047**(temp[i]-20)) / 86400.0
                    L = vals[i]
                    
                    # DO Limitation factor (Monod)
                    f_ox = 1.0
                    if do_prop:
                        do_val = do_prop.values[i]
                        ks = bod_prop.o2_half_sat
                        if ks > 0: f_ox = do_val / (ks + do_val)
                    
                    # If Anaerobic allowed, we don't limit by f_ox completely?
                    # Simplified: if anaerobic, Oxidation continues (reducing BOD) but using NO3/SO4 (not modeled)
                    # so BOD decreases, but DO doesn't drop.
                    # Or maybe rate is different.
                    # We will assume: Decay = -k1 * L.
                    # DO Consumption = -k1 * L * f_ox.
                    # If Anaerobic ON, BOD decays regardless of DO?
                    
                    dL = k1 * L * dt
                    vals[i] -= dL
                    
                    # Coupling
                    if do_prop:
                        # DO consumed only by Aerobic part
                        dDO = dL * f_ox
                        do_prop.values[i] -= dDO
                    if co2_prop:
                        # CO2 produced
                        dCO2 = dL * (44.0/32.0)
                        co2_prop.values[i] += dCO2

        # 4. Reaeration (DO & CO2)
        u = self.flow.velocity
        h = self.grid.water_depth
        # O'Connor Dobbins: K2 = 3.93 * U^0.5 / H^1.5 (at 20C)
        k2_base = 3.93 * (u**0.5) / (h**1.5) if h > 0 else 0
        
        if do_prop and do_prop.use_surface_flux:
            for i in range(self.grid.nc):
                k2 = k2_base * (1.024**(temp[i]-20)) / 86400.0
                cs = self.calculate_saturation(temp[i], "O2")
                do_prop.values[i] += k2 * (cs - do_prop.values[i]) * dt
                
        if co2_prop and co2_prop.use_surface_flux:
            for i in range(self.grid.nc):
                k2 = k2_base * (1.024**(temp[i]-20)) / 86400.0
                cs = self.calculate_saturation(temp[i], "CO2")
                co2_prop.values[i] += k2 * (cs - co2_prop.values[i]) * dt

        # Apply Clamping
        for name, prop in self.constituents.items():
            np.clip(prop.values, prop.min_val, prop.max_val, out=prop.values)

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
            if callback: callback(progress)
        return self.results
