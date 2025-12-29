import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    
    min_val: float = -1e9
    max_val: float = 1e9
    
    bc_left_type: str = "Fixed"
    bc_left_val: float = 0.0
    bc_right_type: str = "ZeroGrad"
    bc_right_val: float = 0.0
    
    use_surface_flux: bool = True     
    use_sensible_heat: bool = True    
    use_latent_heat: bool = True      
    use_radiative_heat: bool = True   
    
    k_decay: float = 0.0              
    k_growth: float = 0.0             
    max_val_logistic: float = 0.0     
    use_logistic: bool = False        
    use_anaerobic: bool = False       
    o2_half_sat: float = 0.0          

@dataclass
class SimulationConfig:
    duration_days: float = 1.0
    dt: float = 200.0
    dt_print: float = 3600.0
    time_discretisation: str = "semi"
    advection_active: bool = True
    advection_type: str = "QUICK"
    quick_up_ratio: float = 4.0
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
        
        self.flow.discharge = discharge
        self.flow.diffusivity = diffusivity
        # Standard velocity calculation for rectangular channel
        if self.grid.area_vertical > 0:
            self.flow.velocity = discharge / self.grid.area_vertical
        else:
            self.flow.velocity = 0.0

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
                        use_surface_flux=True, use_sensible=True, use_latent=True, use_radiative=True,
                        k_decay=0.0, k_growth=0.0, max_logistic=0.0, use_logistic=False, 
                        use_anaerobic=False, o2_half_sat=0.0):
        
        vals = np.full(self.grid.nc, default_val)
        
        if init_mode == "Cell" and init_cells:
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

    def calculate_henry_constant(self, temp_c, gas_type="O2"):
        vals = self.atmos.henry_table_o2 if gas_type == "O2" else self.atmos.henry_table_co2
        return np.interp(temp_c, self.atmos.henry_table_temps, vals)

    def calculate_saturation(self, temp_c, gas_type):
        p_atm = (self.atmos.p_o2 if gas_type == "O2" else self.atmos.p_co2) / 1.01325
        kh = self.calculate_henry_constant(temp_c, gas_type)
        mw = 32000.0 if gas_type == "O2" else 44000.0
        return (p_atm * kh) * mw / 1000.0

    def calculate_solar_radiation(self, time_sec):
        day_of_year = int((time_sec / 86400.0) % 365) + 1
        hour = (time_sec / 3600.0) % 24
        
        phi = math.radians(self.atmos.latitude)
        delta = 0.409 * math.sin(2 * math.pi / 365.0 * (284 + day_of_year))
        omega = (math.pi / 12.0) * (hour - 12.0)
        
        sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
        
        if sin_alpha <= 0: return 0.0
        
        I_0 = self.atmos.solar_constant
        Q_s = I_0 * sin_alpha * (1.0 - 0.65 * (self.atmos.cloud_cover/100.0)**2)
        return max(0.0, Q_s)

    def calculate_heat_fluxes(self, water_temp, time_sec):
        sigma = 5.67e-8
        kelvin = 273.15
        T_w_k = water_temp + kelvin
        T_a_k = self.atmos.air_temp + kelvin
        
        Q_sn = self.calculate_solar_radiation(time_sec)
        Q_solar_absorbed = 0.9 * Q_sn
        
        if self.atmos.sky_temp_imposed:
            T_sky = self.atmos.sky_temp + kelvin
        else:
            T_sky = 0.0552 * (T_a_k**1.5)
        Q_an = 0.97 * sigma * (T_sky**4)
        
        Q_br = 0.97 * sigma * (T_w_k**4)
        
        es_a = 6.11 * 10**((7.5 * self.atmos.air_temp)/(237.3 + self.atmos.air_temp))
        es_w = 6.11 * 10**((7.5 * water_temp)/(237.3 + water_temp))
        ea = es_a * (self.atmos.humidity / 100.0)
        
        h_wind = self.atmos.h_min + 3.0 * self.atmos.wind_speed
        Q_h = 0.62 * h_wind * (water_temp - self.atmos.air_temp)
        
        Q_e = 0.62 * h_wind * (es_w - ea)
        if Q_e < 0 and es_w < ea: Q_e = 0
        
        return {"solar": 0.9 * Q_sn, "atmos": Q_an, "back": -Q_br, "sensible": -Q_h, "latent": -Q_e}

    def apply_discharges(self, dt_split):
        vol_cell = self.grid.area_vertical * self.grid.dx
        for d in self.discharges:
            idx = d["cell"]
            q = d["flow"]
            if q <= 0: continue
            
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
        u = self.flow.velocity if self.config.advection_active else 0.0
        D = self.flow.diffusivity if self.config.diffusion_active else 0.0
        dx = self.grid.dx
        nc = self.grid.nc
        
        sigma = u * dt / dx
        beta = D * dt / (dx**2)
        
        n_iter = 2 if self.config.advection_type == "QUICK" else 1
        
        for name, prop in self.constituents.items():
            if not prop.active: continue
            
            is_cyclic = (prop.bc_left_type == "Cyclic")
            
            C_old = prop.values.copy()
            C_iter = C_old.copy() 
            
            for _ in range(n_iter):
                a = np.zeros(nc)
                b = np.zeros(nc)
                c_diag = np.zeros(nc)
                d_rhs = np.zeros(nc)
                
                theta = 0.5 if self.config.time_discretisation == "semi" else 1.0
                
                # --- INTERNAL NODES ---
                for i in range(1, nc-1):
                    # Diffusion Implicit
                    a[i] = -theta * beta
                    b[i] = 1.0 + 2.0 * theta * beta
                    c_diag[i] = -theta * beta
                    
                    # Diffusion Explicit
                    diff_exp = (1.0-theta)*beta*(C_old[i+1] - 2.0*C_old[i] + C_old[i-1])
                    d_rhs[i] = C_old[i] + diff_exp
                    
                    # Advection
                    if self.config.advection_active:
                        adv_scheme = self.config.advection_type
                        
                        use_upwind = False
                        if adv_scheme == "QUICK":
                            grad1 = C_iter[i] - C_iter[i-1]
                            grad2 = C_iter[i+1] - C_iter[i]
                            if grad1 * grad2 < 0: use_upwind = True
                        
                        if adv_scheme == "Central":
                            a[i] -= theta * sigma / 2.0
                            c_diag[i] += theta * sigma / 2.0
                            d_rhs[i] -= (1.0-theta)*(sigma/2.0)*(C_old[i+1] - C_old[i-1])
                        elif adv_scheme == "Upwind" or use_upwind:
                            a[i] -= theta * sigma
                            b[i] += theta * sigma
                            d_rhs[i] -= (1.0-theta)*sigma*(C_old[i] - C_old[i-1])
                        elif adv_scheme == "QUICK":
                            a[i] -= theta * sigma
                            b[i] += theta * sigma
                            d_rhs[i] -= (1.0-theta)*sigma*(C_old[i] - C_old[i-1])
                            
                            im2 = i-2 if i>=2 else 0
                            fq_out = 0.5*(C_iter[i]+C_iter[i+1]) - 0.125*(C_iter[i-1]-2*C_iter[i]+C_iter[i+1])
                            fq_in = 0.5*(C_iter[i-1]+C_iter[i]) - 0.125*(C_iter[im2]-2*C_iter[i-1]+C_iter[i])
                            fu_out, fu_in = C_iter[i], C_iter[i-1]
                            
                            corr = -(u/dx) * ((fq_out - fq_in) - (fu_out - fu_in))
                            d_rhs[i] += corr * dt * self.config.quick_up_ratio

                # --- BOUNDARY CONDITIONS (Flux-Based Finite Volume) ---
                
                # LEFT (i=0)
                # Flux In (West)
                if is_cyclic and u >= 0:
                    b[0] = 1.0; d_rhs[0] = C_iter[nc-1] # Simple lag for cyclic
                else:
                    # Balance: dC/dt = (Fin - Fout) / dx
                    # C_new - dt/dx * (Fin - Fout) = C_old
                    # Fin is at face 0.
                    # Fixed Val Cb: Fin = u*Cb + (2D/dx)*(Cb - C0)
                    # ZeroGrad: Fin = u*C0
                    
                    term_diff_bound = 2.0 * D / dx if self.config.diffusion_active else 0.0
                    
                    # Implicit contribution to b[0] and d_rhs[0]
                    # We start with basic identity C0_new = C0_old, then add fluxes
                    b[0] = 1.0
                    d_rhs[0] = C_old[0]
                    
                    # Add Flux terms scaled by dt/dx
                    # Outgoing flux to cell 1 (internal) is handled by solving whole system? 
                    # No, for i=0 we must define the connection to 1 explicitely or implicitely.
                    # We use standard coefficients for right side (c_diag)
                    
                    # Let's use the standard discretized equation for node 0, but modified for boundary flux
                    # Standard: C0_new + theta*(-FluxNet_imp) = C0_old + (1-theta)*(-FluxNet_exp)
                    # FluxNet = F_east - F_west
                    
                    # F_east (to node 1):
                    # Diff: -D(C1 - C0)/dx. Adv: u*C0 (upwind) or u*(C0+C1)/2 (central)
                    # Use Upwind/Central mix as per config
                    
                    # F_west (Boundary):
                    # Fixed: u*Cb - 2D/dx(C0 - Cb)
                    
                    # Construct coefs for 0
                    # c_diag[0] (affects C1), b[0] (affects C0)
                    
                    # 1. East Face (Standard)
                    if self.config.diffusion_active:
                        # Diff flux out: -D(C1-C0)/dx
                        # Term in eq: - (-D(C1-C0)/dx) / dx = D/dx^2 * (C1 - C0)
                        # = beta/dt * (C1 - C0)
                        # Implicit part: -theta*beta*C1 + theta*beta*C0
                        b[0] += theta * beta
                        c_diag[0] += -theta * beta
                        d_rhs[0] += (1-theta)*beta*(C_old[1] - C_old[0])
                    
                    if self.config.advection_active:
                        # Adv flux out: u*C0 (Upwind)
                        # Term: + u/dx * C0 = sigma/dt * C0
                        # Implicit: theta*sigma * C0
                        b[0] += theta * sigma
                        d_rhs[0] -= (1-theta)*sigma*C_old[0]
                        
                    # 2. West Face (Boundary)
                    # Flux In. Term in eq: + F_west / dx
                    # Fixed:
                    if prop.bc_left_type == "Fixed":
                        val = prop.bc_left_val
                        # Adv In: u*val. Term: + u/dx * val = sigma/dt * val
                        if self.config.advection_active:
                            d_rhs[0] += theta*sigma*val + (1-theta)*sigma*val # All explicit source
                        
                        # Diff In: -2D/dx(C0 - val) = 2D/dx * val - 2D/dx * C0
                        # Term: + (2D/dx^2) * val - (2D/dx^2) * C0
                        # = 2*beta/dt * val - 2*beta/dt * C0
                        if self.config.diffusion_active:
                            d_rhs[0] += 2*beta*val # Explicit source part
                            b[0] += theta * 2 * beta # Implicit C0 part (moved to LHS)
                            d_rhs[0] -= (1-theta)*2*beta*C_old[0] # Explicit C0 part
                            
                    else: # ZeroGrad
                        # Flux In = u*C0. (Diff=0)
                        # Term: + u/dx * C0 = sigma/dt * C0
                        if self.config.advection_active:
                            b[0] -= theta * sigma # Moves to LHS? 
                            # Wait, Flux In adds to volume. + (theta*sigma*C0_new). 
                            # Moved to LHS (b) becomes -theta*sigma
                            d_rhs[0] += (1-theta)*sigma*C_old[0]

                # RIGHT (i=N-1)
                if apply_cyclic:
                    b[nc-1] = 1.0; d_rhs[nc-1] = C_iter[0]
                else:
                    # Balance Cell N-1
                    b[nc-1] = 1.0
                    d_rhs[nc-1] = C_old[nc-1]
                    
                    # 1. West Face (from N-2)
                    # Standard formulation
                    if self.config.diffusion_active:
                        b[nc-1] += theta * beta
                        a[nc-1] += -theta * beta
                        d_rhs[nc-1] += (1-theta)*beta*(C_old[nc-2] - C_old[nc-1])
                        
                    if self.config.advection_active:
                        # Upwind In: u*C_N-2
                        # Term: + u/dx * C_N-2
                        # Implicit: -theta*sigma*C_N-2 (LHS a)
                        a[nc-1] -= theta * sigma
                        d_rhs[nc-1] += (1-theta)*sigma*C_old[nc-2]
                        
                    # 2. East Face (Boundary)
                    # Flux Out. Term: - F_east / dx
                    if prop.bc_right_type == "Fixed":
                        val = prop.bc_right_val
                        # Adv Out: u*C_N-1 (Assuming upwind outflow even if fixed val? No, if fixed, concentration is fixed at face)
                        # Usually outflow is ZeroGrad. If Fixed, implies forced condition.
                        # Diff Out: -2D/dx (val - C_N-1)
                        if self.config.diffusion_active:
                            # Term: - [ -2D/dx^2 (val - C) ] = 2beta/dt * (val - C)
                            d_rhs[nc-1] += 2*beta*val
                            b[nc-1] += theta * 2 * beta
                            d_rhs[nc-1] -= (1-theta)*2*beta*C_old[nc-1]
                            
                        if self.config.advection_active:
                            # Out: u * val ?? Or u * C_N-1?
                            # Standard transport equation outflow is u*C_N-1.
                            b[nc-1] += theta * sigma
                            d_rhs[nc-1] -= (1-theta)*sigma*C_old[nc-1]
                            
                    else: # ZeroGrad
                        # Flux Out: u*C_N-1
                        if self.config.advection_active:
                            b[nc-1] += theta * sigma
                            d_rhs[nc-1] -= (1-theta)*sigma*C_old[nc-1]

                # --- SOLVE TDMA ---
                if b[0] == 0: b[0] = 1e-15
                cp = np.zeros(nc)
                dp = np.zeros(nc)
                
                cp[0] = c_diag[0] / b[0]
                dp[0] = d_rhs[0] / b[0]
                
                for i in range(1, nc):
                    denom = b[i] - a[i]*cp[i-1]
                    if denom == 0: denom = 1e-15
                    cp[i] = c_diag[i] / denom
                    dp[i] = (d_rhs[i] - a[i]*dp[i-1]) / denom
                
                C_new = np.zeros(nc)
                C_new[nc-1] = dp[nc-1]
                for i in range(nc-2, -1, -1):
                    C_new[i] = dp[i] - cp[i]*C_new[i+1]
                
                C_iter = C_new.copy()
                
            np.clip(C_new, prop.min_val, prop.max_val, out=C_new)
            prop.values = C_new

    def apply_kinetics(self, dt):
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod_prop = self.constituents.get("BOD")
        do_prop = self.constituents.get("DO")
        co2_prop = self.constituents.get("CO2")
        gen_prop = self.constituents.get("Generic")
        
        # 1. Temperature
        if "Temperature" in self.constituents:
            p = self.constituents["Temperature"]
            for i in range(self.grid.nc):
                fluxes = self.calculate_heat_fluxes(p.values[i], self.current_time)
                val_change = 0.0
                if p.use_surface_flux: val_change += fluxes["solar"]
                if p.use_radiative_heat: val_change += fluxes["atmos"] + fluxes["back"]
                if p.use_sensible_heat: val_change += fluxes["sensible"]
                if p.use_latent_heat: val_change += fluxes["latent"]
                
                dTemp = val_change / (self.grid.water_depth * 4186000.0)
                p.values[i] += dTemp * dt

        # 2. Generic
        if gen_prop:
            k = gen_prop.k_decay
            for i in range(self.grid.nc):
                gen_prop.values[i] -= k * gen_prop.values[i] * dt

        # 3. BOD
        if bod_prop:
            vals = bod_prop.values
            for i in range(self.grid.nc):
                rate = 0.0
                if bod_prop.use_logistic:
                    k_g = bod_prop.k_growth
                    L_max = bod_prop.max_val_logistic
                    if L_max != 0:
                        rate = k_g * vals[i] * (1 - vals[i]/L_max)
                    vals[i] += rate * dt
                else:
                    k1 = bod_prop.k_decay * (1.047**(temp[i]-20)) / 86400.0
                    L = vals[i]
                    f_ox = 1.0
                    if do_prop:
                        do_val = do_prop.values[i]
                        ks = bod_prop.o2_half_sat
                        if ks > 0: f_ox = do_val / (ks + do_val)
                    
                    decay_rate = k1 * L
                    if not bod_prop.use_anaerobic:
                        decay_rate *= f_ox
                    
                    vals[i] -= decay_rate * dt
                    
                    if do_prop:
                        dDO = decay_rate * f_ox
                        do_prop.values[i] -= dDO
                    if co2_prop:
                        dCO2 = decay_rate * (44.0/32.0)
                        co2_prop.values[i] += dCO2

        # 4. Reaeration with Wind/Slope Formula
        u = self.flow.velocity
        h = self.grid.water_depth
        slope = self.grid.river_slope
        wind = self.atmos.wind_speed
        
        k_reaeration = (1.0/h) * 0.142 * (abs(wind) + 0.1) * (slope + 1e-5)
        
        for p in [do_prop, co2_prop]:
            if p and p.use_surface_flux:
                gas = "O2" if p.name=="DO" else "CO2"
                for i in range(self.grid.nc):
                    cs = self.calculate_saturation(temp[i], gas)
                    p.values[i] = (p.values[i] + dt * k_reaeration * cs) / (1.0 + dt * k_reaeration)

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
