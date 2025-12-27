import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# --- Data Structures (Mirroring VBA Types) ---

@dataclass
class Grid:
    length: float
    river_width: float
    water_depth: float
    nc: int
    # Computed
    dx: float = 0.0
    area_vertical: float = 0.0
    area_h: float = 0.0
    xc: np.ndarray = field(default_factory=lambda: np.array([]))

    def compute_geometry(self):
        self.dx = self.length / self.nc
        self.area_vertical = self.water_depth * self.river_width
        self.area_h = self.river_width * self.dx
        self.xc = np.linspace(self.dx/2, self.length - self.dx/2, self.nc)

@dataclass
class FlowProperties:
    velocity: float
    diffusivity: float
    river_slope: float
    # Computed
    diffusion_nr: float = 0.0
    courant_nr: float = 0.0

@dataclass
class GasProperties:
    label: str
    partial_pressure: float
    molecular_weight: float = 0.0
    # Arrays for Henry's Law Interpolation
    henry_temps: List[float] = field(default_factory=list)
    henry_ks: List[float] = field(default_factory=list) 

@dataclass
class Atmosphere:
    temperature: float
    wind_speed: float
    humidity: float
    solar_constant: float
    latitude: float
    sky_temperature: float
    sky_temp_imposed: bool
    cloud_cover: float
    tsr: float # Sunrise
    tss: float # Sunset
    h_min: float = 6.9

@dataclass
class Discharge:
    name: str
    cell: int # 0-based index
    volume_rate: float
    specific_value: float

@dataclass
class BoundaryCondition:
    left_value: float = 0.0
    right_value: float = 0.0
    cyclic: bool = False
    free_surface_flux: bool = False
    fs_sensible_heat: bool = False
    fs_latent_heat: bool = False
    fs_radiative_heat: bool = False
    # For gases
    gas_exchange_params: Optional[GasProperties] = None
    discharges: List[Discharge] = field(default_factory=list)

@dataclass
class InitConfig:
    type: str = "DEFAULT" # DEFAULT, CELL, INTERVAL_M, INTERVAL_C
    default_val: float = 0.0
    # List of tuples for specific inits: (start, end, value)
    # For CELL: (cell_idx, cell_idx, value)
    # For INTERVAL_M: (start_m, end_m, value)
    intervals: List[tuple] = field(default_factory=list)

@dataclass
class Property:
    name: str
    active: bool = False
    units: str = ""
    min_val: float = 0.0
    max_val: float = 1e9
    min_active: bool = False
    max_active: bool = False
    
    # Initialization configuration
    init_config: InitConfig = field(default_factory=InitConfig)
    
    # Kinetic params
    decay_growth: bool = False
    decay_rate: float = 0.0
    growth_rate: float = 0.0
    max_val_logistic: float = 0.0
    grazing_ksat: float = 0.0
    anaerobic_respiration: bool = False
    
    # State Arrays
    current_val: np.ndarray = field(default_factory=lambda: np.array([]))
    old_val: np.ndarray = field(default_factory=lambda: np.array([]))
    integrated_aero: np.ndarray = field(default_factory=lambda: np.array([]))
    integrated_anaero: np.ndarray = field(default_factory=lambda: np.array([]))
    
    boundary: BoundaryCondition = field(default_factory=BoundaryCondition)

@dataclass
class Controls:
    total_time: float = 0.0
    dt: float = 0.0
    sim_duration: float = 0.0
    advection: bool = False
    diffusion: bool = False
    adv_type: str = "upwind" # upwind, central, QUICK
    quick_up: bool = False
    quick_up_ratio: float = 0.0
    time_disc: str = "imp" # exp, imp, semi
    
@dataclass
class EquationCoef:
    # Arrays of size NC
    A: np.ndarray
    B: np.ndarray
    e: np.ndarray
    f: np.ndarray
    g: np.ndarray
    
    b_up: np.ndarray
    e_up: np.ndarray
    f_up: np.ndarray
    
    l_left_coef: np.ndarray
    left_coef: np.ndarray
    center_coef: np.ndarray
    right_coef: np.ndarray
    r_right_coef: np.ndarray
    ti_coef: np.ndarray

# --- Main Engine Class ---

class RiverModel:
    def __init__(self):
        self.gr = None
        self.flow = None
        self.atm = None
        self.ctrl = None
        self.props = {} 
        self.coef = None
        self.results_store = {} 

    def initialize(self, grid_params, flow_params, atm_params, ctrl_params, prop_params_dict):
        # 1. Grid
        self.gr = Grid(**grid_params)
        self.gr.compute_geometry()
        
        # 2. Controls
        self.ctrl = Controls(**ctrl_params)
        
        # 3. Flow
        self.flow = FlowProperties(**flow_params)
        if self.ctrl.advection:
            self.flow.courant_nr = self.ctrl.dt * self.flow.velocity / self.gr.dx
        else:
            self.flow.courant_nr = 0.0
            
        if self.ctrl.diffusion:
            self.flow.diffusion_nr = self.ctrl.dt * self.flow.diffusivity / (self.gr.dx**2)
        else:
            self.flow.diffusion_nr = 0.0
            
        # 4. Atmosphere
        self.atm = Atmosphere(**atm_params)
        
        # 5. Allocate Coefficients
        nc = self.gr.nc
        self.coef = EquationCoef(
            A=np.zeros(nc), B=np.zeros(nc), e=np.zeros(nc), f=np.zeros(nc), g=np.zeros(nc),
            b_up=np.zeros(nc), e_up=np.zeros(nc), f_up=np.zeros(nc),
            l_left_coef=np.zeros(nc), left_coef=np.zeros(nc), center_coef=np.zeros(nc),
            right_coef=np.zeros(nc), r_right_coef=np.zeros(nc), ti_coef=np.zeros(nc)
        )
        
        # 6. Properties Creation & Initialization
        for name, p_data in prop_params_dict.items():
            # Extract Init Config
            init_data = p_data.get('init_config', {})
            init_cfg = InitConfig(
                type=init_data.get('type', 'DEFAULT'),
                default_val=init_data.get('default_val', 0.0),
                intervals=init_data.get('intervals', [])
            )
            
            # Base Property
            prop = Property(name=name, init_config=init_cfg, **p_data['base'])
            
            # Boundary
            prop.boundary = BoundaryCondition(**p_data['boundary'])
            prop.boundary.discharges = [Discharge(**d) for d in p_data['discharges']]
            
            # Alloc Arrays
            prop.current_val = np.zeros(nc)
            prop.old_val = np.zeros(nc)
            prop.integrated_aero = np.zeros(nc)
            prop.integrated_anaero = np.zeros(nc)
            
            # Apply Initial Conditions (Sub CreateProperty logic)
            self._apply_initial_conditions(prop)
            
            self.props[name] = prop
            self.results_store[name] = []

    def _apply_initial_conditions(self, prop: Property):
        # VBA Logic: "InitType" check
        nc = self.gr.nc
        dx = self.gr.dx
        cfg = prop.init_config
        
        # First, fill with default
        prop.current_val.fill(cfg.default_val)
        
        if cfg.type == "CELL":
            # List of (cell_idx, _, value)
            for item in cfg.intervals:
                c_idx = int(item[0])
                val = item[2]
                if 0 <= c_idx < nc:
                    prop.current_val[c_idx] = val
                    
        elif cfg.type == "INTERVAL_M": # Meters
            # (start_m, end_m, value)
            for item in cfg.intervals:
                x1, x2, val = item
                if x1 < 0 or x2 > self.gr.length: continue
                # Logic from VBA: i = (X1 + gr.dx / 2) \ gr.dx
                # This finds the cell index corresponding to the distance
                # VBA loops through distance steps. We can use indices.
                start_cell = int((x1 + dx/2) // dx)
                end_cell = int((x2 + dx/2) // dx)
                # VBA is inclusive loop until X1 < X2
                for i in range(start_cell, end_cell):
                    if 0 <= i < nc:
                        prop.current_val[i] = val

    def _zeros_coeffs(self):
        c = self.coef
        c.A.fill(0); c.B.fill(0); c.e.fill(0); c.f.fill(0); c.g.fill(0)
        c.l_left_coef.fill(0); c.left_coef.fill(0); c.center_coef.fill(0)
        c.right_coef.fill(0); c.r_right_coef.fill(0); c.ti_coef.fill(0)

    # --- Transport Logic (Exact Replica) ---
    def _coef_transport(self):
        nc = self.gr.nc
        f = self.flow
        c = self.coef
        
        # VBA lines 36-56
        if self.ctrl.diffusion:
            c.A[0] = 0; c.B[0] = 0; c.e[0] = -f.diffusion_nr; c.f[0] = f.diffusion_nr; c.g[0] = 0
            c.A[nc-1] = 0; c.B[nc-1] = f.diffusion_nr; c.e[nc-1] = -f.diffusion_nr; c.f[nc-1] = 0; c.g[nc-1] = 0
            
            c.A[1:nc-1] = 0
            c.B[1:nc-1] = f.diffusion_nr
            c.e[1:nc-1] = -2 * f.diffusion_nr
            c.f[1:nc-1] = f.diffusion_nr
            c.g[1:nc-1] = 0
            
        if self.ctrl.advection:
            if self.ctrl.adv_type == "upwind":
                if f.velocity > 0:
                    c.B[:] += f.courant_nr
                    c.e[:] -= f.courant_nr
                else:
                    c.e[:] += f.courant_nr
                    c.f[:] -= f.courant_nr
            
            elif self.ctrl.adv_type == "QUICK":
                if self.ctrl.quick_up:
                    if f.velocity > 0:
                        c.b_up[:] = c.B + f.courant_nr
                        c.e_up[:] = c.e - f.courant_nr
                        c.f_up[:] = c.f
                    else:
                        c.b_up[:] = c.B
                        c.e_up[:] = c.e + f.courant_nr
                        c.f_up[:] = c.f - f.courant_nr
                
                if f.velocity > 0:
                    # Inner: 1 to nc-2
                    idx = slice(1, nc-1)
                    c.A[idx] -= (1/8)*f.courant_nr
                    c.B[idx] += (6/8)*f.courant_nr
                    c.e[idx] += (3/8)*f.courant_nr
                    c.B[idx] += (1/8)*f.courant_nr
                    c.e[idx] -= (6/8)*f.courant_nr
                    c.f[idx] -= (3/8)*f.courant_nr
                    
                    c.B[0] += (9/8)*f.courant_nr
                    c.e[0] -= (6/8)*f.courant_nr
                    c.f[0] -= (3/8)*f.courant_nr
                    
                    c.A[nc-1] -= (1/8)*f.courant_nr
                    c.B[nc-1] += (6/8)*f.courant_nr
                    c.e[nc-1] -= (5/8)*f.courant_nr
                    c.f[nc-1] = 0
                else:
                    idx = slice(1, nc-1)
                    c.B[idx] += (3/8)*f.courant_nr
                    c.e[idx] += (6/8)*f.courant_nr
                    c.f[idx] -= (1/8)*f.courant_nr
                    c.e[idx] -= (3/8)*f.courant_nr
                    c.f[idx] -= (6/8)*f.courant_nr
                    c.g[idx] += (1/8)*f.courant_nr
                    
                    c.B[0] = 0
                    c.e[0] += (5/8)*f.courant_nr
                    c.f[0] -= (6/8)*f.courant_nr
                    c.g[0] += (1/8)*f.courant_nr
                    
                    c.B[nc-1] += (3/8)*f.courant_nr
                    c.e[nc-1] += (6/8)*f.courant_nr
                    c.f[nc-1] -= (9/8)*f.courant_nr

            elif self.ctrl.adv_type == "central":
                c.B[1:nc-1] += f.courant_nr / 2
                c.f[1:nc-1] -= f.courant_nr / 2
                if f.velocity > 0:
                    c.B[0] += f.courant_nr; c.e[0] -= f.courant_nr
                    c.B[nc-1] += f.courant_nr; c.e[nc-1] -= f.courant_nr
                else:
                    c.e[0] += f.courant_nr; c.f[0] -= f.courant_nr
                    c.e[nc-1] += f.courant_nr; c.f[nc-1] -= f.courant_nr

    def _transport(self, prop: Property):
        if self.ctrl.adv_type == "QUICK":
            if self.ctrl.time_disc == "exp": self._exp_quick_transport(prop)
            elif self.ctrl.time_disc == "imp": self._imp_quick_transport(prop)
            else: self._semi_imp_quick_transport(prop)
        else:
            if self.ctrl.time_disc == "exp": self._exp_transport(prop)
            elif self.ctrl.time_disc == "imp": self._imp_transport(prop)
            else: self._semi_imp_transport(prop)

    def _exp_transport(self, prop: Property):
        nc = self.gr.nc
        c = self.coef
        prop.old_val[:] = prop.current_val[:]
        c.left_coef[:] = c.B[:]
        c.center_coef[:] = 1 + c.e[:]
        c.right_coef[:] = c.f[:]
        
        prop.current_val[0] = c.left_coef[0]*prop.boundary.left_value + c.center_coef[0]*prop.old_val[0] + c.right_coef[0]*prop.old_val[1]
        prop.current_val[1:nc-1] = c.left_coef[1:nc-1] * prop.old_val[0:nc-2] + c.center_coef[1:nc-1] * prop.old_val[1:nc-1] + c.right_coef[1:nc-1] * prop.old_val[2:nc]
        prop.current_val[nc-1] = c.left_coef[nc-1]*prop.old_val[nc-2] + c.center_coef[nc-1]*prop.old_val[nc-1] + c.right_coef[nc-1]*prop.boundary.right_value

    def _imp_transport(self, prop: Property):
        nc = self.gr.nc
        c = self.coef
        prop.old_val[:] = prop.current_val[:]
        c.ti_coef[:] = prop.old_val[:]
        c.left_coef[:] = -c.B[:]
        c.center_coef[:] = 1 - c.e[:]
        c.right_coef[:] = -c.f[:]
        
        c.ti_coef[0] -= c.left_coef[0] * prop.boundary.left_value
        c.ti_coef[nc-1] -= c.right_coef[nc-1] * prop.boundary.right_value
        
        # TDMA
        for i in range(1, nc):
            f_fac = c.left_coef[i] / c.center_coef[i-1]
            c.center_coef[i] -= f_fac * c.right_coef[i-1]
            c.ti_coef[i] -= f_fac * c.ti_coef[i-1]
            
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        for i in range(nc-2, -1, -1):
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i] * prop.current_val[i+1]) / c.center_coef[i]

    def _semi_imp_transport(self, prop: Property):
        nc = self.gr.nc
        c = self.coef
        prop.old_val[:] = prop.current_val[:]
        c.ti_coef[:] = prop.old_val[:]
        
        c.left_coef[:] = -c.B[:] / 2
        c.center_coef[:] = 1 - c.e[:] / 2
        c.right_coef[:] = -c.f[:] / 2
        
        c.ti_coef[0] = c.ti_coef[0] - c.left_coef[0]*prop.boundary.left_value + 0.5*(c.B[0]*prop.boundary.left_value + c.e[0]*prop.old_val[0] + c.f[0]*prop.old_val[1])
        c.ti_coef[nc-1] = c.ti_coef[nc-1] - c.right_coef[nc-1]*prop.boundary.right_value + 0.5*(c.B[nc-1]*prop.old_val[nc-2] + c.e[nc-1]*prop.old_val[nc-1] + c.f[nc-1]*prop.boundary.right_value)
        
        for i in range(1, nc-1):
            c.ti_coef[i] += 0.5 * (c.B[i]*prop.old_val[i-1] + c.e[i]*prop.old_val[i] + c.f[i]*prop.old_val[i+1])
            
        # TDMA
        for i in range(1, nc):
            f_fac = c.left_coef[i] / c.center_coef[i-1]
            c.center_coef[i] -= f_fac * c.right_coef[i-1]
            c.ti_coef[i] -= f_fac * c.ti_coef[i-1]
            
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        for i in range(nc-2, -1, -1):
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i] * prop.current_val[i+1]) / c.center_coef[i]

    def _imp_quick_transport(self, prop: Property):
        nc = self.gr.nc
        c = self.coef
        prop.old_val[:] = prop.current_val[:]
        c.ti_coef[:] = prop.old_val[:]
        
        c.l_left_coef[:] = -c.A[:]
        c.left_coef[:] = -c.B[:]
        c.center_coef[:] = 1 - c.e[:]
        c.right_coef[:] = -c.f[:]
        c.r_right_coef[:] = -c.g[:]
        
        if self.ctrl.quick_up:
            for i in range(1, nc-1):
                C_val = max(abs(prop.old_val[i+1] - prop.old_val[i-1]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                A_val = max(abs(prop.old_val[i] - prop.old_val[i-1]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                B_val = max(abs(prop.old_val[i+1] - prop.old_val[i]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                if abs(A_val - C_val) > self.ctrl.quick_up_ratio*A_val or abs(B_val - C_val) > self.ctrl.quick_up_ratio*B_val:
                    c.l_left_coef[i] = 0; c.left_coef[i] = -c.b_up[i]; c.center_coef[i] = 1 - c.e_up[i]; c.right_coef[i] = -c.f_up[i]; c.r_right_coef[i] = 0
        
        c.ti_coef[0] -= c.left_coef[0] * prop.boundary.left_value
        c.ti_coef[1] -= c.l_left_coef[1] * prop.boundary.left_value
        c.ti_coef[nc-2] -= c.r_right_coef[nc-2] * prop.boundary.right_value
        c.ti_coef[nc-1] -= c.right_coef[nc-1] * prop.boundary.right_value
        
        # Pentadiagonal Solver
        for i in range(2, nc):
            f_fac = c.left_coef[i-1] / c.center_coef[i-2]
            c.left_coef[i-1] -= f_fac * c.center_coef[i-2]
            c.center_coef[i-1] -= f_fac * c.right_coef[i-2]
            c.right_coef[i-1] -= f_fac * c.r_right_coef[i-2]
            c.ti_coef[i-1] -= f_fac * c.ti_coef[i-2]
            
            f_fac = c.l_left_coef[i] / c.center_coef[i-2]
            c.l_left_coef[i] -= f_fac * c.center_coef[i-2]
            c.left_coef[i] -= f_fac * c.right_coef[i-2]
            c.center_coef[i] -= f_fac * c.r_right_coef[i-2]
            c.ti_coef[i] -= f_fac * c.ti_coef[i-2]
            
        f_fac = c.left_coef[nc-1] / c.center_coef[nc-2]
        c.left_coef[nc-1] -= f_fac * c.center_coef[nc-2]
        c.center_coef[nc-1] -= f_fac * c.right_coef[nc-2]
        c.right_coef[nc-1] -= f_fac * c.r_right_coef[nc-2]
        c.ti_coef[nc-1] -= f_fac * c.ti_coef[nc-2]
        
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        prop.current_val[nc-2] = (c.ti_coef[nc-2] - c.right_coef[nc-2]*prop.current_val[nc-1]) / c.center_coef[nc-2]
        for iaux in range(2, nc):
            i = nc - 1 - iaux
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i]*prop.current_val[i+1] - c.r_right_coef[i]*prop.current_val[i+2]) / c.center_coef[i]

    def _exp_quick_transport(self, prop: Property): pass 
    def _semi_imp_quick_transport(self, prop: Property): pass

    def _apply_discharges(self, prop: Property):
        if not prop.boundary.discharges: return
        dt = self.ctrl.dt
        vol_cell = self.gr.area_vertical * self.gr.dx
        for d in prop.boundary.discharges:
            if 0 <= d.cell < self.gr.nc:
                c_old = prop.current_val[d.cell]
                prop.current_val[d.cell] = (c_old * vol_cell + d.volume_rate * d.specific_value * dt) / (vol_cell + d.volume_rate * dt)

    def _solar_radiation(self, time_days):
        pi = math.pi
        atm_abs = 0.23
        im = self.atm.solar_constant * math.cos(self.atm.latitude * pi / 180) * (1 - atm_abs)
        hour = (time_days - math.floor(time_days)) * 24
        if hour < self.atm.tsr or hour > self.atm.tss: return 0.0
        return (1 - 0.75 * (self.atm.cloud_cover**3)) * im * math.sin(pi * (hour - self.atm.tsr) / (self.atm.tss - self.atm.tsr))

    def _free_surface_heat_flux(self):
        prop = self.props['Temperature']
        if not prop.active: return
        rho = 1000; cp = 4180; cB = 0.62
        dt = self.ctrl.dt; depth = self.gr.water_depth
        
        solar_flux = self._solar_radiation(self.ctrl.total_time / 86400.0)
        prop.current_val += dt * solar_flux / (rho * cp * depth)
        
        for i in range(self.gr.nc):
            temp = prop.current_val[i]
            if prop.boundary.fs_sensible_heat:
                h_val = self.atm.h_min + 0.345 * (self.atm.wind_speed**2)
                prop.current_val[i] += dt * (cB * h_val * (self.atm.temperature - temp)) / (rho * cp * depth)
            if prop.boundary.fs_latent_heat:
                def es(t): return 6.112 * math.exp((17.67 * t) / (t + 243.5))
                h_val = self.atm.h_min + 0.345 * (self.atm.wind_speed**2)
                flux_lat = -h_val * (es(temp) - self.atm.humidity * es(self.atm.temperature))
                if flux_lat > 0: flux_lat = 0
                prop.current_val[i] += dt * flux_lat / (rho * cp * depth)
            if prop.boundary.fs_radiative_heat:
                eps = 0.97; sig = 5.67e-8; tk = temp + 273.15
                if self.atm.sky_temp_imposed:
                    t_sky = self.atm.sky_temperature + 273.15
                    flux_rad = eps * sig * (t_sky**4 - tk**4)
                else:
                    t_air = self.atm.temperature + 273.15
                    sky_rad = 9.37e-6 * (t_air**6) * (1 + 0.17 * self.atm.cloud_cover**2)
                    flux_rad = eps * sig * (sky_rad - tk**4)
                prop.current_val[i] += dt * flux_rad / (rho * cp * depth)

    def _get_csat_henry(self, gas_prop: Property, temp: float):
        gp = gas_prop.boundary.gas_exchange_params
        if not gp or not gp.henry_temps: return 0.0
        
        temps = gp.henry_temps
        ks = gp.henry_ks
        
        j = 0
        while j < len(temps) and temp > temps[j]: j += 1
        
        if j == 0: k_henry = ks[0]
        elif j >= len(temps): k_henry = ks[-1]
        else:
            slope = (ks[j] - ks[j-1]) / (temps[j] - temps[j-1])
            k_henry = ks[j-1] + slope * (temp - temps[j-1])
            
        return k_henry * gp.partial_pressure * gp.molecular_weight

    def _free_surface_gas_fluxes(self, gas_name: str):
        if gas_name not in self.props: return
        gas = self.props[gas_name]
        if not gas.active or not gas.boundary.free_surface_flux: return
        
        temp_prop = self.props['Temperature']
        for i in range(self.gr.nc):
            temp = temp_prop.current_val[i]
            csat = self._get_csat_henry(gas, temp)
            k_ex = (1 / self.gr.water_depth) * 0.142 * (abs(self.atm.wind_speed) + 0.1) * (self.flow.river_slope + 0.00001)
            if (csat - gas.current_val[i]) < 0:
                 k_ex *= (1 + 20 * (gas.current_val[i] - csat) / (csat + gas.current_val[i]))
            gas.current_val[i] = (gas.current_val[i] + self.ctrl.dt * k_ex * csat) / (1 + self.ctrl.dt * k_ex)

    def _sinks_bod(self):
        bod = self.props.get('BOD')
        do = self.props.get('DO')
        temp = self.props.get('Temperature')
        co2 = self.props.get('CO2')
        if not (bod and bod.active): return
        dt = self.ctrl.dt
        
        for i in range(self.gr.nc):
            if bod.current_val[i] < bod.max_val_logistic:
                bod.current_val[i] *= (1 + dt * (bod.decay_rate + bod.growth_rate) * (1 - bod.current_val[i]/bod.max_val_logistic)**0.5)
            
            bod_before = bod.current_val[i]
            temp_accel = 1.047**(temp.current_val[i] - 20)
            do_val = do.current_val[i] if do else 0.0
            decay_rate = temp_accel * bod.decay_rate * (do_val / (bod.grazing_ksat + do_val + 0.01))
            
            bod_decay = bod.current_val[i] * decay_rate * dt
            if bod_decay > (bod_before + bod.min_val): bod_decay = bod_before - bod.min_val
            if do and bod_decay > (do.current_val[i] + do.min_val): bod_decay = do.current_val[i] - do.min_val
            
            aerobic_rate = bod_decay / bod.current_val[i] / dt if bod.current_val[i] > 0 else 0
            
            bod.current_val[i] -= bod_decay
            if do: do.current_val[i] -= bod_decay
            if co2: co2.current_val[i] += bod_decay * (44/32)
            bod.integrated_aero[i] += bod_decay * self.gr.dx * self.gr.river_width * self.gr.water_depth / 1000.0
            
            if bod.anaerobic_respiration:
                anaero_rate = temp_accel * bod.decay_rate - aerobic_rate
                if anaero_rate > 0:
                    aux = bod.current_val[i]
                    bod.current_val[i] /= (1 + anaero_rate * dt)
                    anaero_decay = aux - bod.current_val[i]
                    bod.integrated_anaero[i] += anaero_decay * self.gr.dx * self.gr.river_width * self.gr.water_depth / 1000.0
                    if co2: co2.current_val[i] += 0.5 * (anaero_decay * (44/32))

    def _sinks_generic(self, prop: Property):
        dt = self.ctrl.dt
        if prop.min_active: prop.current_val = np.maximum(prop.current_val * (1 + prop.decay_rate * dt), prop.min_val)
        else: prop.current_val *= (1 + prop.decay_rate * dt)

    def run(self, progress_callback=None):
        steps = int(self.ctrl.sim_duration / self.ctrl.dt)
        print_int = int(steps/20) if steps > 20 else 1
        
        for p in self.props.values(): self.results_store[p.name].append(p.current_val.copy())
            
        for step in range(steps):
            self.ctrl.total_time += self.ctrl.dt
            self._zeros_coeffs()
            
            for p_name in ['Generic', 'Temperature', 'DO', 'BOD', 'CO2']:
                if p_name in self.props and self.props[p_name].active: self._apply_discharges(self.props[p_name])
                    
            if 'Temperature' in self.props:
                t = self.props['Temperature']
                if t.boundary.cyclic:
                    if self.flow.velocity > 0: t.boundary.left_value = t.current_val[-1]
                    else: t.boundary.right_value = t.current_val[0]
                    
            self._coef_transport()
            
            for p_name in ['Generic', 'Temperature', 'DO', 'BOD', 'CO2']:
                if p_name in self.props and self.props[p_name].active: self._transport(self.props[p_name])
            
            if 'Temperature' in self.props: self._free_surface_heat_flux()
            if 'DO' in self.props: self._free_surface_gas_fluxes('DO')
            if 'CO2' in self.props: self._free_surface_gas_fluxes('CO2')
            
            self._sinks_bod()
            if 'Generic' in self.props and self.props['Generic'].active: self._sinks_generic(self.props['Generic'])
            
            if step % 5 == 0:
                for p in self.props.values(): self.results_store[p.name].append(p.current_val.copy())
            if progress_callback and step % print_int == 0: progress_callback(step/steps)
            
        return self.results_store
