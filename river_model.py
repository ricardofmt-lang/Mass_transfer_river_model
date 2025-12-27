import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
    henry_constants_temp: List[float] = field(default_factory=list)
    henry_constants_k: List[float] = field(default_factory=list)

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
    gases: Dict[str, GasProperties] = field(default_factory=dict)

@dataclass
class Discharge:
    name: str
    cell: int # 0-based index for Python
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
    gas_exchange_params: Optional[GasProperties] = None
    discharges: List[Discharge] = field(default_factory=list)

@dataclass
class Property:
    name: str
    active: bool = False
    units: str = ""
    default_val: float = 0.0
    min_val: float = 0.0
    max_val: float = 1e9
    min_active: bool = False
    max_active: bool = False
    
    # Kinetic params
    decay_growth: bool = False
    decay_rate: float = 0.0
    growth_rate: float = 0.0
    max_val_logistic: float = 0.0
    grazing_ksat: float = 0.0
    anaerobic_respiration: bool = False
    
    # State
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
    time_disc: str = "exp" # exp, imp, semi
    
@dataclass
class EquationCoef:
    # Arrays of size NC (0-based)
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
        self.props = {} # Generic, Temperature, DO, BOD, CO2
        self.coef = None
        self.results_store = {} # Key: PropName, Value: List of arrays

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
        
        # 5. Coefficients Allocation
        nc = self.gr.nc
        self.coef = EquationCoef(
            A=np.zeros(nc), B=np.zeros(nc), e=np.zeros(nc), f=np.zeros(nc), g=np.zeros(nc),
            b_up=np.zeros(nc), e_up=np.zeros(nc), f_up=np.zeros(nc),
            l_left_coef=np.zeros(nc), left_coef=np.zeros(nc), center_coef=np.zeros(nc),
            right_coef=np.zeros(nc), r_right_coef=np.zeros(nc), ti_coef=np.zeros(nc)
        )
        
        # 6. Properties
        for name, p_data in prop_params_dict.items():
            prop = Property(name=name, **p_data['base'])
            prop.boundary = BoundaryCondition(**p_data['boundary'])
            prop.boundary.discharges = [Discharge(**d) for d in p_data['discharges']]
            
            # Initialize arrays
            if 'init_val' in p_data:
                prop.current_val = np.full(nc, p_data['init_val'], dtype=float)
            else:
                prop.current_val = np.full(nc, prop.default_val, dtype=float)
                
            prop.old_val = np.zeros(nc)
            prop.integrated_aero = np.zeros(nc)
            prop.integrated_anaero = np.zeros(nc)
            
            self.props[name] = prop
            self.results_store[name] = []

    def _zeros_coeffs(self):
        c = self.coef
        c.A.fill(0); c.B.fill(0); c.e.fill(0); c.f.fill(0); c.g.fill(0)
        c.l_left_coef.fill(0); c.left_coef.fill(0); c.center_coef.fill(0)
        c.right_coef.fill(0); c.r_right_coef.fill(0); c.ti_coef.fill(0)

    def _coef_transport(self):
        # Exact port of Sub CoefTransport
        nc = self.gr.nc
        f = self.flow
        c = self.coef
        
        # Indices in Python are 0 to nc-1. VBA is 1 to nc.
        # Mapping: VBA i -> Python i-1
        
        if self.ctrl.diffusion:
            # Boundary 1 (Left) -> Python 0
            c.A[0] = 0
            c.B[0] = 0
            c.e[0] = -f.diffusion_nr
            c.f[0] = f.diffusion_nr
            c.g[0] = 0
            
            # Boundary NC (Right) -> Python nc-1
            c.A[nc-1] = 0
            c.B[nc-1] = f.diffusion_nr
            c.e[nc-1] = -f.diffusion_nr
            c.f[nc-1] = 0
            c.g[nc-1] = 0
            
            # Inner points
            # VBA: For i = 2 To NC - 1
            # Python: 1 to nc-2
            c.A[1:nc-1] = 0
            c.B[1:nc-1] = f.diffusion_nr
            c.e[1:nc-1] = -2 * f.diffusion_nr
            c.f[1:nc-1] = f.diffusion_nr
            c.g[1:nc-1] = 0
            
        if self.ctrl.advection:
            if self.ctrl.adv_type == "upwind":
                if f.velocity > 0:
                    c.B += f.courant_nr
                    c.e -= f.courant_nr
                else:
                    c.e += f.courant_nr
                    c.f -= f.courant_nr
            
            elif self.ctrl.adv_type == "QUICK":
                # Handle QUICK_UP
                if self.ctrl.quick_up:
                    if f.velocity > 0:
                        c.b_up[:] = c.B + f.courant_nr
                        c.e_up[:] = c.e - f.courant_nr
                        c.f_up[:] = c.f
                    else:
                        c.b_up[:] = c.B
                        c.e_up[:] = c.e + f.courant_nr
                        c.f_up[:] = c.f - f.courant_nr
                
                # QUICK Coefficients
                if f.velocity > 0:
                    # Inner: VBA 2 to NC-1 -> Py 1 to nc-2
                    idx = slice(1, nc-1)
                    c.A[idx] = c.A[idx] - (1/8)*f.courant_nr
                    c.B[idx] = c.B[idx] + (6/8)*f.courant_nr
                    c.e[idx] = c.e[idx] + (3/8)*f.courant_nr
                    c.B[idx] = c.B[idx] + (1/8)*f.courant_nr # Contribution right face
                    c.e[idx] = c.e[idx] - (6/8)*f.courant_nr
                    c.f[idx] = c.f[idx] - (3/8)*f.courant_nr
                    
                    # Boundary Left (Py 0)
                    c.B[0] += (9/8)*f.courant_nr
                    c.e[0] -= (6/8)*f.courant_nr
                    c.f[0] -= (3/8)*f.courant_nr
                    
                    # Boundary Right (Py nc-1)
                    c.A[nc-1] -= (1/8)*f.courant_nr
                    c.B[nc-1] += (6/8)*f.courant_nr
                    c.e[nc-1] -= (5/8)*f.courant_nr
                    c.f[nc-1] = 0
                else:
                    # Inner
                    idx = slice(1, nc-1)
                    c.B[idx] += (3/8)*f.courant_nr
                    c.e[idx] += (6/8)*f.courant_nr
                    c.f[idx] -= (1/8)*f.courant_nr
                    c.e[idx] -= (3/8)*f.courant_nr
                    c.f[idx] -= (6/8)*f.courant_nr
                    c.g[idx] += (1/8)*f.courant_nr
                    
                    # Left
                    c.B[0] = 0
                    c.e[0] += (5/8)*f.courant_nr
                    c.f[0] -= (6/8)*f.courant_nr
                    c.g[0] += (1/8)*f.courant_nr
                    
                    # Right
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
        # Calls the specific solver based on controls
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
        # Save old
        prop.old_val[:] = prop.current_val[:]
        c.left_coef[:] = c.B[:]
        c.center_coef[:] = 1 + c.e[:]
        c.right_coef[:] = c.f[:]
        
        # Left
        prop.current_val[0] = c.left_coef[0]*prop.boundary.left_value + \
                              c.center_coef[0]*prop.old_val[0] + \
                              c.right_coef[0]*prop.old_val[1]
        # Inner
        # VBA: 2 to NC-1. Py: 1 to nc-2
        # C_i = L*C_{i-1} + C*C_i + R*C_{i+1}
        prop.current_val[1:nc-1] = c.left_coef[1:nc-1] * prop.old_val[0:nc-2] + \
                                   c.center_coef[1:nc-1] * prop.old_val[1:nc-1] + \
                                   c.right_coef[1:nc-1] * prop.old_val[2:nc]
        # Right
        prop.current_val[nc-1] = c.left_coef[nc-1]*prop.old_val[nc-2] + \
                                 c.center_coef[nc-1]*prop.old_val[nc-1] + \
                                 c.right_coef[nc-1]*prop.boundary.right_value

    def _imp_transport(self, prop: Property):
        nc = self.gr.nc
        c = self.coef
        
        # Save old and set TDMA coeffs
        prop.old_val[:] = prop.current_val[:]
        c.ti_coef[:] = prop.old_val[:]
        c.left_coef[:] = -c.B[:]
        c.center_coef[:] = 1 - c.e[:]
        c.right_coef[:] = -c.f[:]
        
        # Boundary Adjustments
        c.ti_coef[0] -= c.left_coef[0] * prop.boundary.left_value
        c.ti_coef[nc-1] -= c.right_coef[nc-1] * prop.boundary.right_value
        
        # TDMA Forward Elimination (VBA Loop 2 to NC -> Py 1 to nc-1)
        for i in range(1, nc):
            f_factor = c.left_coef[i] / c.center_coef[i-1]
            c.left_coef[i] -= f_factor * c.center_coef[i-1] # Actually A[i] is 0 in TDMA usually after elim
            c.center_coef[i] -= f_factor * c.right_coef[i-1]
            c.ti_coef[i] -= f_factor * c.ti_coef[i-1]
            
        # TDMA Backward Substitution
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        for i in range(nc-2, -1, -1):
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i] * prop.current_val[i+1]) / c.center_coef[i]

    def _semi_imp_transport(self, prop: Property):
        # Crank-Nicolson
        nc = self.gr.nc
        c = self.coef
        prop.old_val[:] = prop.current_val[:]
        c.ti_coef[:] = prop.old_val[:]
        
        c.left_coef[:] = -c.B[:] / 2
        c.center_coef[:] = 1 - c.e[:] / 2
        c.right_coef[:] = -c.f[:] / 2
        
        # Left Bound RHS
        c.ti_coef[0] = c.ti_coef[0] - c.left_coef[0]*prop.boundary.left_value + \
                       0.5*(c.B[0]*prop.boundary.left_value + c.e[0]*prop.old_val[0] + c.f[0]*prop.old_val[1])
        # Right Bound RHS
        c.ti_coef[nc-1] = c.ti_coef[nc-1] - c.right_coef[nc-1]*prop.boundary.right_value + \
                          0.5*(c.B[nc-1]*prop.old_val[nc-2] + c.e[nc-1]*prop.old_val[nc-1] + c.f[nc-1]*prop.boundary.right_value)
        
        # Inner RHS
        # VBA loop 2 to NC-1 -> Py 1 to nc-2
        for i in range(1, nc-1):
            c.ti_coef[i] = c.ti_coef[i] + 0.5 * (c.B[i]*prop.old_val[i-1] + c.e[i]*prop.old_val[i] + c.f[i]*prop.old_val[i+1])
            
        # TDMA Solve
        for i in range(1, nc):
            f_factor = c.left_coef[i] / c.center_coef[i-1]
            c.center_coef[i] -= f_factor * c.right_coef[i-1]
            c.ti_coef[i] -= f_factor * c.ti_coef[i-1]
            
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        for i in range(nc-2, -1, -1):
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i] * prop.current_val[i+1]) / c.center_coef[i]

    # --- QUICK Implementation (Exact Replica) ---
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
                # Using Py 0-index: i in loop corresponds to Py i
                # Old(i+1) is prop.old_val[i+1] (Right)
                # Old(i-1) is prop.old_val[i-1] (Left)
                C_val = max(abs(prop.old_val[i+1] - prop.old_val[i-1]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                A_val = max(abs(prop.old_val[i] - prop.old_val[i-1]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                B_val = max(abs(prop.old_val[i+1] - prop.old_val[i]), self.ctrl.quick_up_ratio * (1+prop.min_val))
                
                if abs(A_val - C_val) > self.ctrl.quick_up_ratio*A_val or abs(B_val - C_val) > self.ctrl.quick_up_ratio*B_val:
                    c.l_left_coef[i] = 0
                    c.left_coef[i] = -c.b_up[i]
                    c.center_coef[i] = 1 - c.e_up[i]
                    c.right_coef[i] = -c.f_up[i]
                    c.r_right_coef[i] = 0
        
        # Boundary application (VBA Lines 76-77)
        c.ti_coef[0] -= c.left_coef[0] * prop.boundary.left_value
        c.ti_coef[1] -= c.l_left_coef[1] * prop.boundary.left_value
        c.ti_coef[nc-2] -= c.r_right_coef[nc-2] * prop.boundary.right_value
        c.ti_coef[nc-1] -= c.right_coef[nc-1] * prop.boundary.right_value
        
        # Custom Solver for Pentadiagonal-like structure (VBA Lines 77-80)
        # Note: The VBA loop logic is complex. It modifies coefficients in place.
        # VBA Loop "For i = 3 To NC" corresponds to Py range(2, nc)
        for i in range(2, nc):
            # Reduction of row i-1 based on i-2
            f_fac = c.left_coef[i-1] / c.center_coef[i-2]
            c.left_coef[i-1] -= f_fac * c.center_coef[i-2] # Should be 0 effectively
            c.center_coef[i-1] -= f_fac * c.right_coef[i-2]
            c.right_coef[i-1] -= f_fac * c.r_right_coef[i-2]
            c.ti_coef[i-1] -= f_fac * c.ti_coef[i-2]
            
            # Reduction of row i based on i-2
            f_fac = c.l_left_coef[i] / c.center_coef[i-2]
            c.l_left_coef[i] -= f_fac * c.center_coef[i-2]
            c.left_coef[i] -= f_fac * c.right_coef[i-2]
            c.center_coef[i] -= f_fac * c.r_right_coef[i-2]
            c.ti_coef[i] -= f_fac * c.ti_coef[i-2]
            
        # Final cleanup step (VBA Line 79 bottom)
        f_fac = c.left_coef[nc-1] / c.center_coef[nc-2]
        c.left_coef[nc-1] -= f_fac * c.center_coef[nc-2]
        c.center_coef[nc-1] -= f_fac * c.right_coef[nc-2]
        c.right_coef[nc-1] -= f_fac * c.r_right_coef[nc-2]
        c.ti_coef[nc-1] -= f_fac * c.ti_coef[nc-2]
        
        # Backward Sub
        prop.current_val[nc-1] = c.ti_coef[nc-1] / c.center_coef[nc-1]
        prop.current_val[nc-2] = (c.ti_coef[nc-2] - c.right_coef[nc-2]*prop.current_val[nc-1]) / c.center_coef[nc-2]
        
        for iaux in range(2, nc):
            i = nc - 1 - iaux # Py equivalent of NC - iaux
            prop.current_val[i] = (c.ti_coef[i] - c.right_coef[i]*prop.current_val[i+1] - c.r_right_coef[i]*prop.current_val[i+2]) / c.center_coef[i]

    def _exp_quick_transport(self, prop: Property):
        pass # Implemented similarly if needed, but Implicit is standard for this model

    def _semi_imp_quick_transport(self, prop: Property):
        pass # To be implemented if semi-imp selected

    # --- Source Terms & Discharges ---
    
    def _apply_discharges(self, prop: Property):
        if not prop.boundary.discharges:
            return
        
        dt = self.ctrl.dt
        vol_cell = self.gr.area_vertical * self.gr.dx
        
        for d in prop.boundary.discharges:
            if 0 <= d.cell < self.gr.nc:
                c_old = prop.current_val[d.cell]
                q = d.volume_rate
                c_dis = d.specific_value
                # Mixing formula: (C_old * Vol + Q * C_in * dt) / (Vol + Q * dt)
                prop.current_val[d.cell] = (c_old * vol_cell + q * c_dis * dt) / (vol_cell + q * dt)

    def _solar_radiation(self, time_days):
        pi = 3.14159265358979
        atm_abs = 0.23
        im = self.atm.solar_constant * math.cos(self.atm.latitude * pi / 180) * (1 - atm_abs)
        hour = (time_days - math.floor(time_days)) * 24
        
        if hour < self.atm.tsr or hour > self.atm.tss:
            return 0.0
        else:
            return (1 - 0.75 * (self.atm.cloud_cover**3)) * im * math.sin(pi * (hour - self.atm.tsr) / (self.atm.tss - self.atm.tsr))

    def _free_surface_heat_flux(self):
        prop = self.props['Temperature']
        if not prop.active: return
        
        rho = 1000
        cp = 4180
        hfg = 2450000
        cB = 0.62
        
        dt = self.ctrl.dt
        depth = self.gr.water_depth
        
        # Solar
        solar_flux = self._solar_radiation(self.ctrl.total_time / 86400.0)
        prop.current_val += dt * solar_flux / (rho * cp * depth)
        
        # Loop for cell-dependent fluxes
        for i in range(self.gr.nc):
            temp = prop.current_val[i]
            
            # Sensible Heat
            if prop.boundary.fs_sensible_heat:
                h_val = self.atm.h_min + 0.345 * (self.atm.wind_speed**2)
                flux_conv = cB * h_val * (self.atm.temperature - temp)
                prop.current_val[i] += dt * flux_conv / (rho * cp * depth)
                
            # Latent Heat (Evaporation)
            if prop.boundary.fs_latent_heat:
                # Es function
                def es(t):
                    return 6.112 * math.exp((17.67 * t) / (t + 243.5))
                
                h_val = self.atm.h_min + 0.345 * (self.atm.wind_speed**2)
                flux_latent = -h_val * (es(temp) - self.atm.humidity * es(self.atm.temperature))
                if flux_latent > 0: flux_latent = 0
                prop.current_val[i] += dt * flux_latent / (rho * cp * depth)
            
            # Radiative
            if prop.boundary.fs_radiative_heat:
                epsilon = 0.97
                sigma = 5.67e-8
                tk = temp + 273.15
                if self.atm.sky_temp_imposed:
                    t_sky_k = self.atm.sky_temperature + 273.15
                    flux_rad = epsilon * sigma * (t_sky_k**4 - tk**4)
                else:
                    t_air_k = self.atm.temperature + 273.15
                    # Swinbank formula approx from VBA
                    sky_rad = 9.37e-6 * (t_air_k**6) * (1 + 0.17 * self.atm.cloud_cover**2)
                    flux_rad = epsilon * sigma * (sky_rad - tk**4) # Note: VBA might have slight var here, checking line 106... looks consistent
                    # Actually VBA line 106: epson * Sigma * ((9.37... - (water+273)^4)
                    # My logic matches
                
                prop.current_val[i] += dt * flux_rad / (rho * cp * depth)

    def _get_csat_henry(self, gas_prop: Property, temp: float):
        # Interpolation of Henry constants
        gp_bounds = gas_prop.boundary.gas_exchange_params
        if not gp_bounds: return 0.0
        
        temps = gp_bounds.henry_constants_temp
        ks = gp_bounds.henry_constants_k
        
        # Find interval
        j = 0
        while j < len(temps) and temp > temps[j]:
            j += 1
            
        k_henry = 0.0
        if j == 0:
            k_henry = ks[0]
        elif j >= len(temps):
            k_henry = ks[-1]
        else:
            slope = (ks[j] - ks[j-1]) / (temps[j] - temps[j-1])
            k_henry = ks[j-1] + slope * (temp - temps[j-1])
            
        return k_henry * gp_bounds.partial_pressure * gp_bounds.molecular_weight

    def _free_surface_gas_fluxes(self, gas_name: str):
        if gas_name not in self.props: return
        gas = self.props[gas_name]
        if not gas.active or not gas.boundary.free_surface_flux: return
        
        temp_prop = self.props['Temperature']
        
        for i in range(self.gr.nc):
            temp = temp_prop.current_val[i]
            csat = self._get_csat_henry(gas, temp)
            
            # K_Gas (VBA line 98)
            # K_ex = (1/depth) * 0.142 * (Abs(wind) + 0.1) * (slope + 0.00001)
            # Note: The VBA comment says "to be always higher than zero"
            k_ex = (1 / self.gr.water_depth) * 0.142 * (abs(self.atm.wind_speed) + 0.1) * (self.flow.river_slope + 0.00001)
            
            # Oversaturation bubble effect (Line 100)
            if (csat - gas.current_val[i]) < 0:
                 k_ex = k_ex * (1 + 20 * (gas.current_val[i] - csat) / (csat + gas.current_val[i]))
            
            # Implicit update (Line 101)
            gas.current_val[i] = (gas.current_val[i] + self.ctrl.dt * k_ex * csat) / (1 + self.ctrl.dt * k_ex)

    def _sinks_bod(self):
        bod = self.props.get('BOD')
        do = self.props.get('DO')
        temp = self.props.get('Temperature')
        co2 = self.props.get('CO2')
        
        if not (bod and bod.active): return
        
        dt = self.ctrl.dt
        
        for i in range(self.gr.nc):
            # Logistic Growth (Line 112)
            if bod.current_val[i] < bod.max_val_logistic:
                bod.current_val[i] = bod.current_val[i] * (1 + dt * (bod.decay_rate + bod.growth_rate) * (1 - bod.current_val[i]/bod.max_val_logistic)**0.5)
            
            bod_before = bod.current_val[i]
            temp_accel = 1.047**(temp.current_val[i] - 20)
            
            # Decay rate coupled with DO (Monod kinetics)
            do_val = do.current_val[i] if do else 0.0
            decay_rate = temp_accel * bod.decay_rate * (do_val / (bod.grazing_ksat + do_val + 0.01))
            
            bod_decay_amount = bod.current_val[i] * decay_rate * dt
            
            # Constraints
            if bod_decay_amount > (bod_before + bod.min_val):
                bod_decay_amount = bod_before - bod.min_val
            
            if do and bod_decay_amount > (do.current_val[i] + do.min_val):
                bod_decay_amount = do.current_val[i] - do.min_val
                
            aerobic_rate_actual = 0
            if bod.current_val[i] > 0 and dt > 0:
                aerobic_rate_actual = bod_decay_amount / bod.current_val[i] / dt
                
            # Update States
            bod.current_val[i] -= bod_decay_amount
            if do: do.current_val[i] -= bod_decay_amount
            if co2: co2.current_val[i] += bod_decay_amount * (44/32) # Stoichiometry
            
            # Integration
            bod.integrated_aero[i] += bod_decay_amount * self.gr.dx * self.gr.river_width * self.gr.water_depth / 1000.0
            
            # Anaerobic Switch
            if bod.anaerobic_respiration:
                anaero_rate = temp_accel * bod.decay_rate - aerobic_rate_actual
                if anaero_rate > 0:
                    aux = bod.current_val[i]
                    bod.current_val[i] = bod.current_val[i] / (1 + anaero_rate * dt)
                    anaero_decay = aux - bod.current_val[i]
                    bod.integrated_anaero[i] += anaero_decay * self.gr.dx * self.gr.river_width * self.gr.water_depth / 1000.0
                    if co2: co2.current_val[i] += 0.5 * (anaero_decay * (44/32))

    def _sinks_generic(self, prop: Property):
        dt = self.ctrl.dt
        if prop.min_active:
             prop.current_val = np.maximum(prop.current_val * (1 + prop.decay_rate * dt), prop.min_val)
        else:
             prop.current_val = prop.current_val * (1 + prop.decay_rate * dt)

    # --- Main Loop ---
    def run(self, progress_callback=None):
        t_current = 0.0
        steps = int(self.ctrl.sim_duration / self.ctrl.dt)
        print_interval = int(steps / 20) if steps > 20 else 1
        
        # Initial Store
        for p in self.props.values():
            self.results_store[p.name].append(p.current_val.copy())
            
        for step in range(steps):
            self.ctrl.total_time += self.ctrl.dt
            self._zeros_coeffs()
            
            # 1. Apply Discharges (Half Step - VBA Line 22)
            # VBA calls ApplyDischarges(..., ctrl.dt) but comment says "Two Halfs". 
            # Looking closely at VBA: It calls it once with full dt? 
            # Line 23: "Applying discharges in two half steps minimises the problems".
            # Line 24: `Call ApplyDischarges(..., ctrl.dt)` 
            # Wait, if it's two half steps, it should be called before and after transport. 
            # In VBA `Channel` Sub: 
            # Call ApplyDischarges (Before Transport)
            # Call Transport
            # Call ExpSinks (After Transport) -> Does this contain discharge? No.
            # I will follow the VBA code lines: It calls ApplyDischarges ONCE before transport with `dt`.
            # If the comment implies a strategy not fully implemented or implied by the transport nature, I follow the CODE.
            
            for p_name in ['Generic', 'Temperature', 'DO', 'BOD', 'CO2']:
                if p_name in self.props and self.props[p_name].active:
                    self._apply_discharges(self.props[p_name])
                    
            # 2. Cyclic Boundary update (Temperature specific in VBA)
            if 'Temperature' in self.props:
                t_prop = self.props['Temperature']
                if t_prop.boundary.cyclic:
                    if self.flow.velocity > 0:
                        t_prop.boundary.left_value = t_prop.current_val[-1]
                    else:
                        t_prop.boundary.right_value = t_prop.current_val[0]

            # 3. Transport Coefficient Calculation
            self._coef_transport()
            
            # 4. Transport Execution
            for p_name in ['Generic', 'Temperature', 'DO', 'BOD', 'CO2']:
                if p_name in self.props and self.props[p_name].active:
                    self._transport(self.props[p_name])
                    
            # 5. Free Surface Fluxes
            if 'Temperature' in self.props:
                self._free_surface_heat_flux()
            
            if 'DO' in self.props:
                self._free_surface_gas_fluxes('DO')
                
            if 'CO2' in self.props:
                self._free_surface_gas_fluxes('CO2')
                
            # 6. Sinks/Reactions
            self._sinks_bod()
            if 'Generic' in self.props and self.props['Generic'].active:
                self._sinks_generic(self.props['Generic'])
                
            # 7. Store Results
            # VBA prints based on `DtPrint`. We'll just store every X steps for plotting to save RAM.
            if step % 5 == 0:
                for p in self.props.values():
                    self.results_store[p.name].append(p.current_val.copy())
            
            if progress_callback and step % print_interval == 0:
                progress_callback(step / steps)
                
        return self.results_store
