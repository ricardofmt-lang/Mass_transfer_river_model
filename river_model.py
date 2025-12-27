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
    
    # Robustness: Accept init_val here even if unused by class logic directly
    # This prevents crashes if app.py passes it in 'base'
    init_val: Optional[float] = None
    
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
            # Initialize Property with 'base' args. 
            # Note: The modified dataclass now accepts 'init_val' silently if present in base.
            prop = Property(name=name, **p_data['base'])
            
            prop.boundary = BoundaryCondition(**p_data['boundary'])
            prop.boundary.discharges = [Discharge(**d) for d in p_data['discharges']]
            
            # Determine Initialization Value (Robust Logic)
            start_val = prop.default_val
            
            # Priority 1: explicitly passed in p_data (Correct app.py)
            if 'init_val' in p_data:
                start_val = p_data['init_val']
            # Priority 2: passed inside 'base' dict (Old app.py) - captured by dataclass
            elif prop.init_val is not None:
                start_val = prop.init_val
                
            prop.current_val = np.full(nc, start_val, dtype=float)
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
            c.left_coef[i] -= f_factor * c.center_coef[i-1] 
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
                    c.left_coef[i] = -c.b_up
