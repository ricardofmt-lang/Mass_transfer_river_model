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
    
    # Constraints
    min_val: float = -1e9
    max_val: float = 1e9
    
    # Boundary Conditions
    bc_left_type: str = "Fixed"
    bc_left_val: float = 0.0
    bc_right_type: str = "ZeroGrad"
    bc_right_val: float = 0.0
    
    # Flux Toggles
    use_surface_flux: bool = True     
    use_sensible_heat: bool = True    
    use_latent_heat: bool = True      
    use_radiative_heat: bool = True   
    
    # Kinetics
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
        # If area is 0, avoid div zero
        self.flow.velocity = discharge / self.grid.area_vertical if self.grid.area_vertical > 0 else 0.0

    def setup_atmos(self, temp, wind, humidity, solar, lat, cloud, sunrise, sunset, h_min, sky_temp, sky_imposed):
        # store atmospheric parameters.  Humidity and cloud cover are
        # specified by the user as percentages (0–100).  Convert them
        # to fractions for internal use.  Always impose sky temperature
        # (no automatic computation) — the VBA code treats any defined
        # sky temperature as imposed.
        self.atmos.air_temp = temp
        self.atmos.wind_speed = wind
        self.atmos.humidity = humidity / 100.0 if humidity > 1.0 else humidity
        self.atmos.solar_constant = solar
        self.atmos.latitude = lat
        self.atmos.cloud_cover = cloud / 100.0 if cloud > 1.0 else cloud
        self.atmos.sunrise_hour = sunrise
        self.atmos.sunset_hour = sunset
        self.atmos.h_min = h_min
        self.atmos.sky_temp = sky_temp
        # always impose sky temperature; per user request the sky temperature is mandatory
        self.atmos.sky_temp_imposed = True

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

    # -------------------------------------------------------------------------
    # PHYSICS
    # -------------------------------------------------------------------------

    def calculate_henry_constant(self, temp_c, gas_type="O2"):
        vals = self.atmos.henry_table_o2 if gas_type == "O2" else self.atmos.henry_table_co2
        return np.interp(temp_c, self.atmos.henry_table_temps, vals)

    def calculate_saturation(self, temp_c, gas_type):
        p_atm = (self.atmos.p_o2 if gas_type == "O2" else self.atmos.p_co2) / 1.01325
        kh = self.calculate_henry_constant(temp_c, gas_type)
        mw = 32000.0 if gas_type == "O2" else 44000.0
        return (p_atm * kh) * mw / 1000.0

    def calculate_heat_fluxes(self, water_temp, time_sec):
        """
        Compute heat flux components for a water parcel.

        This routine aims to mirror the VBA implementation found in the original
        Excel macros.  It returns a dictionary with the solar, sensible,
        latent and radiative heat fluxes (in W/m²).  The calling code is
        responsible for dividing by (rho*cp*depth) and multiplying by dt
        when updating temperature.  Humidity and cloud cover are expected
        to be supplied as fractions (0–1) at the point of call.  The sky
        temperature is always imposed by user choice; there is no automatic
        computation of an effective sky temperature in this model.
        """

        # physical constants
        sigma = 5.67e-8  # Stefan–Boltzmann constant (W/m²·K⁴)
        rho = 1000.0     # water density (kg/m³)
        cp = 4180.0      # water specific heat (J/kg·K)
        kelvin = 273.15

        # convert temperatures to Kelvin
        T_w_k = water_temp + kelvin
        T_a_k = self.atmos.air_temp + kelvin

        # convert humidity and cloud cover to fractions (0–1) for flux computation
        hum = self.atmos.humidity if self.atmos.humidity <= 1.0 else self.atmos.humidity / 100.0
        cloud = self.atmos.cloud_cover if self.atmos.cloud_cover <= 1.0 else self.atmos.cloud_cover / 100.0

        # Solar radiation as in VBA: uses cos(latitude), atmospheric absorption 0.23,
        # and a cloud attenuation of (1 - 0.75 * cloud³).  Only applies between
        # sunrise and sunset hours.  time_sec is simulation time in seconds.
        hour = (time_sec / 3600.0) % 24.0
        solar_flux = 0.0
        if self.atmos.sunrise_hour < hour < self.atmos.sunset_hour:
            day_len = self.atmos.sunset_hour - self.atmos.sunrise_hour
            norm_time = (hour - self.atmos.sunrise_hour) / day_len
            lat_rad = math.radians(self.atmos.latitude)
            # instantaneous solar after atmospheric absorption (0.23 per VBA)
            IM = self.atmos.solar_constant * math.cos(lat_rad) * (1.0 - 0.23)
            # cloud attenuation
            factor_cloud = (1.0 - 0.75 * (cloud ** 3))
            # diurnal sinusoid
            solar_flux = factor_cloud * IM * math.sin(math.pi * norm_time)

        # sensible heat flux: cB * h(wind, h_min) * (air_temp - water_temp)
        # where cB = 0.62 (mb/K) and h = h_min + 0.345 * wind²
        cB = 0.62
        h_coeff = self.atmos.h_min + 0.345 * (self.atmos.wind_speed ** 2)
        sensible = cB * h_coeff * (self.atmos.air_temp - water_temp)

        # latent heat flux: -h * (Es(Tw) - humidity * Es(Ta))
        # Es(T) returns saturation vapour pressure in Pa; humidity is fraction.
        def es_func(T):
            # Saturation vapour pressure (Pa) from VBA: A=6.112, B=17.67, C=243.5
            A = 6.112
            B = 17.67
            C = 243.5
            return A * math.exp(B * T / (T + C)) * 100.0  # convert from mb to Pa

        es_w = es_func(water_temp)
        es_a = es_func(self.atmos.air_temp)
        ea = hum * es_a
        latent = -h_coeff * (es_w - ea)
        # cap positive evaporation flux at zero (per VBA latent flux check)
        if latent > 0.0 and (es_w - ea) > 0.0:
            latent = 0.0

        # radiative longwave flux: depends on sky temperature (always imposed)
        if self.atmos.sky_temp_imposed:
            T_sky_k = self.atmos.sky_temp + kelvin
            radiative = 0.97 * sigma * ((T_sky_k ** 4) - (T_w_k ** 4))
        else:
            # Use empirical formulation when no imposed sky temperature; keep old behaviour
            radiative = 0.97 * sigma * ((9.37e-6 * (T_a_k ** 6) * (1.0 + 0.17 * (cloud ** 2))) - (T_w_k ** 4))

        return {
            "solar": solar_flux,
            "sensible": sensible,
            "latent": latent,
            "radiative": radiative,
        }

    def apply_discharges(self, dt_split):
        """
        Mix point discharges into the corresponding grid cell over a half time step.

        In the original VBA code, discharges are applied explicitly by mass
        balance: (C V + Q C_in dt) / (V + Q dt).  This implementation
        replicates that behaviour rather than the exponential relaxation used
        previously.

        dt_split is the half-step duration (s).  For stability, this function
        should be called twice per full time step: once before transport and
        once after.
        """
        vol_cell = self.grid.area_vertical * self.grid.dx
        for d in self.discharges:
            idx = d["cell"]
            q = d.get("flow", 0.0)
            if q <= 0:
                continue
            # volume ratio for this half step
            vol_ratio = q * dt_split
            denom = vol_cell + vol_ratio
            if denom <= 0:
                continue
            # temperature
            if "Temperature" in self.constituents:
                T_c = self.constituents["Temperature"].values[idx]
                T_in = d.get("temp", T_c)
                self.constituents["Temperature"].values[idx] = (T_c * vol_cell + T_in * vol_ratio) / denom
            # scalar constituents (BOD, DO, CO2, Generic)
            for name in ["BOD", "DO", "CO2", "Generic"]:
                if name in self.constituents:
                    C_c = self.constituents[name].values[idx]
                    key = name.lower()
                    val_in = d.get(key, C_c)
                    self.constituents[name].values[idx] = (C_c * vol_cell + val_in * vol_ratio) / denom

    def solve_transport(self, dt):
        # The core transport solver has been completely rewritten to
        # faithfully reproduce the VBA implementation.  It constructs
        # coefficient arrays for diffusion and advection and then
        # invokes explicit, semi-implicit or fully implicit solvers
        # depending on the selected discretisation.  QUICK advection is
        # supported with optional upwinding in steep gradient regions.

        u = self.flow.velocity if self.config.advection_active else 0.0
        D = self.flow.diffusivity if self.config.diffusion_active else 0.0
        dx = self.grid.dx
        nc = self.grid.nc
        # diffusion and Courant numbers (per cell)
        if dx > 0:
            sigma = u * dt / dx
            beta = D * dt / (dx * dx)
        else:
            sigma = 0.0
            beta = 0.0

        # Precompute base coefficients for transport
        # Arrays are zero-indexed here but map to 1-based arrays in VBA
        base_A = np.zeros(nc)
        base_B = np.zeros(nc)
        base_e = np.zeros(nc)
        base_f = np.zeros(nc)
        base_g = np.zeros(nc)
        # upwind counterparts for QUICK upwinding
        up_b = np.zeros(nc)
        up_e = np.zeros(nc)
        up_f = np.zeros(nc)

        # Populate diffusion contribution (Neumann zero-flux at boundaries)
        if self.config.diffusion_active and beta != 0.0:
            # left boundary
            base_B[0] = 0.0
            base_e[0] = -beta
            base_f[0] = beta
            base_g[0] = 0.0
            base_A[0] = 0.0
            # right boundary
            base_B[nc-1] = beta
            base_e[nc-1] = -beta
            base_f[nc-1] = 0.0
            base_g[nc-1] = 0.0
            base_A[nc-1] = 0.0
            # interior
            for i in range(1, nc-1):
                base_A[i] = 0.0
                base_B[i] = beta
                base_e[i] = -2.0 * beta
                base_f[i] = beta
                base_g[i] = 0.0

        # Populate advection contribution on top of diffusion
        if self.config.advection_active and sigma != 0.0:
            adv_type = self.config.advection_type.lower()
            # quick_up always considered for QUICK advection
            quick_up_ratio = self.config.quick_up_ratio
            # base upwind arrays for QUICK upwinding
            # Upwind only modifies B,e,f
            if u > 0:
                for i in range(nc):
                    up_b[i] = base_B[i] + sigma
                    up_e[i] = base_e[i] - sigma
                    up_f[i] = base_f[i]
            else:
                for i in range(nc):
                    up_b[i] = base_B[i]
                    up_e[i] = base_e[i] + sigma
                    up_f[i] = base_f[i] - sigma

            if adv_type == "upwind":
                # modify base_B, base_e, base_f for upwind
                if u > 0:
                    for i in range(nc):
                        base_B[i] += sigma
                        base_e[i] -= sigma
                else:
                    for i in range(nc):
                        base_e[i] += sigma
                        base_f[i] -= sigma
            elif adv_type == "central":
                # central differences; upwind at boundaries
                for i in range(1, nc-1):
                    base_B[i] += sigma / 2.0
                    base_f[i] -= sigma / 2.0
                # boundaries use upwind
                if u > 0:
                    base_B[0] += sigma
                    base_e[0] -= sigma
                    base_B[nc-1] += sigma
                    base_e[nc-1] -= sigma
                else:
                    base_e[0] += sigma
                    base_f[0] -= sigma
                    base_e[nc-1] += sigma
                    base_f[nc-1] -= sigma
            elif adv_type == "quick":
                # QUICK corrections: modifies base_A,B,e,f,g
                if u > 0:
                    # interior 1..n-2 (1-based 2..NC-1)
                    for i in range(1, nc-1):
                        base_A[i] -= (1.0/8.0) * sigma
                        base_B[i] += (6.0/8.0) * sigma
                        base_e[i] += (3.0/8.0) * sigma
                        base_B[i] += (1.0/8.0) * sigma
                        base_e[i] -= (6.0/8.0) * sigma
                        base_f[i] -= (3.0/8.0) * sigma
                    # left boundary (0 index)
                    base_B[0] += (9.0/8.0) * sigma
                    base_e[0] -= (6.0/8.0) * sigma
                    base_f[0] -= (3.0/8.0) * sigma
                    # right boundary (last)
                    base_A[nc-1] -= (1.0/8.0) * sigma
                    base_B[nc-1] += (6.0/8.0) * sigma
                    base_e[nc-1] -= (5.0/8.0) * sigma
                    # base_f[nc-1] already includes diffusion; advective right face term is zero per VBA
                else:
                    # u < 0
                    for i in range(1, nc-1):
                        base_B[i] += (3.0/8.0) * sigma
                        base_e[i] += (6.0/8.0) * sigma
                        base_f[i] -= (1.0/8.0) * sigma
                        base_e[i] -= (3.0/8.0) * sigma
                        base_f[i] -= (6.0/8.0) * sigma
                        base_g[i] += (1.0/8.0) * sigma
                    # left boundary
                    base_e[0] += (5.0/8.0) * sigma
                    base_f[0] -= (6.0/8.0) * sigma
                    base_g[0] += (1.0/8.0) * sigma
                    # right boundary
                    base_B[nc-1] += (3.0/8.0) * sigma
                    base_e[nc-1] += (6.0/8.0) * sigma
                    base_f[nc-1] -= (9.0/8.0) * sigma
                    # base_g[nc-1] remains 0
            else:
                # unsupported scheme: do nothing
                pass

        # Solve for each active constituent
        for name, prop in self.constituents.items():
            if not prop.active:
                continue
            # snapshot of old values
            C_old = prop.values.copy()
            # boundary values for this property based on type
            # fixed: given; zero gradient: equal to nearest cell; cyclic handled later
            left_val = None
            right_val = None
            if prop.bc_left_type == "Cyclic":
                # cyclic boundaries wrap around.  We'll handle this in the solver.
                left_val = None
            elif prop.bc_left_type == "Fixed":
                left_val = prop.bc_left_val
            else:  # ZeroGrad
                left_val = C_old[0]
            if prop.bc_right_type == "Cyclic":
                right_val = None
            elif prop.bc_right_type == "Fixed":
                right_val = prop.bc_right_val
            else:
                right_val = C_old[-1]
            # Determine solver type
            disc = self.config.time_discretisation.lower()
            adv_type = self.config.advection_type.lower()
            # For QUICK, we may use upwind in steep gradient regions
            quick_up = (adv_type == "quick")
            quick_ratio = self.config.quick_up_ratio
            # Build coefficient arrays for this property
            if adv_type == "quick":
                # Start with base arrays; copy to local arrays we can modify
                A = base_A.copy()
                B = base_B.copy()
                e = base_e.copy()
                f = base_f.copy()
                g = base_g.copy()
                # for QUICK upwind, b_up, e_up, f_up arrays computed above
                # Determine if upwind should be used at each interior cell based on gradient test
                quick_flags = np.zeros(nc, dtype=bool)
                # Evaluate gradients based on current values
                for i in range(1, nc-1):
                    # compute A,B,C as in VBA
                    # avoid index errors at extremes by clamping
                    # convert to 0-based; A-> |C_old[i] - C_old[i-1]| etc.
                    diff_prev = abs(C_old[i] - C_old[i-1])
                    diff_next = abs(C_old[i+1] - C_old[i])
                    diff_cross = abs(C_old[i+1] - C_old[i-1])
                    # ensure threshold uses (1 + minimum) to allow test when min=0
                    threshold = quick_ratio * (1.0 + prop.min_val)
                    A_val = max(diff_prev, threshold)
                    B_val = max(diff_next, threshold)
                    C_val = max(diff_cross, threshold)
                    if (abs(A_val - C_val) > quick_ratio * A_val) or (abs(B_val - C_val) > quick_ratio * B_val):
                        quick_flags[i] = True
                # modify coefficients for upwind where flagged
                for i in range(1, nc-1):
                    if quick_flags[i]:
                        # Upwind coefficients replace QUICK coefficients
                        A[i] = 0.0
                        B[i] = up_b[i]
                        e[i] = up_e[i]
                        f[i] = up_f[i]
                        g[i] = 0.0
                # Also handle first and last cells for upwind if flagged
                # (flags[0] and flags[nc-1] are unused since no gradient test on boundaries)
            else:
                A = np.zeros(nc)
                B = base_B.copy()
                e = base_e.copy()
                f = base_f.copy()
                g = np.zeros(nc)

            # Choose solver based on discretisation and advection type
            if disc == "exp":
                if adv_type == "quick":
                    new_vals = self._exp_quick_transport(C_old, A, B, e, f, g, left_val, right_val)
                else:
                    new_vals = self._exp_transport(C_old, B, e, f, left_val, right_val, prop)
            elif disc == "imp":
                if adv_type == "quick":
                    new_vals = self._imp_quick_transport(C_old, A, B, e, f, g, up_b, up_e, up_f, left_val, right_val)
                else:
                    new_vals = self._imp_transport(C_old, B, e, f, left_val, right_val, prop)
            else:  # semi-implicit
                if adv_type == "quick":
                    new_vals = self._semi_imp_quick_transport(C_old, A, B, e, f, g, up_b, up_e, up_f, left_val, right_val)
                else:
                    new_vals = self._semi_imp_transport(C_old, B, e, f, left_val, right_val, prop)
            # clamp to min/max and assign
            np.clip(new_vals, prop.min_val, prop.max_val, out=new_vals)
            prop.values = new_vals

    # ------------------------------------------------------------------
    # Transport solvers replicating the VBA routines
    #
    # Each of the following methods receives arrays of coefficients
    # (A, B, e, f, g) that correspond to the diffusion/advection
    # contribution.  The arrays are 0-indexed and correspond to
    # positions 0..NC-1.  left_val and right_val are the boundary
    # values (or None for cyclic boundaries).  C_old is the state
    # vector prior to the current step.  up_b, up_e, up_f are the
    # upwind coefficients used in QUICK implicit/semi-implicit solvers.
    #
    # These solvers return a new numpy array of length NC with the
    # updated values.

    def _exp_transport(self, C_old, B, e, f, left_val, right_val, prop):
        nc = len(C_old)
        new_vals = np.zeros(nc)
        # handle cyclic boundaries by wrapping C_old
        cyclic = (prop.bc_left_type == "Cyclic" or prop.bc_right_type == "Cyclic")
        for i in range(nc):
            # compute neighbours indices or use boundary values
            if i == 0:
                c_left = C_old[nc-1] if prop.bc_left_type == "Cyclic" else left_val
                c_right = C_old[i+1] if nc > 1 else (C_old[0] if prop.bc_right_type == "Cyclic" else right_val)
            elif i == nc-1:
                c_left = C_old[i-1]
                c_right = C_old[0] if prop.bc_right_type == "Cyclic" else right_val
            else:
                c_left = C_old[i-1]
                c_right = C_old[i+1]
            # update
            new_vals[i] = B[i] * c_left + (1.0 + e[i]) * C_old[i] + f[i] * c_right
        return new_vals

    def _exp_quick_transport(self, C_old, A, B, e, f, g, left_val, right_val):
        nc = len(C_old)
        new_vals = np.zeros(nc)
        for i in range(nc):
            if i == 0:
                # cell 1: uses left boundary, cell0, cell1, cell2
                c_left = left_val
                c0 = C_old[0]
                c1 = C_old[1] if nc > 1 else C_old[0]
                c2 = C_old[2] if nc > 2 else c1
                new_vals[0] = B[0] * c_left + (1.0 + e[0]) * c0 + f[0] * c1 + g[0] * c2
            elif i == 1:
                c_ll = left_val
                c_l = C_old[0]
                c0 = C_old[1]
                c1 = C_old[2] if nc > 2 else C_old[1]
                c2 = C_old[3] if nc > 3 else c1
                new_vals[1] = A[1] * c_ll + B[1] * c_l + (1.0 + e[1]) * c0 + f[1] * c1 + g[1] * c2
            elif i == nc - 2:
                # second last
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = C_old[i+1]
                c2 = right_val
                new_vals[i] = A[i] * c_ll + B[i] * c_l + (1.0 + e[i]) * c0 + f[i] * c1 + g[i] * c2
            elif i == nc - 1:
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = right_val
                c2 = 0.0
                new_vals[i] = A[i] * c_ll + B[i] * c_l + (1.0 + e[i]) * c0 + f[i] * c1 + g[i] * c2
            else:
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = C_old[i+1]
                c2 = C_old[i+2]
                new_vals[i] = A[i] * c_ll + B[i] * c_l + (1.0 + e[i]) * c0 + f[i] * c1 + g[i] * c2
        return new_vals

    def _imp_transport(self, C_old, B, e, f, left_val, right_val, prop):
        """
        Fully implicit tridiagonal solver for diffusion/advection (non-QUICK).

        This corresponds to ImpTransport in the VBA code.  Coefficients
        B, e, f are used to form a tridiagonal matrix (A is zero).
        """
        nc = len(C_old)
        # Initialize tridiagonal arrays
        a = -B.copy()
        b = 1.0 - e.copy()
        c = -f.copy()
        # Right-hand side is just the old values
        rhs = C_old.copy()
        # Apply boundary conditions to RHS
        # left boundary
        if prop.bc_left_type == "Fixed":
            rhs[0] -= a[0] * left_val
        # right boundary
        if prop.bc_right_type == "Fixed":
            rhs[nc-1] -= c[nc-1] * right_val
        # Solve tridiagonal system via Thomas algorithm
        cp = np.zeros(nc)
        dp = np.zeros(nc)
        if b[0] == 0.0:
            b[0] = 1e-15
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        for i in range(1, nc):
            denom = b[i] - a[i] * cp[i-1]
            if denom == 0.0:
                denom = 1e-15
            cp[i] = c[i] / denom
            dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom
        new_vals = np.zeros(nc)
        new_vals[nc-1] = dp[nc-1]
        for i in range(nc - 2, -1, -1):
            new_vals[i] = dp[i] - cp[i] * new_vals[i+1]
        return new_vals

    def _imp_quick_transport(self, C_old, A, B, e, f, g, up_b, up_e, up_f, left_val, right_val):
        """
        Fully implicit QUICK solver using a five-diagonal scheme.  This
        implementation follows the VBA ImpQUICKTransport routine.  It
        eliminates the lower secondary diagonal and solves the resulting
        tridiagonal system via Thomas algorithm.
        """
        nc = len(C_old)
        # Allocate coefficient arrays
        L_left = -A.copy()
        Left = -B.copy()
        Center = 1.0 - e.copy()
        Right = -f.copy()
        R_right = -g.copy()
        rhs = C_old.copy()
        # upwind replacements are embedded in A,B,e,f,g already when flagged
        # Apply boundary values to rhs (left and right) as in VBA
        if left_val is not None:
            rhs[0] -= Left[0] * left_val
            rhs[1] -= L_left[1] * left_val
        if right_val is not None:
            rhs[nc-1] -= Right[nc-1] * right_val
            rhs[nc-2] -= R_right[nc-2] * right_val
        # Forward elimination
        for i in range(2, nc):
            # eliminate L_left[i]
            f1 = Left[i-1] / Center[i-2]
            Left[i-1] = Left[i-1] - f1 * Center[i-2]
            Center[i-1] = Center[i-1] - f1 * Right[i-2]
            Right[i-1] = Right[i-1] - f1 * R_right[i-2]
            rhs[i-1] = rhs[i-1] - f1 * rhs[i-2]
            f2 = L_left[i] / Center[i-2]
            L_left[i] = L_left[i] - f2 * Center[i-2]
            Left[i] = Left[i] - f2 * Right[i-2]
            Center[i] = Center[i] - f2 * R_right[i-2]
            rhs[i] = rhs[i] - f2 * rhs[i-2]
        # Handle last equation elimination
        f_last = Left[nc-1] / Center[nc-2]
        Left[nc-1] = Left[nc-1] - f_last * Center[nc-2]
        Center[nc-1] = Center[nc-1] - f_last * Right[nc-2]
        Right[nc-1] = Right[nc-1] - f_last * R_right[nc-2]
        rhs[nc-1] = rhs[nc-1] - f_last * rhs[nc-2]
        # Back substitution
        new_vals = np.zeros(nc)
        new_vals[nc-1] = rhs[nc-1] / Center[nc-1]
        new_vals[nc-2] = (rhs[nc-2] - Right[nc-2] * new_vals[nc-1]) / Center[nc-2]
        for k in range(2, nc):
            i = nc - 1 - k
            new_vals[i] = (rhs[i] - Right[i] * new_vals[i+1] - R_right[i] * new_vals[i+2]) / Center[i]
        return new_vals

    def _semi_imp_transport(self, C_old, B, e, f, left_val, right_val, prop):
        """
        Semi-implicit solver for non-QUICK advection/diffusion.  It uses
        Crank–Nicolson-like averaging on the RHS.  This mirrors the
        semiImpTransport routine in VBA.
        """
        nc = len(C_old)
        # compute coefficients
        a = -B.copy() / 2.0
        b = 1.0 - e.copy() / 2.0
        c = -f.copy() / 2.0
        rhs = np.zeros(nc)
        # Build RHS = C_old + 0.5 * (B * left + e * C_old + f * right)
        for i in range(nc):
            if i == 0:
                c_left = C_old[nc-1] if prop.bc_left_type == "Cyclic" else left_val
                c_right = C_old[i+1] if nc > 1 else (C_old[0] if prop.bc_right_type == "Cyclic" else right_val)
            elif i == nc-1:
                c_left = C_old[i-1]
                c_right = C_old[0] if prop.bc_right_type == "Cyclic" else right_val
            else:
                c_left = C_old[i-1]
                c_right = C_old[i+1]
            rhs[i] = C_old[i] + 0.5 * (B[i] * c_left + e[i] * C_old[i] + f[i] * c_right)
        # Apply boundary terms to rhs for fixed BC
        if prop.bc_left_type == "Fixed":
            rhs[0] -= a[0] * left_val
        if prop.bc_right_type == "Fixed":
            rhs[nc-1] -= c[nc-1] * right_val
        # Solve tridiagonal system
        cp = np.zeros(nc)
        dp = np.zeros(nc)
        if b[0] == 0.0:
            b[0] = 1e-15
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        for i in range(1, nc):
            denom = b[i] - a[i] * cp[i-1]
            if denom == 0.0:
                denom = 1e-15
            cp[i] = c[i] / denom
            dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom
        new_vals = np.zeros(nc)
        new_vals[nc-1] = dp[nc-1]
        for i in range(nc-2, -1, -1):
            new_vals[i] = dp[i] - cp[i] * new_vals[i+1]
        return new_vals

    def _semi_imp_quick_transport(self, C_old, A, B, e, f, g, up_b, up_e, up_f, left_val, right_val):
        """
        Semi-implicit QUICK solver.  This mirrors SemiImpQUICKTransport in
        the VBA code, using half-step contributions on both sides of
        the equation.
        """
        nc = len(C_old)
        # Initialize coefficient arrays for semi-implicit: divide QUICK coeffs by 2
        L_left = -A.copy() / 2.0
        Left = -B.copy() / 2.0
        Center = 1.0 - e.copy() / 2.0
        Right = -f.copy() / 2.0
        R_right = -g.copy() / 2.0
        rhs = C_old.copy()
        # Compute RHS modifications: add 0.5*(A,B,e,f,g)*Old values
        for i in range(nc):
            if i == 0:
                c_ll = left_val
                c_l = C_old[0]
                c0 = C_old[0]
                c1 = C_old[1] if nc > 1 else C_old[0]
                c2 = C_old[2] if nc > 2 else c1
            elif i == 1:
                c_ll = left_val
                c_l = C_old[0]
                c0 = C_old[1]
                c1 = C_old[2] if nc > 2 else C_old[1]
                c2 = C_old[3] if nc > 3 else c1
            elif i == nc - 2:
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = C_old[i+1]
                c2 = right_val
            elif i == nc - 1:
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = right_val
                c2 = 0.0
            else:
                c_ll = C_old[i-2]
                c_l = C_old[i-1]
                c0 = C_old[i]
                c1 = C_old[i+1]
                c2 = C_old[i+2]
            rhs[i] += 0.5 * (A[i] * c_ll + B[i] * c_l + e[i] * c0 + f[i] * c1 + g[i] * c2)
        # Apply boundary corrections for semi-implicit quick
        if left_val is not None:
            rhs[0] -= 2.0 * Left[0] * left_val + (Center[0] - 1.0) * C_old[0] + Right[0] * C_old[1] + R_right[0] * (C_old[2] if nc > 2 else C_old[1])
            rhs[1] -= 2.0 * L_left[1] * left_val + Left[1] * C_old[0] + (Center[1] - 1.0) * C_old[1] + Right[1] * C_old[2] + R_right[1] * (C_old[3] if nc > 3 else C_old[2])
        if right_val is not None:
            rhs[nc-1] -= Left[nc-1] * C_old[nc-2] + (Center[nc-1] - 1.0) * C_old[nc-1] + 2.0 * Right[nc-1] * right_val
            rhs[nc-2] -= L_left[nc-2] * C_old[nc-4] + Left[nc-2] * C_old[nc-3] + (Center[nc-2] - 1.0) * C_old[nc-2] + Right[nc-2] * C_old[nc-1] + 2.0 * R_right[nc-2] * right_val
        # Forward elimination (similar to implicit quick)
        for i in range(2, nc):
            f1 = Left[i-1] / Center[i-2]
            Left[i-1] = Left[i-1] - f1 * Center[i-2]
            Center[i-1] = Center[i-1] - f1 * Right[i-2]
            Right[i-1] = Right[i-1] - f1 * R_right[i-2]
            rhs[i-1] = rhs[i-1] - f1 * rhs[i-2]
            f2 = L_left[i] / Center[i-2]
            L_left[i] = L_left[i] - f2 * Center[i-2]
            Left[i] = Left[i] - f2 * Right[i-2]
            Center[i] = Center[i] - f2 * R_right[i-2]
            rhs[i] = rhs[i] - f2 * rhs[i-2]
        f_last = Left[nc-1] / Center[nc-2]
        Left[nc-1] = Left[nc-1] - f_last * Center[nc-2]
        Center[nc-1] = Center[nc-1] - f_last * Right[nc-2]
        Right[nc-1] = Right[nc-1] - f_last * R_right[nc-2]
        rhs[nc-1] = rhs[nc-1] - f_last * rhs[nc-2]
        # Back substitution
        new_vals = np.zeros(nc)
        new_vals[nc-1] = rhs[nc-1] / Center[nc-1]
        new_vals[nc-2] = (rhs[nc-2] - Right[nc-2] * new_vals[nc-1]) / Center[nc-2]
        for k in range(2, nc):
            i = nc - 1 - k
            new_vals[i] = (rhs[i] - Right[i] * new_vals[i+1] - R_right[i] * new_vals[i+2]) / Center[i]
        return new_vals


    def apply_kinetics(self, dt):
        temp = self.constituents["Temperature"].values if "Temperature" in self.constituents else np.full(self.grid.nc, 20.0)
        bod_prop = self.constituents.get("BOD")
        do_prop = self.constituents.get("DO")
        co2_prop = self.constituents.get("CO2")
        gen_prop = self.constituents.get("Generic")
        
        # 1. Temperature: apply free-surface fluxes sequentially as in VBA
        if "Temperature" in self.constituents:
            p = self.constituents["Temperature"]
            # physical constants for conversion (kg/m³·J/kg·K)
            rho_cp = 1000.0 * 4180.0
            depth = self.grid.water_depth if self.grid.water_depth > 0 else 1.0
            for i in range(self.grid.nc):
                T = p.values[i]
                # compute heat fluxes at current time (self.current_time references
                # the start of the current step; the VBA routine evaluates
                # radiation etc. using ctrl.timedays which is updated
                # immediately before transport).  We therefore use
                # self.current_time + dt here to match the VBA ordering where
                # dt is added before fluxes are applied.
                flux = self.calculate_heat_fluxes(T, self.current_time + dt)
                # apply solar
                if p.use_surface_flux:
                    dT = flux["solar"] * dt / (rho_cp * depth)
                    p.values[i] += dT
                    T = p.values[i]
                # sensible heat
                if p.use_sensible_heat:
                    dT = flux["sensible"] * dt / (rho_cp * depth)
                    p.values[i] += dT
                    T = p.values[i]
                # latent heat
                if p.use_latent_heat:
                    dT = flux["latent"] * dt / (rho_cp * depth)
                    p.values[i] += dT
                    T = p.values[i]
                # radiative heat
                if p.use_radiative_heat:
                    dT = flux["radiative"] * dt / (rho_cp * depth)
                    p.values[i] += dT

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
        
        # VBA Formula: K = (1/H) * 0.142 * (|W|+0.1) * (S + 1e-5) [1/s]
        k_reaeration = (1.0/h) * 0.142 * (abs(wind) + 0.1) * (slope + 1e-5)
        
        for p in [do_prop, co2_prop]:
            if p and p.use_surface_flux:
                gas = "O2" if p.name=="DO" else "CO2"
                for i in range(self.grid.nc):
                    cs = self.calculate_saturation(temp[i], gas)
                    # Implicit linear update for stability: (C + dt*K*Cs)/(1+dt*K)
                    p.values[i] = (p.values[i] + dt * k_reaeration * cs) / (1.0 + dt * k_reaeration)

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
        """
        Run the simulation and record results at specified print intervals.

        The original VBA code prints results whenever the simulation time
        exceeds the next print time.  It does not include an explicit
        initial output at t=0 in the spreadsheet, but for clarity and
        easier comparison we include the initial state as the first
        entry.  Subsequent outputs are recorded whenever
        current_time >= next_print_time (with next_print_time
        incremented by dt_print after each print).
        """
        total_steps = int((self.config.duration_days * 86400.0) / self.config.dt)
        # Ensure there is at least one step
        if total_steps < 1:
            total_steps = 1
        # Initialise results
        self.results["times"] = []
        for name in self.constituents:
            self.results[name] = []
        # Record initial state at t=0
        self.results["times"].append(0.0)
        for name, prop in self.constituents.items():
            self.results[name].append(prop.values.copy())
        # Determine next print time in seconds
        next_print = self.config.dt_print
        # Time stepping loop
        for step_n in range(1, total_steps + 1):
            self.step()
            # If current_time exceeds next_print, record and increment
            if self.current_time >= next_print - 1e-9:
                self.results["times"].append(self.current_time / 86400.0)
                for name, prop in self.constituents.items():
                    self.results[name].append(prop.values.copy())
                # advance next_print by multiples of dt_print (catching up if dt > dt_print)
                # ensure we do not accumulate floating point drift
                while self.current_time >= next_print - 1e-9:
                    next_print += self.config.dt_print
            yield step_n / total_steps
            
    def run(self, callback=None):
        runner = self.run_simulation()
        for progress in runner:
            if callback: callback(progress)
        return self.results
