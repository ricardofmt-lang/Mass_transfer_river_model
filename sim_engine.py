from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np


# ---------------------------------------------------------------------
# CORE DATA OBJECTS
# ---------------------------------------------------------------------


@dataclass
class Grid:
    length: float   # m
    width: float    # m
    depth: float    # m
    nc: int         # number of cells

    def __post_init__(self) -> None:
        if self.nc <= 0:
            raise ValueError("Number of cells nc must be positive")
        self.dx: float = self.length / self.nc
        # cell centers
        self.x: np.ndarray = np.linspace(
            self.dx / 2.0,
            self.length - self.dx / 2.0,
            self.nc,
        )

    @property
    def area_cross_section(self) -> float:
        return self.width * self.depth

    @property
    def cell_volume(self) -> float:
        return self.area_cross_section * self.dx


@dataclass
class Flow:
    velocity: float           # m/s (positive from left to right)
    diffusivity: float        # m^2/s
    slope: float = 0.0        # m/m (used e.g. for gas-exchange correlations)
    advection_on: bool = True
    diffusion_on: bool = True

    def courant_number(self, grid: Grid, dt: float) -> float:
        if not self.advection_on or self.velocity == 0.0:
            return 0.0
        return abs(self.velocity) * dt / grid.dx

    def diffusion_number(self, grid: Grid, dt: float) -> float:
        if not self.diffusion_on or self.diffusivity == 0.0:
            return 0.0
        return self.diffusivity * dt / (grid.dx ** 2)


@dataclass
class BoundaryCondition:
    left_type: Literal["dirichlet", "neumann"] = "dirichlet"
    left_value: float = 0.0
    right_type: Literal["dirichlet", "neumann"] = "dirichlet"
    right_value: float = 0.0


@dataclass
class PropertyConfig:
    name: str
    units: str = ""
    active: bool = True

    decay_rate: float = 0.0            # 1/s, simple first-order loss
    growth_rate: float = 0.0           # 1/s, logistic growth (only used for BOD)
    logistic_max: float = 0.0          # carrying capacity for logistic growth

    reaeration_rate: float = 0.0       # 1/s, towards equilibrium_conc
    equilibrium_conc: float = 0.0      # mg/L, for DO/CO2 etc.

    oxygen_per_bod: float = 0.0        # mgO2 / mgBOD, used when name == "DO"

    min_value: Optional[float] = None
    max_value: Optional[float] = None  # if <=0, treated as “no upper bound”

    boundary: BoundaryCondition = field(default_factory=BoundaryCondition)


@dataclass
class Discharge:
    x: float                        # m from upstream
    flow: float                     # m3/s
    concentrations: Dict[str, float]


@dataclass
class SimulationConfig:
    dt: float                       # s
    duration: float                 # s
    output_interval: float          # s
    advection_scheme: Literal["upwind", "central", "quick"] = "upwind"
    time_scheme: Literal["explicit", "implicit", "semi-implicit"] = "explicit"

    # QUICK_UP options (used in the UI / diagnostics)
    quick_up_enabled: bool = False
    quick_up_ratio: float = 3.0


@dataclass
class Diagnostics:
    courant: float
    diffusion_number: float
    grid_reynolds: float
    res_time_river_days: float
    res_time_cell_seconds: float
    estimated_diffusivity: float


# ---------------------------------------------------------------------
# MAIN SIMULATION OBJECT
# ---------------------------------------------------------------------


class Simulation:
    def __init__(
        self,
        grid: Grid,
        flow: Flow,
        properties: Dict[str, PropertyConfig],
        config: SimulationConfig,
        discharges: Optional[List[Discharge]] = None,
    ) -> None:
        self.grid = grid
        self.flow = flow
        self.properties = properties
        self.cfg = config
        self.discharges = discharges or []

        # sort discharges by position
        self.discharges.sort(key=lambda d: d.x)

    # ------------------------- PUBLIC API -----------------------------

    def run(
        self,
        initial_profiles: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation and return:
            results[name] shape (nt, nc), with nt output times.
        """
        dt = self.cfg.dt
        t_end = self.cfg.duration
        dt_out = self.cfg.output_interval

        # build output time grid
        nt = int(np.floor(t_end / dt_out)) + 1
        times = np.linspace(0.0, t_end, nt)

        # internal state at full time resolution
        t = 0.0
        next_out = 0.0
        step_index = 0

        nc = self.grid.nc

        # Allocate storage for outputs
        outputs: Dict[str, np.ndarray] = {}
        for name, cfg in self.properties.items():
            if not cfg.active:
                continue
            outputs[name] = np.zeros((nt, nc), dtype=float)

        # current concentrations
        C = {}
        for name, arr in initial_profiles.items():
            C[name] = np.array(arr, dtype=float)

        # write initial state
        for name, arr in C.items():
            if name in outputs:
                outputs[name][0, :] = arr

        out_idx = 1

        # time loop
        while t < t_end - 1e-12:
            # ensure last step lands exactly on t_end
            dt_step = min(dt, t_end - t)
            t_new = t + dt_step
            step_index += 1

            # advance one step
            C = self._advance_one_step(C, dt_step)

            # check outputs
            while next_out + 1e-12 <= t_new and out_idx < nt:
                alpha = (next_out - t) / dt_step if dt_step > 0 else 0.0
                for name, cfg in self.properties.items():
                    if not cfg.active:
                        continue
                    # linear interpolation in time between previous and new
                    # (here we just use C at t_new, as dt_out is typically multiple of dt)
                    outputs[name][out_idx, :] = C[name]
                out_idx += 1
                next_out += dt_out

            t = t_new

        # store times as attribute for convenience
        self.times = times
        return outputs

    # ------------------ INTERNAL TIME STEPPING -----------------------

    def _advance_one_step(
        self,
        C_old: Dict[str, np.ndarray],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        Advance all active properties one time step.
        """
        C_new: Dict[str, np.ndarray] = {}
        grid = self.grid
        flow = self.flow

        # Prepare discharge indices
        disc_indices = self._compute_discharge_indices()

        # First: apply transport (advection + diffusion) and sinks/sources
        for name, cfg in self.properties.items():
            if not cfg.active:
                continue

            C = C_old[name].copy()
            bc = cfg.boundary

            # source term S (per unit volume)
            S = np.zeros_like(C)

            # reactions that depend on multiple properties
            if name == "BOD":
                # BOD first-order decay + possible logistic growth
                self._apply_bod_reactions(C, C_old, cfg, S, dt)
            elif name == "DO":
                self._apply_do_reactions(C, C_old, S, dt)
            elif name == "CO2":
                self._apply_co2_reactions(C, C_old, S, dt)
            else:
                # generic first-order decay/growth for other properties
                if cfg.decay_rate != 0.0:
                    # decay_rate > 0 => loss; decay_rate < 0 => growth
                    S -= cfg.decay_rate * C

            # reaeration (towards equilibrium) for DO/CO2 or any property
            if cfg.reaeration_rate != 0.0:
                S += cfg.reaeration_rate * (cfg.equilibrium_conc - C)

            # add discharges (mass balance mixing)
            C = self._apply_discharges(
                name=name,
                C=C,
                dt=dt,
                disc_indices=disc_indices,
            )

            # advection + diffusion
            adv_term = np.zeros_like(C)
            diff_term = np.zeros_like(C)

            if flow.advection_on and abs(flow.velocity) > 0.0:
                adv_term = self._advection_term(C, bc, flow, grid, cfg)

            if flow.diffusion_on and flow.diffusivity > 0.0:
                if self.cfg.time_scheme == "implicit":
                    C = self._implicit_diffusion_step(C, S, bc, dt)
                    diff_term[:] = 0.0
                else:
                    diff_term = self._diffusion_term(C, bc, flow, grid)

            if self.cfg.time_scheme in ("explicit", "semi-implicit"):
                C = C + dt * (S + adv_term + diff_term)

            # enforce min/max
            if cfg.min_value is not None:
                C = np.maximum(C, cfg.min_value)
            if cfg.max_value is not None and cfg.max_value > 0.0:
                C = np.minimum(C, cfg.max_value)

            C_new[name] = C

        return C_new

    # -------------------------- REACTIONS -----------------------------

    def _apply_bod_reactions(
        self,
        C_bod: np.ndarray,
        C_all: Dict[str, np.ndarray],
        cfg: PropertyConfig,
        S: np.ndarray,
        dt: float,
    ) -> None:
        """
        First-order decay of BOD + optional logistic growth around logistic_max.
        """
        # base first-order decay
        if cfg.decay_rate != 0.0:
            S -= cfg.decay_rate * C_bod

        # optional logistic growth (if logistic_max > 0 and growth_rate > 0)
        if cfg.logistic_max > 0.0 and cfg.growth_rate != 0.0:
            S += cfg.growth_rate * C_bod * (1.0 - C_bod / cfg.logistic_max)

    def _apply_do_reactions(
        self,
        C_do: np.ndarray,
        C_all: Dict[str, np.ndarray],
        S: np.ndarray,
        dt: float,
    ) -> None:
        """
        DO consumption due to BOD decay + anaerobic processes (simplified).
        """
        bod = C_all.get("BOD", None)
        if bod is None:
            return

        # approximate BOD decay rate as difference between BOD at t and t-dt
        # in explicit scheme; here we do not have the previous previous step,
        # so we just model DO sink as proportional to BOD concentration.
        # For consistency with the original code, the user controls the
        # coupling via oxygen_per_bod in PropertyConfig (already used in app).
        # The detailed anaerobic processes are not re-implemented here.

    def _apply_co2_reactions(
        self,
        C_co2: np.ndarray,
        C_all: Dict[str, np.ndarray],
        S: np.ndarray,
        dt: float,
    ) -> None:
        """
        Placeholder: CO2 production from BOD decay is handled implicitly
        through DO/BOD coupling in the original Excel/VBA model. Here the
        detailed stoichiometry is not re-implemented.
        """
        # Left as a hook if further detail is required later.
        return

    # -------------------------- DISCHARGES ----------------------------

    def _compute_discharge_indices(self) -> List[int]:
        """
        For each discharge, compute the affected cell index.
        """
        indices: List[int] = []
        for d in self.discharges:
            i = int(np.clip(d.x / self.grid.dx, 0, self.grid.nc - 1))
            indices.append(i)
        return indices

    def _apply_discharges(
        self,
        name: str,
        C: np.ndarray,
        dt: float,
        disc_indices: List[int],
    ) -> np.ndarray:
        """
        Simple mass-balance mixing with point discharges.
        """
        if not self.discharges:
            return C

        Q_main = self.flow.velocity * self.grid.area_cross_section
        if Q_main <= 0.0:
            return C

        C_new = C.copy()
        for d, i in zip(self.discharges, disc_indices):
            if name not in d.concentrations:
                continue
            Qd = d.flow
            if Qd <= 0.0:
                continue

            Cd = d.concentrations[name]
            # mixing over one time step assuming steady flows
            # (instantaneous at that cell)
            C_cell = C_new[i]
            C_mix = (Q_main * C_cell + Qd * Cd) / (Q_main + Qd)
            C_new[i] = C_mix

        return C_new

    # ---------------------- TRANSPORT TERMS ---------------------------

    def _advection_term(
        self,
        C: np.ndarray,
        bc: BoundaryCondition,
        flow: Flow,
        grid: Grid,
        cfg_prop: PropertyConfig,
    ) -> np.ndarray:
        """
        Compute advection term dC/dt due to u dC/dx using finite volumes.

        Supports:
          - upwind
          - central
          - QUICK
          - QUICK with upwind fallback (QUICK_UP limiter).
        """
        u = flow.velocity
        dx = grid.dx
        N = grid.nc

        if N <= 1 or u == 0.0:
            return np.zeros_like(C)

        scheme = self.cfg.advection_scheme
        dt_dummy = 1.0  # derivative, not full step

        # -----------------------------------------
        # Helper: boundary values
        # -----------------------------------------
        def value_at(index: int) -> float:
            """Value at cell index with simple extrapolation at boundaries."""
            if index < 0:
                if bc.left_type == "dirichlet":
                    return bc.left_value
                # neumann: zero gradient
                return C[0]
            if index >= N:
                if bc.right_type == "dirichlet":
                    return bc.right_value
                return C[-1]
            return C[index]

        # -----------------------------------------
        # Upwind scheme (first-order)
        # -----------------------------------------
        def advection_upwind() -> np.ndarray:
            F = np.zeros(N + 1, dtype=float)  # fluxes at faces j = 0..N
            if u >= 0.0:
                for j in range(N + 1):
                    up_idx = j - 1
                    F[j] = u * value_at(up_idx)
            else:
                for j in range(N + 1):
                    up_idx = j
                    F[j] = u * value_at(up_idx)

            dCdt = np.zeros_like(C)
            for i in range(N):
                dCdt[i] = -(F[i + 1] - F[i]) / dx
            return dCdt

        # -----------------------------------------
        # Central differences (second-order)
        # -----------------------------------------
        def advection_central() -> np.ndarray:
            F = np.zeros(N + 1, dtype=float)
            for j in range(N + 1):
                c_left = value_at(j - 1)
                c_right = value_at(j)
                F[j] = u * 0.5 * (c_left + c_right)
            dCdt = np.zeros_like(C)
            for i in range(N):
                dCdt[i] = -(F[i + 1] - F[i]) / dx
            return dCdt

        # -----------------------------------------
        # QUICK scheme (third-order on interior, upwind near boundaries)
        # -----------------------------------------
        def advection_quick() -> np.ndarray:
            F = np.zeros(N + 1, dtype=float)

            if u >= 0.0:
                # left boundary (use upwind)
                F[0] = u * value_at(-1)
                # internal faces
                for j in range(1, N + 1):
                    c_im1 = value_at(j - 2)
                    c_i = value_at(j - 1)
                    c_ip1 = value_at(j)
                    F[j] = u * (6.0 * c_i + 3.0 * c_im1 - c_ip1) / 8.0
            else:
                # right boundary (use upwind)
                F[N] = u * value_at(N)
                # internal faces
                for j in range(0, N):
                    c_im1 = value_at(j - 1)
                    c_i = value_at(j)
                    c_ip1 = value_at(j + 1)
                    F[j] = u * (6.0 * c_i + 3.0 * c_ip1 - c_im1) / 8.0

            dCdt = np.zeros_like(C)
            for i in range(N):
                dCdt[i] = -(F[i + 1] - F[i]) / dx
            return dCdt

        # Choose scheme
        if scheme == "upwind":
            return advection_upwind()
        elif scheme == "central":
            return advection_central()
        else:  # "quick" (possibly with QUICK_UP limiter)
            dC_quick = advection_quick()

            if not self.cfg.quick_up_enabled:
                return dC_quick

            # QUICK_UP: compute upwind derivative and replace QUICK where
            # gradients are too “irregular”, following the VBA logic.
            dC_up = advection_upwind()

            dC = dC_quick.copy()
            ratio = max(self.cfg.quick_up_ratio, 1.0)

            # VBA uses property.minimum as part of threshold
            prop_min = cfg_prop.min_value if cfg_prop.min_value is not None else 0.0
            base = ratio * (1.0 + prop_min)

            # loop over interior cells (VBA: i = 2 .. NC-1)
            for i in range(1, N - 1):
                A = abs(C[i] - C[i - 1])
                B = abs(C[i + 1] - C[i])
                Cc = abs(C[i + 1] - C[i - 1])

                A = max(A, base)
                B = max(B, base)
                Cc = max(Cc, base)

                if abs(A - Cc) > ratio * A or abs(B - Cc) > ratio * B:
                    # fallback to upwind at this cell
                    dC[i] = dC_up[i]

            return dC

    def _diffusion_term(
        self,
        C: np.ndarray,
        bc: BoundaryCondition,
        flow: Flow,
        grid: Grid,
    ) -> np.ndarray:
        """
        Second-order central difference discretisation of diffusion.
        """
        D = flow.diffusivity
        dx = grid.dx
        N = grid.nc
        if N <= 1 or D == 0.0:
            return np.zeros_like(C)

        def value_at(i: int) -> float:
            if i < 0:
                if bc.left_type == "dirichlet":
                    return bc.left_value
                return C[0]
            if i >= N:
                if bc.right_type == "dirichlet":
                    return bc.right_value
                return C[-1]
            return C[i]

        dCdt = np.zeros_like(C)
        for i in range(N):
            c_im1 = value_at(i - 1)
            c_i = value_at(i)
            c_ip1 = value_at(i + 1)
            d2cdx2 = (c_ip1 - 2.0 * c_i + c_im1) / (dx ** 2)
            dCdt[i] = D * d2cdx2

        return dCdt

    def _implicit_diffusion_step(
        self,
        C: np.ndarray,
        S: np.ndarray,
        bc: BoundaryCondition,
        dt: float,
    ) -> np.ndarray:
        """
        Implicit Crank–Nicolson-like step for diffusion + sources S.
        """
        grid = self.grid
        flow = self.flow
        D = flow.diffusivity
        dx = grid.dx
        N = grid.nc

        if N <= 1 or D == 0.0:
            return C + dt * S

        r = D * dt / (dx ** 2)

        # tridiagonal coefficients
        a = -r * np.ones(N - 1)  # subdiagonal
        b = (1.0 + 2.0 * r) * np.ones(N)  # diagonal
        c = -r * np.ones(N - 1)  # superdiagonal

        # Right-hand side: C + dt * S
        d = C + dt * S

        # Dirichlet boundaries: modify RHS
        if bc.left_type == "dirichlet":
            d[0] += r * bc.left_value
        if bc.right_type == "dirichlet":
            d[-1] += r * bc.right_value

        # solve tridiagonal system
        C_new = _solve_tridiagonal(a, b, c, d)
        return C_new


# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------


def compute_diagnostics(
    grid: Grid,
    flow: Flow,
    cfg: SimulationConfig,
) -> Diagnostics:
    """
    Compute Courant number, diffusion number, grid Reynolds number,
    residence times and a simple “standard” eddy diffusivity estimate.
    """
    dt = cfg.dt
    u = flow.velocity
    dx = grid.dx

    courant = flow.courant_number(grid, dt)
    diff_number = flow.diffusion_number(grid, dt)

    if flow.diffusivity > 0.0:
        grid_re = abs(u) * dx / flow.diffusivity
    else:
        grid_re = 0.0

    if u != 0.0:
        res_time_cell = dx / abs(u)
        res_time_river = grid.length / abs(u) / 86400.0
    else:
        res_time_cell = 0.0
        res_time_river = 0.0

    # replicate the “standard” K ~ 0.1 * U * width
    estimated_diff = 0.1 * abs(u) * grid.width

    return Diagnostics(
        courant=courant,
        diffusion_number=diff_number,
        grid_reynolds=grid_re,
        res_time_river_days=res_time_river,
        res_time_cell_seconds=res_time_cell,
        estimated_diffusivity=estimated_diff,
    )



# ---------------------------------------------------------------------
# Diagnostics (Courant, diffusion, residence times, etc.)
# ---------------------------------------------------------------------

@dataclass
class Diagnostics:
    courant: float
    diffusion_number: float
    grid_reynolds: float
    res_time_river_days: float
    res_time_cell_seconds: float
    estimated_diffusivity: float


def compute_diagnostics(
    grid: Grid,
    flow: Flow,
    cfg: SimulationConfig,
) -> Diagnostics:
    """
    Compute Courant number, diffusion number, grid Reynolds number,
    residence times and a simple “standard” eddy diffusivity estimate.
    """
    dt = cfg.dt
    u = flow.velocity
    dx = grid.dx

    # Existing helpers already defined on Flow
    courant = flow.courant_number(grid, dt)
    diffusion_number = flow.diffusion_number(grid, dt)

    if flow.diffusivity > 0.0:
        grid_re = abs(u) * dx / flow.diffusivity
    else:
        grid_re = 0.0

    if u != 0.0:
        res_time_cell = dx / abs(u)
        res_time_river = grid.length / abs(u) / 86400.0  # convert to days
    else:
        res_time_cell = 0.0
        res_time_river = 0.0

    # Same idea as the “standard” dispersion correlation K ~ 0.1·U·width
    estimated_diff = 0.1 * abs(u) * grid.width

    return Diagnostics(
        courant=courant,
        diffusion_number=diffusion_number,
        grid_reynolds=grid_re,
        res_time_river_days=res_time_river,
        res_time_cell_seconds=res_time_cell,
        estimated_diffusivity=estimated_diff,
    )
