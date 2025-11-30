from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np


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
        self.x: np.ndarray = np.linspace(self.dx / 2.0,
                                         self.length - self.dx / 2.0,
                                         self.nc)

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
    growth_rate: float = 0.0           # 1/s, logistic growth
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


class Simulation:
    # 1D advection–dispersion with simple source/sink terms.

    def __init__(
        self,
        grid: Grid,
        flow: Flow,
        sim_cfg: SimulationConfig,
        properties: Dict[str, PropertyConfig],
        initial_profiles: Dict[str, np.ndarray],
        discharges: Optional[List[Discharge]] = None,
    ) -> None:
        self.grid = grid
        self.flow = flow
        self.cfg = sim_cfg
        self.properties = properties
        self.discharges = discharges or []

        self.names: List[str] = [name for name, cfg in properties.items()
                                 if cfg.active]

        self.C: Dict[str, np.ndarray] = {}
        for name in self.names:
            profile = np.array(initial_profiles[name], dtype=float)
            if profile.shape != (grid.nc,):
                raise ValueError(
                    f"Initial profile for {name} must have shape ({grid.nc},)"
                )
            self.C[name] = profile.copy()

        # precompute discharge indices
        self._discharge_idx: List[Dict[str, int]] = []
        for d in self.discharges:
            idx = int(np.argmin(np.abs(self.grid.x - d.x)))
            self._discharge_idx.append({"cell": idx})

    # -------------------- main loop --------------------------------------

    def run(self) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        dt = self.cfg.dt
        n_steps = int(np.ceil(self.cfg.duration / dt))

        times_out: List[float] = [0.0]
        out_profiles: Dict[str, List[np.ndarray]] = {
            name: [self.C[name].copy()] for name in self.names
        }
        next_out_time = self.cfg.output_interval

        current_time = 0.0
        for step in range(1, n_steps + 1):
            current_time = step * dt

            self._advance_one_step(dt)

            if current_time + 1e-12 >= next_out_time:
                times_out.append(current_time)
                for name in self.names:
                    out_profiles[name].append(self.C[name].copy())
                next_out_time += self.cfg.output_interval

        times_arr = np.array(times_out)
        results = {name: np.vstack(out_profiles[name])
                   for name in self.names}
        return times_arr, results

    # -------------------- internal helpers -------------------------------

    def _advance_one_step(self, dt: float) -> None:
        # copy current state
        Cn = {name: self.C[name].copy() for name in self.names}

        # source terms
        sources = {name: np.zeros_like(Cn[name]) for name in self.names}

        # identify special properties, if present
        name_map = {name.lower(): name for name in self.names}
        bod_name = name_map.get("bod")
        do_name = name_map.get("do")
        co2_name = name_map.get("co2")

        # generic decay/growth and gas exchange
        for name in self.names:
            cfg = self.properties[name]
            C = Cn[name]
            S = sources[name]

            if cfg.decay_rate != 0.0:
                S -= cfg.decay_rate * C

            if cfg.growth_rate != 0.0:
                if cfg.logistic_max > 0.0:
                    S += cfg.growth_rate * C * (1.0 - C / cfg.logistic_max)
                else:
                    S += cfg.growth_rate * C

            if cfg.reaeration_rate != 0.0:
                S += cfg.reaeration_rate * (cfg.equilibrium_conc - C)

        # BOD – DO coupling
        if bod_name is not None and do_name is not None:
            bod_cfg = self.properties[bod_name]
            do_cfg = self.properties[do_name]
            C_bod = Cn[bod_name]
            C_do = Cn[do_name]

            k_bod = bod_cfg.decay_rate
            if k_bod != 0.0 and do_cfg.oxygen_per_bod != 0.0:
                bod_decay_potential = k_bod * C_bod  # mg/L/s

                o2_needed = do_cfg.oxygen_per_bod * bod_decay_potential
                ratio = np.ones_like(C_do)
                mask = o2_needed > C_do / dt
                ratio[mask] = (C_do[mask] / dt) / (o2_needed[mask] + 1e-30)
                ratio = np.clip(ratio, 0.0, 1.0)

                bod_decay_effective = bod_decay_potential * ratio

                sources[bod_name] -= bod_decay_effective
                sources[do_name] -= do_cfg.oxygen_per_bod * bod_decay_effective

        # discharges
        for d, d_idx in zip(self.discharges, self._discharge_idx):
            i = d_idx["cell"]
            if d.flow <= 0.0:
                continue
            for name in self.names:
                if name in d.concentrations:
                    Cin = d.concentrations[name]
                    Ccell = Cn[name][i]
                    sources[name][i] += (
                        d.flow / self.grid.cell_volume * (Cin - Ccell)
                    )

        # advection + diffusion per property
        new_C: Dict[str, np.ndarray] = {}
        for name in self.names:
            cfg = self.properties[name]
            C0 = Cn[name]
            S = sources[name]
            C1 = self._step_single_property(C0, S, cfg.boundary, dt)

            if cfg.min_value is not None:
                C1 = np.maximum(C1, cfg.min_value)
            if cfg.max_value is not None and cfg.max_value > 0.0:
                C1 = np.minimum(C1, cfg.max_value)

            new_C[name] = C1

        self.C = new_C

    def _step_single_property(
        self,
        C: np.ndarray,
        S: np.ndarray,
        bc: BoundaryCondition,
        dt: float,
    ) -> np.ndarray:
        grid = self.grid
        flow = self.flow

        adv_term = np.zeros_like(C)
        diff_explicit_term = np.zeros_like(C)

        if flow.advection_on and flow.velocity != 0.0:
            adv_term = self._advection_term(C, bc, flow, grid)

        if flow.diffusion_on and flow.diffusivity > 0.0 and \
                self.cfg.time_scheme == "explicit":
            diff_explicit_term = self._diffusion_term(C, bc, flow, grid)

        rhs = C + dt * (adv_term + diff_explicit_term + S)

        if not (flow.diffusion_on and flow.diffusivity > 0.0 and
                self.cfg.time_scheme != "explicit"):
            C_new = rhs
        else:
            theta = 1.0 if self.cfg.time_scheme == "implicit" else 0.5
            C_new = self._implicit_diffusion_step(
                C, rhs, bc, flow, grid, dt, theta
            )

        C_new = self._apply_boundary(C_new, bc)
        return C_new

    # ------------------- discretisation pieces ---------------------------

    @staticmethod
    def _apply_boundary(C: np.ndarray, bc: BoundaryCondition) -> np.ndarray:
        if C.size == 0:
            return C

        if bc.left_type == "dirichlet":
            C[0] = bc.left_value
        elif bc.left_type == "neumann":
            C[0] = C[1]

        if bc.right_type == "dirichlet":
            C[-1] = bc.right_value
        elif bc.right_type == "neumann":
            C[-1] = C[-2]
        return C

    def _advection_term(
        self,
        C: np.ndarray,
        bc: BoundaryCondition,
        flow: Flow,
        grid: Grid,
    ) -> np.ndarray:
        N = grid.nc
        dx = grid.dx
        u = flow.velocity
        scheme = self.cfg.advection_scheme

        def value_at(idx: int) -> float:
            if 0 <= idx < N:
                return C[idx]
            if idx < 0:
                if bc.left_type == "dirichlet":
                    return bc.left_value
                return C[0]
            else:
                if bc.right_type == "dirichlet":
                    return bc.right_value
                return C[-1]

        F = np.zeros(N + 1)

        if scheme == "upwind":
            if u >= 0.0:
                for j in range(N + 1):
                    up_idx = j - 1
                    F[j] = u * value_at(up_idx)
            else:
                for j in range(N + 1):
                    up_idx = j
                    F[j] = u * value_at(up_idx)

        elif scheme == "central":
            for j in range(N + 1):
                left_idx = j - 1
                right_idx = j
                C_left = value_at(left_idx)
                C_right = value_at(right_idx)
                F[j] = u * 0.5 * (C_left + C_right)

        else:  # "quick"
            if u >= 0.0:
                for j in range(N + 1):
                    if 1 <= j <= N - 1:
                        C_im1 = value_at(j - 2)
                        C_i = value_at(j - 1)
                        C_ip1 = value_at(j)
                        F[j] = u * (-C_im1 + 6 * C_i + 3 * C_ip1) / 8.0
                    else:
                        up_idx = j - 1
                        F[j] = u * value_at(up_idx)
            else:
                for j in range(N + 1):
                    if 1 <= j <= N - 1:
                        C_im1 = value_at(j - 1)
                        C_i = value_at(j)
                        C_ip1 = value_at(j + 1)
                        F[j] = u * (3 * C_im1 + 6 * C_i - C_ip1) / 8.0
                    else:
                        up_idx = j
                        F[j] = u * value_at(up_idx)

        dCdt = np.zeros_like(C)
        for i in range(N):
            dCdt[i] = -(F[i + 1] - F[i]) / dx
        return dCdt

    def _diffusion_term(
        self,
        C: np.ndarray,
        bc: BoundaryCondition,
        flow: Flow,
        grid: Grid,
    ) -> np.ndarray:
        N = grid.nc
        dx = grid.dx
        K = flow.diffusivity

        C_ext = np.empty(N + 2)
        C_ext[1:-1] = C

        if bc.left_type == "dirichlet":
            C_ext[0] = bc.left_value
        else:
            C_ext[0] = C[0]

        if bc.right_type == "dirichlet":
            C_ext[-1] = bc.right_value
        else:
            C_ext[-1] = C[-1]

        dCdt = np.zeros_like(C)
        for i in range(N):
            d2Cdx2 = (C_ext[i] - 2.0 * C_ext[i + 1] + C_ext[i + 2]) / (dx ** 2)
            dCdt[i] = K * d2Cdx2
        return dCdt

    def _implicit_diffusion_step(
        self,
        C_old: np.ndarray,
        rhs: np.ndarray,
        bc: BoundaryCondition,
        flow: Flow,
        grid: Grid,
        dt: float,
        theta: float,
    ) -> np.ndarray:
        N = grid.nc
        dx = grid.dx
        K = flow.diffusivity

        if K == 0.0:
            return rhs

        alpha = K * dt / (dx ** 2)

        a = np.zeros(N)  # sub-diagonal
        b = np.zeros(N)  # main
        c = np.zeros(N)  # super

        for i in range(N):
            b[i] = 1.0 + 2.0 * theta * alpha
        for i in range(N - 1):
            a[i + 1] = -theta * alpha
            c[i] = -theta * alpha

        if bc.left_type == "dirichlet":
            b[0] = 1.0
            c[0] = 0.0
        else:
            b[0] = 1.0 + theta * alpha
            c[0] = -theta * alpha

        if bc.right_type == "dirichlet":
            b[-1] = 1.0
            a[-1] = 0.0
        else:
            b[-1] = 1.0 + theta * alpha
            a[-1] = -theta * alpha

        rhs_eff = rhs.copy()

        if theta < 1.0:
            diff_old = self._diffusion_term(C_old, bc, flow, grid)
            rhs_eff += (1.0 - theta) * dt * diff_old

        if bc.left_type == "dirichlet":
            rhs_eff[0] = bc.left_value
        if bc.right_type == "dirichlet":
            rhs_eff[-1] = bc.right_value

        C_new = self._solve_tridiagonal(a, b, c, rhs_eff)
        return C_new

    @staticmethod
    def _solve_tridiagonal(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        n = len(b)
        cp = np.zeros(n)
        dp = np.zeros(n)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if denom == 0.0:
                denom = 1e-20
            cp[i] = c[i] / denom if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x
