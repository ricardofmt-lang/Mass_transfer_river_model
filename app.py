from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import numpy as np

AdvectionScheme = Literal["upwind", "central", "quick"]
TimeScheme = Literal["explicit", "implicit", "semi-implicit"]


# ------------------------- Core data structures ------------------------------


@dataclass
class Grid:
    length: float   # m
    width: float    # m
    depth: float    # m
    nc: int         # number of cells

    def __post_init__(self):
        self.dx = self.length / self.nc
        # cell centers
        self.x = np.linspace(self.dx / 2, self.length - self.dx / 2, self.nc)

    @property
    def area_cross_section(self) -> float:
        return self.width * self.depth


@dataclass
class Flow:
    velocity: float      # m/s
    diffusivity: float   # m^2/s
    diffusion_on: bool = True

    def compute_numbers(self, dt: float, dx: float) -> Dict[str, float]:
        return {
            "courant": self.velocity * dt / dx,
            "diffusion": self.diffusivity * dt / dx**2 if self.diffusion_on else 0.0,
        }


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
    decay_rate: float = 0.0             # 1/s, first-order decay
    reaeration_rate: float = 0.0        # 1/s, towards equilibrium_conc
    equilibrium_conc: float = 0.0       # mg/L etc., for DO/CO2
    oxygen_per_bod: float = 0.0         # mg O2 per mg BOD (used only for DO)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    boundary: BoundaryCondition = field(default_factory=BoundaryCondition)


@dataclass
class Discharge:
    x: float                            # m from upstream
    flow: float                         # m^3/s
    concentrations: Dict[str, float]    # property name -> conc
    # cell_index will be filled when Simulation is created
    cell_index: int = 0


@dataclass
class SimulationConfig:
    dt: float
    duration: float
    output_interval: float
    advection_scheme: AdvectionScheme
    time_scheme: TimeScheme


# ------------------------- Simulation class ----------------------------------


class Simulation:
    def __init__(
        self,
        grid: Grid,
        flow: Flow,
        sim_cfg: SimulationConfig,
        properties: Dict[str, PropertyConfig],
        initial_profiles: Dict[str, np.ndarray],
        discharges: List[Discharge],
    ):
        self.grid = grid
        self.flow = flow
        self.sim_cfg = sim_cfg
        self.properties_cfg = properties
        self.discharges = discharges

        # active properties
        self.prop_names = [name for name, pc in properties.items() if pc.active]

        # current state (name -> 1D array)
        self.state: Dict[str, np.ndarray] = {}
        for name in self.prop_names:
            prof = np.array(initial_profiles[name], dtype=float)
            if prof.shape[0] != self.grid.nc:
                raise ValueError(f"Initial profile for {name} has wrong length")
            self.state[name] = prof

        # main channel flow (for mixing with discharges)
        self.Q_main = self.flow.velocity * self.grid.area_cross_section

        # map discharge x to nearest cell index
        for d in self.discharges:
            if d.x < 0 or d.x > self.grid.length:
                raise ValueError(f"Discharge at x={d.x} outside domain")
            # nearest cell center; subtract 0.5 cell because centers are at dx/2, 3dx/2,...
            d.cell_index = int(round(d.x / self.grid.dx - 0.5))
            d.cell_index = max(0, min(self.grid.nc - 1, d.cell_index))

    # -------------------- boundary conditions ---------------------------------

    def _apply_boundary(self, arr: np.ndarray, bc: BoundaryCondition) -> np.ndarray:
        """Apply BCs directly on the 1D array (in-place)."""
        if bc.left_type == "dirichlet":
            arr[0] = bc.left_value
        elif bc.left_type == "neumann":
            arr[0] = arr[1]

        if bc.right_type == "dirichlet":
            arr[-1] = bc.right_value
        elif bc.right_type == "neumann":
            arr[-1] = arr[-2]

        return arr

    # -------------------- explicit step --------------------------------------

    def _explicit_step_single(self, name: str, C: np.ndarray) -> np.ndarray:
        pc = self.properties_cfg[name]
        v = self.flow.velocity
        D = self.flow.diffusivity if self.flow.diffusion_on else 0.0
        dx = self.grid.dx
        dt = self.sim_cfg.dt
        nc = self.grid.nc

        C_old = C.copy()
        C_new = C.copy()

        # BCs on old values
        C_old = self._apply_boundary(C_old, pc.boundary)

        for i in range(1, nc - 1):
            # advection
            if self.sim_cfg.advection_scheme == "upwind":
                if v >= 0:
                    dCdx = (C_old[i] - C_old[i - 1]) / dx
                else:
                    dCdx = (C_old[i + 1] - C_old[i]) / dx
            elif self.sim_cfg.advection_scheme == "central":
                dCdx = (C_old[i + 1] - C_old[i - 1]) / (2 * dx)
            else:  # "quick" – simple higher-order upwind-like approximation
                if v >= 0:
                    im2 = max(0, i - 2)
                    dCdx = (-C_old[im2] + 5 * C_old[i - 1] + 2 * C_old[i]) / (6 * dx)
                else:
                    ip2 = min(nc - 1, i + 2)
                    dCdx = (C_old[ip2] - 5 * C_old[i + 1] - 2 * C_old[i]) / (6 * dx)

            adv = -v * dCdx

            # diffusion
            if self.flow.diffusion_on and D > 0:
                d2Cdx2 = (C_old[i + 1] - 2 * C_old[i] + C_old[i - 1]) / dx**2
                diff = D * d2Cdx2
            else:
                diff = 0.0

            C_new[i] = C_old[i] + dt * (adv + diff)

        # Re-apply BCs after update
        C_new = self._apply_boundary(C_new, pc.boundary)
        return C_new

    # -------------------- implicit step --------------------------------------

    def _implicit_step_single(self, name: str, C: np.ndarray) -> np.ndarray:
        """
        Fully implicit step: central advection + diffusion.
        """
        pc = self.properties_cfg[name]
        v = self.flow.velocity
        D = self.flow.diffusivity if self.flow.diffusion_on else 0.0
        dx = self.grid.dx
        dt = self.sim_cfg.dt
        nc = self.grid.nc

        C_old = C.copy()
        bc = pc.boundary

        a = np.zeros(nc)  # lower diagonal
        b = np.zeros(nc)  # main diagonal
        c = np.zeros(nc)  # upper diagonal
        rhs = C_old.copy()

        # interior nodes
        for i in range(1, nc - 1):
            # central advection discretization (implicit)
            adv_left = -v * (-1 / (2 * dx))  # coeff for C_{i-1}
            adv_center = 0.0                 # coeff for C_i
            adv_right = -v * (1 / (2 * dx))  # coeff for C_{i+1}

            diff_left = D / dx**2
            diff_center = -2 * D / dx**2
            diff_right = D / dx**2

            a[i] = dt * (adv_left + diff_left)
            b[i] = 1.0 + dt * (adv_center + diff_center)
            c[i] = dt * (adv_right + diff_right)

        # left boundary
        if bc.left_type == "dirichlet":
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            rhs[0] = bc.left_value
        else:  # neumann ~ zero gradient
            i = 0
            diff_left = D / dx**2
            diff_center = -2 * D / dx**2
            diff_right = D / dx**2
            a[i] = 0.0
            b[i] = 1.0 + dt * (diff_center + diff_left)
            c[i] = dt * diff_right

        # right boundary
        if bc.right_type == "dirichlet":
            a[-1] = 0.0
            b[-1] = 1.0
            c[-1] = 0.0
            rhs[-1] = bc.right_value
        else:  # neumann
            i = nc - 1
            diff_left = D / dx**2
            diff_center = -2 * D / dx**2
            diff_right = D / dx**2
            a[i] = dt * diff_left
            b[i] = 1.0 + dt * (diff_center + diff_right)
            c[i] = 0.0

        # tridiagonal solve (Thomas algorithm)
        for i in range(1, nc):
            if b[i - 1] == 0:
                continue
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            rhs[i] -= m * rhs[i - 1]

        C_new = np.zeros(nc)
        C_new[-1] = rhs[-1] / b[-1] if b[-1] != 0 else rhs[-1]
        for i in range(nc - 2, -1, -1):
            C_new[i] = (rhs[i] - c[i] * C_new[i + 1]) / b[i] if b[i] != 0 else rhs[i]

        C_new = self._apply_boundary(C_new, bc)
        return C_new

    # -------------------- semi-implicit step ---------------------------------

    def _semi_implicit_step_single(self, name: str, C: np.ndarray) -> np.ndarray:
        """
        Semi-implicit: advection explicit, diffusion implicit.
        """
        pc = self.properties_cfg[name]
        v = self.flow.velocity
        D = self.flow.diffusivity if self.flow.diffusion_on else 0.0
        dx = self.grid.dx
        dt = self.sim_cfg.dt
        nc = self.grid.nc

        C_old = C.copy()
        bc = pc.boundary

        # explicit advection -> RHS
        C_tmp = C_old.copy()
        C_tmp = self._apply_boundary(C_tmp, bc)
        adv_term = np.zeros(nc)

        for i in range(1, nc - 1):
            if self.sim_cfg.advection_scheme == "upwind":
                if v >= 0:
                    dCdx = (C_tmp[i] - C_tmp[i - 1]) / dx
                else:
                    dCdx = (C_tmp[i + 1] - C_tmp[i]) / dx
            elif self.sim_cfg.advection_scheme == "central":
                dCdx = (C_tmp[i + 1] - C_tmp[i - 1]) / (2 * dx)
            else:  # "quick" approx -> central
                dCdx = (C_tmp[i + 1] - C_tmp[i - 1]) / (2 * dx)

            adv_term[i] = -v * dCdx

        rhs = C_old + dt * adv_term

        # implicit diffusion
        a = np.zeros(nc)
        b = np.zeros(nc)
        c = np.zeros(nc)

        for i in range(1, nc - 1):
            a[i] = dt * D / dx**2
            b[i] = 1.0 - 2 * dt * D / dx**2
            c[i] = dt * D / dx**2

        # boundaries
        if bc.left_type == "dirichlet":
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            rhs[0] = bc.left_value
        else:
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0

        if bc.right_type == "dirichlet":
            a[-1] = 0.0
            b[-1] = 1.0
            c[-1] = 0.0
            rhs[-1] = bc.right_value
        else:
            a[-1] = 0.0
            b[-1] = 1.0
            c[-1] = 0.0

        # tridiagonal solve
        for i in range(1, nc):
            if b[i - 1] == 0:
                continue
            m = a[i] / b[i - 1]
            b[i] -= m * c[i - 1]
            rhs[i] -= m * rhs[i - 1]

        C_new = np.zeros(nc)
        C_new[-1] = rhs[-1] / b[-1] if b[-1] != 0 else rhs[-1]
        for i in range(nc - 2, -1, -1):
            C_new[i] = (rhs[i] - c[i] * C_new[i + 1]) / b[i] if b[i] != 0 else rhs[i]

        C_new = self._apply_boundary(C_new, bc)
        return C_new

    # -------------------- reactions & discharges -----------------------------

    def _apply_reactions(self):
        dt = self.sim_cfg.dt

        # detect BOD and DO names if present
        bod_name = None
        do_name = None
        for n in self.prop_names:
            if n.lower() == "bod":
                bod_name = n
            if n.lower() == "do":
                do_name = n

        # independent reactions (decay, reaeration)
        for name in self.prop_names:
            pc = self.properties_cfg[name]
            C = self.state[name]

            # first-order decay
            if pc.decay_rate != 0.0:
                C *= np.exp(-pc.decay_rate * dt)

            # simple equilibration toward equilibrium concentration
            if pc.reaeration_rate != 0.0:
                C += dt * pc.reaeration_rate * (pc.equilibrium_conc - C)

            self.state[name] = C

        # BOD consuming DO (after BOD decay applied)
        if bod_name is not None and do_name is not None:
            bod_cfg = self.properties_cfg[bod_name]
            do_cfg = self.properties_cfg[do_name]

            if do_cfg.oxygen_per_bod != 0.0 and bod_cfg.decay_rate != 0.0:
                B = self.state[bod_name]
                # amount of BOD removed in dt
                dB = (1.0 - np.exp(-bod_cfg.decay_rate * dt)) * B
                self.state[bod_name] = B - dB
                self.state[do_name] = self.state[do_name] - do_cfg.oxygen_per_bod * dB

    def _apply_discharges(self):
        dt = self.sim_cfg.dt
        Q_main = self.Q_main if self.Q_main > 0 else 1e-9

        for d in self.discharges:
            cell = d.cell_index
            for name in self.prop_names:
                C_cell = self.state[name][cell]
                C_d = d.concentrations.get(name, C_cell)
                vol_main = Q_main * dt
                vol_dis = d.flow * dt
                if vol_dis <= 0:
                    continue
                C_new = (C_cell * vol_main + C_d * vol_dis) / (vol_main + vol_dis)
                self.state[name][cell] = C_new

    # -------------------- public run method ----------------------------------

    def run(self):
        dt = self.sim_cfg.dt
        duration = self.sim_cfg.duration
        n_steps = int(np.round(duration / dt))
        output_interval = self.sim_cfg.output_interval

        output_times: List[float] = []
        results: Dict[str, List[np.ndarray]] = {n: [] for n in self.prop_names}

        t = 0.0
        next_output = 0.0

        # store initial state
        for n in self.prop_names:
            results[n].append(self.state[n].copy())
        output_times.append(0.0)
        next_output += output_interval

        for step in range(1, n_steps + 1):
            # transport
            for name in self.prop_names:
                C = self.state[name]
                if self.sim_cfg.time_scheme == "explicit":
                    C_new = self._explicit_step_single(name, C)
                elif self.sim_cfg.time_scheme == "implicit":
                    C_new = self._implicit_step_single(name, C)
                else:  # semi-implicit
                    C_new = self._semi_implicit_step_single(name, C)
                self.state[name] = C_new

            # reactions (decay, reaeration, DO–BOD coupling)
            self._apply_reactions()

            # lateral discharges
            self._apply_discharges()

            t = step * dt
            # record if time to output
            if t + 1e-9 >= next_output:
                for n in self.prop_names:
                    results[n].append(self.state[n].copy())
                output_times.append(t)
                next_output += output_interval

        out = {n: np.vstack(v) for n, v in results.items()}
        return np.array(output_times), out
