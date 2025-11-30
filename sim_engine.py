# sim_engine.py
import numpy as np

class Grid:
    def __init__(self, length, width, depth, NC):
        self.length = length       # river length (m)
        self.width = width         # channel width (m)
        self.depth = depth         # avg depth (m)
        self.NC = NC               # number of cells
        self.dx = length / NC      # cell size
        self.x = np.linspace(self.dx/2, length-self.dx/2, NC)  # cell centers

class FlowProperties:
    def __init__(self, velocity, diffusivity, slope):
        self.v = velocity         # flow velocity (m/s)
        self.D = diffusivity      # longitudinal diffusivity (m^2/s)
        self.slope = slope        # channel slope (used for gas transfer)
        self.Courant = None       # to be computed

    def compute_numbers(self, dt, dx):
        self.Courant = self.v * dt / dx
        self.DiffusionNr = self.D * dt / dx**2

class Atmosphere:
    def __init__(self, temp, wind, humidity, solar, latitude, cloud, gas_data):
        self.temp = temp         # air temp (°C)
        self.wind = wind         # wind speed (m/s)
        self.humidity = humidity # RH (%)
        self.solar = solar       # solar irradiance (W/m^2)
        self.latitude = latitude # for radiation calc
        self.cloud = cloud       # cloud cover (%)
        self.gas_data = gas_data # dict of GasProperties (e.g. Henry constants, atms. p)

class BoundaryCondition:
    def __init__(self, left_type, left_value, right_type, right_value, cyclic=False):
        # types: 0=fixed value, 1=zero-flux (Neumann), etc.
        self.left_type = left_type
        self.left_value = left_value
        self.right_type = right_type
        self.right_value = right_value
        self.cyclic = cyclic

class Discharge:
    def __init__(self, cell, rate, properties):
        self.cell = cell          # grid cell index of discharge
        self.rate = rate          # flow rate (m^3/s)
        self.props = properties   # dict of property concentration at discharge

class Property:
    def __init__(self, name, initial, decay_rate=0.0):
        self.name = name
        self.C = np.array(initial, dtype=float)  # concentration in each cell
        self.decay_rate = decay_rate  # first-order decay (s⁻¹)
        # For DO/BOD, one might add coupling (e.g. BOD to DO sink/growth)
        # For Temperature, DO, CO2: handle free-surface fluxes separately
        self.boundary = None     # BoundaryCondition object

    def apply_decay(self, dt):
        """Apply first-order decay (or growth if negative rate)."""
        self.C *= np.exp(-self.decay_rate * dt)

class Simulation:
    def __init__(self, grid, flow, atmosphere, dt, duration, time_scheme='explicit', adv_scheme='central'):
        self.grid = grid
        self.flow = flow
        self.atmo = atmosphere
        self.dt = dt
        self.Tmax = duration
        self.nt = int(duration / dt)
        self.properties = {}  # dict of Property objects
        self.discharges = []  # list of Discharge objects
        self.time_scheme = time_scheme    # 'explicit', 'implicit', 'semi-implicit'
        self.adv_scheme = adv_scheme      # 'upwind', 'central', 'QUICK'

    def add_property(self, prop: Property):
        self.properties[prop.name] = prop

    def add_discharge(self, d: Discharge):
        self.discharges.append(d)

    def step_explicit(self, prop: Property):
        """One explicit time step for advection-diffusion of a property."""
        C_old = prop.C.copy()
        C_new = prop.C.copy()
        v = self.flow.v
        D = self.flow.D
        dx = self.grid.dx
        dt = self.dt
        
        # Compute Courant and diffusion numbers
        Cr = v*dt/dx
        Fo = D*dt/dx**2

        # Apply boundary conditions (simple Dirichlet example)
        if prop.boundary.left_type == 0:
            C_old[0] = prop.boundary.left_value
        if prop.boundary.right_type == 0:
            C_old[-1] = prop.boundary.right_value
        
        # Update interior points
        for i in range(1, self.grid.NC-1):
            adv_term = 0
            if self.adv_scheme == 'upwind':
                # Upwind differencing
                if v >= 0:
                    adv_term = - v * dt / dx * (C_old[i] - C_old[i-1])
                else:
                    adv_term = - v * dt / dx * (C_old[i+1] - C_old[i])
            elif self.adv_scheme == 'central':
                # Central differencing
                adv_term = - v * dt / (2*dx) * (C_old[i+1] - C_old[i-1])
            else:
                # Placeholder for QUICK scheme (more complex)
                # For QUICK, use a quadratic interpolation (requires ghost cells)
                adv_term = - v * dt / (2*dx) * (C_old[i+1] - C_old[i-1])
            
            diff_term = D * dt / dx**2 * (C_old[i+1] - 2*C_old[i] + C_old[i-1])
            C_new[i] = C_old[i] + adv_term + diff_term

        # (Reflecting boundaries for simplicity; other BCs can be coded)
        C_new[0]  = C_new[1]
        C_new[-1] = C_new[-2]

        prop.C = C_new

    def step_implicit(self, prop: Property):
        """One implicit time step (solving a linear system)."""
        # Assemble tridiagonal system for implicit scheme: A * C_new = C_old + sources
        N = self.grid.NC
        v = self.flow.v
        D = self.flow.D
        dx = self.grid.dx
        dt = self.dt
        Cr = v*dt/dx
        Fo = D*dt/dx**2

        # Coefficients for tridiagonal matrix (assuming constant v, D)
        a = np.zeros(N)  # lower diag
        b = np.zeros(N)  # main diag
        c = np.zeros(N)  # upper diag
        RHS = prop.C.copy()

        for i in range(N):
            # interior
            a[i] = -Fo + max(v*dt/(2*dx), 0)
            b[i] = 1 + 2*Fo
            c[i] = -Fo + max(-v*dt/(2*dx), 0)
            # adjust for boundaries
            if i == 0:
                # Left boundary: enforce Dirichlet if specified
                if prop.boundary.left_type == 0:
                    b[i] = 1.0
                    a[i] = c[i] = 0.0
                    RHS[i] = prop.boundary.left_value
                else:
                    # Zero-flux: use one-sided discretization
                    b[i] = 1 + 2*Fo
                    c[i] = -2*Fo
            if i == N-1:
                if prop.boundary.right_type == 0:
                    b[i] = 1.0
                    a[i] = c[i] = 0.0
                    RHS[i] = prop.boundary.right_value
                else:
                    b[i] = 1 + 2*Fo
                    a[i] = -2*Fo
        
        # Solve tridiagonal system (Thomas algorithm)
        # Forward elimination
        for i in range(1, N):
            m = a[i]/b[i-1]
            b[i] -= m*c[i-1]
            RHS[i] -= m*RHS[i-1]
        # Back substitution
        C_new = np.zeros(N)
        C_new[-1] = RHS[-1]/b[-1]
        for i in range(N-2, -1, -1):
            C_new[i] = (RHS[i] - c[i]*C_new[i+1]) / b[i]

        prop.C = C_new

    def apply_sinks_and_sources(self):
        """Apply source/sink terms (discharges, decay, gas exchange)."""
        # Example: apply first-order decay
        for prop in self.properties.values():
            prop.apply_decay(self.dt)
        # Additional sinks (e.g. BOD consuming DO) or gas exchange would be added here.
        # For instance, update DO based on BOD decay: DO decreases by (BOD_decayed * stoichiometry).

        # Gas exchange at free surface (ODE-style):
        # Example: DO flux = k_O2 * (C*_O2 - C_DO)
        # where C*_O2 is saturation conc (Henry's law * atmospheric p).
        # Similarly for CO2. These fluxes can be converted to a per-volume source term.
        # Here we would update prop.C accordingly (details omitted for brevity).
        pass

    def run(self):
        """Run the simulation over all time steps, returning stored results."""
        results = {name: [prop.C.copy()] for name, prop in self.properties.items()}
        for n in range(self.nt):
            # apply lateral discharges as boundary sources
            for dis in self.discharges:
                cell = dis.cell
                prop = self.properties.get(dis.props['name'])
                if prop:
                    # add mass from discharge: C_new = (C_old*Q + C_disch*dis.rate) / (Q + dis.rate)
                    # (This is a simple mixing at the cell; more complex schemes possible.)
                    Q = self.flow.v * self.grid.width * self.grid.depth
                    Cd = dis.props['concentration']
                    prop.C[cell] = (prop.C[cell]*Q + Cd*dis.rate) / (Q + dis.rate)

            # choose time-stepping
            for prop in self.properties.values():
                if self.time_scheme == 'explicit':
                    self.step_explicit(prop)
                else:
                    self.step_implicit(prop)
            
            # apply sinks/sources (decay, gas, etc.)
            self.apply_sinks_and_sources()

            # Record results
            for name, prop in self.properties.items():
                results[name].append(prop.C.copy())
        return results
