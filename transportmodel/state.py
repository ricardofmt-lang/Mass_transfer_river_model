"""Mutable one-based state matching the VBA user-defined types.

The public app and JSON configuration can use readable snake-case keys. The
calculation layer uses the original field names and 1-based arrays so each
VBA assignment has an unambiguous Python counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Index zero is deliberately unused; Excel/VBA arrays in this model start at one.
def zeros(n: int) -> list[float]:
    """Allocate indices 1..n; index zero remains an unused VBA-style slot."""
    return [0.0] * (n + 1)


@dataclass
# Grid stores geometry and cell-centre positions. Length and dx are metres; areas are square metres.
class Grid:
    Length: float = 0.0
    AreaVertical: float = 0.0
    AreaH: float = 0.0
    waterdepth: float = 0.0
    RiverWidth: float = 0.0
    NC: int = 0
    dx: float = 0.0
    xc: list[float] = field(default_factory=list)


@dataclass
# FlowProperties couples velocity and dispersion with dimensionless Courant and diffusion numbers.
class FlowProperties:
    velocity: float = 0.0
    Diffusivity: float = 0.0
    Riverslope: float = 0.0
    DiffusionNr: float = 0.0
    CourantNr: float = 0.0


@dataclass
# GasProperties keeps pressure, molecular weight, and the temperature/Henry lookup table together.
class GasProperties:
    Label: str = ""
    partialpressure: float = 0.0
    N_HenryConstants: int = 0
    HenryConstValues: list[list[float]] = field(default_factory=list)
    MolecularWeight: float = 0.0


@dataclass
# Atmosphere holds forcing shared by surface heat and gas exchange across all river cells.
class Atmosphere:
    Temperature: float = 0.0
    windspeed: float = 0.0
    Humidity: float = 0.0
    h_min: float = 0.0
    SolarConstant: float = 0.0
    Latitude: float = 0.0
    SkyTemperature: float = 0.0
    SkyTemperatureImposed: bool = False
    CloudCover: float = 0.0
    Tsr: float = 0.0
    Tss: float = 0.0
    Gas: list[GasProperties] = field(default_factory=list)


@dataclass
# Controls mirrors the macro's time, print, and scheme switches; T advances in seconds.
class Controls:
    T: float = 0.0
    timedays: float = 0.0
    dt: float = 0.0
    SimulationDuration: float = 0.0
    printtime: float = 0.0
    Printnumber: float = 0.0
    NumberOfOutputs: int = 0
    DtPrint: float = 0.0
    PrintLine: int = 26
    FirstPrintLine: int = 26
    FirstOutput: bool = False
    Advection: bool = False
    Diffusion: bool = False
    AdvType: str = ""
    QUICK_UP: bool = False
    QUICK_UP_Ratio: float = 0.0
    TimeDiscretisation: str = ""


@dataclass
# One lateral source has a cell, volumetric flow, and a property-specific value.
class DischargeProperties:
    Number: int = 0
    name: str = ""
    cell: int = 0
    VolumeRate: float = 0.0
    SpecificValue: float = 0.0


@dataclass
# Each property has its own discharge list, because a source can have different concentrations.
class DischargesList:
    NumberOfDischarges: int = 0
    DischargesActive: bool = False
    Discharge: list[DischargeProperties] = field(default_factory=list)


@dataclass
# This groups inlet values, surface switches, and lateral sources for one property.
class BoundaryCondition:
    LeftType: int = 0
    LeftValue: float = 0.0
    RightType: int = 0
    RightValue: float = 0.0
    CyclicBoundary: bool = False
    FreeSurfaceFlux: bool = False
    FreeSurfaceSensibleHeatFlux: bool = False
    FreeSurfaceLatentHeatFlux: bool = False
    FreeSurfaceRadiativeHeatFlux: bool = False
    ValuesForFreeSurfaceGasExchange: GasProperties = field(default_factory=GasProperties)
    PropDischarges: DischargesList = field(default_factory=DischargesList)


@dataclass
# Old and NewValue are one-based concentration arrays; source terms mutate NewValue in place.
class PropertyState:
    SheetName: str = ""
    name: str = ""
    units: str = ""
    Active: bool = False
    Old: list[float] = field(default_factory=list)
    NewValue: list[float] = field(default_factory=list)
    Default: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    MinimumActive: bool = False
    MaximumActive: bool = False
    DecayOrGrowth: bool = False
    Decayrate: float = 0.0
    GrowthRate: float = 0.0
    MaxValLogisticEquation: float = 0.0
    GrazingKsat: float = 0.0
    AnaerobicRespiration: bool = False
    IntegratedAerobicResp: list[float] = field(default_factory=list)
    IntegratedAnaerobicResp: list[float] = field(default_factory=list)
    Boundary: BoundaryCondition = field(default_factory=BoundaryCondition)


@dataclass
# The five named properties share one time loop, but each can be active independently.
class PropertiesList:
    Generic: PropertyState = field(default_factory=PropertyState)
    Temperature: PropertyState = field(default_factory=PropertyState)
    DO: PropertyState = field(default_factory=PropertyState)
    BOD: PropertyState = field(default_factory=PropertyState)
    CO2: PropertyState = field(default_factory=PropertyState)


@dataclass
# These arrays represent the five-point transport stencil and the linear-system diagonals.
class EquationCoef:
    A: list[float] = field(default_factory=list)
    B: list[float] = field(default_factory=list)
    e: list[float] = field(default_factory=list)
    f: list[float] = field(default_factory=list)
    g: list[float] = field(default_factory=list)
    b_up: list[float] = field(default_factory=list)
    e_up: list[float] = field(default_factory=list)
    f_up: list[float] = field(default_factory=list)
    L_LeftCoef: list[float] = field(default_factory=list)
    LeftCoef: list[float] = field(default_factory=list)
    CenterCoef: list[float] = field(default_factory=list)
    RigthCoef: list[float] = field(default_factory=list)
    R_RigthCoef: list[float] = field(default_factory=list)
    TiCoef: list[float] = field(default_factory=list)

    @classmethod
    # Allocate every coefficient array with the unused index-zero slot required by the port.
    def allocate(cls, n: int) -> "EquationCoef":
        return cls(**{name: zeros(n) for name in cls.__dataclass_fields__})


# Names used literally in the VBA source remain available to the port.
Flow_properties = FlowProperties
GasPropertiesVBA = GasProperties
Discharge_properties = DischargeProperties
Equation_Coef = EquationCoef
