"""Convert editable workbook-shaped input data to the VBA model's mutable state."""

from __future__ import annotations

import math
from typing import Any

from .state import (Atmosphere, Controls, DischargeProperties, DischargesList,
                    EquationCoef, FlowProperties, GasProperties, Grid, PropertiesList,
                    PropertyState, zeros)
from .workbook import PROPERTIES


class ModelInputError(ValueError):
    """An input cannot be run by the workbook's Channel macro."""


# Convert an Excel-like cell value to a finite Double while preserving blank-as-zero behavior.
def _float(value: Any, label: str) -> float:
    # Excel's empty B-cell is an Empty Variant, coerced to zero when assigned
    # to a VBA Double. Label-presence validation is separate from its value.
    if value is None or value == "":
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelInputError(f"{label} requires a number") from exc
    if not math.isfinite(result):
        raise ModelInputError(f"{label} requires a finite number")
    return result


# VBA Integer conversion rounds half values to even and has a signed 16-bit range.
def _int(value: Any, label: str) -> int:
    number = _float(value, label)
    # Assignment to a VBA Integer rounds halves to even (CInt semantics).
    result = round(number)
    if result < -32768 or result > 32767:
        raise OverflowError(f"VBA Integer overflow in {label}")
    return result


# Use the macro's default only when the input cell is absent or blank.
def _optional_float(value: Any, default: float = 0.0) -> float:
    return default if value is None or value == "" else _float(value, "input")


# Convert days to seconds once, then validate the selected time and space schemes.
def build_controls(data: dict[str, Any]) -> Controls:
    # Entered duration is in days; the numerical clock and source rates use
    # seconds, so convert before the time loop begins.
    c = Controls()
    c.dt = _float(data.get("time_step_seconds"), "Time step")
    c.SimulationDuration = _float(data.get("simulation_duration_days"), "Duration") * 24 * 3600
    c.DtPrint = _float(data.get("print_interval_seconds"), "Print interval")
    if c.DtPrint == 0:
        raise ZeroDivisionError("VBA divides duration by the print interval")
    if c.SimulationDuration > 0 and c.dt <= 0:
        raise ModelInputError("A nonpositive time step would leave the VBA time loop running forever")
    time_method = str(data.get("time_method", ""))
    if time_method not in ("exp", "EXP", "Exp", "imp", "IMP", "Imp", "semi", "Semi", "SEMI"):
        raise ModelInputError("Time method must be exp, imp, or semi")
    c.TimeDiscretisation = time_method.lower()
    c.NumberOfOutputs = round(c.SimulationDuration / c.DtPrint)
    if c.NumberOfOutputs > 32767 or c.NumberOfOutputs < -32768:
        raise OverflowError("VBA Integer overflow in NumberOfOutputs")
    c.Advection = bool(data.get("advection", False))
    c.Diffusion = bool(data.get("diffusion", False))
    if c.Advection:
        raw_adv = str(data.get("advection_type", ""))
        allowed = {"upwind": ("upwind", "Upwind", "UPWIND"),
                   "central": ("central", "Central", "CENTRAL"),
                   "quick": ("QUICK", "Quick", "quick"),
                   "quick_up": ("QUICK_UP", "Quick_up", "Quick_UP", "quick_up")}
        adv = next((key for key, variants in allowed.items() if raw_adv in variants), None)
        if adv is None:
            raise ModelInputError("Advection type must be upwind, central, QUICK, or QUICK_UP")
        c.AdvType = "QUICK" if adv in ("quick", "quick_up") else adv
        c.QUICK_UP = adv == "quick_up"
        if c.QUICK_UP:
            if not data.get("quick_up_ratio_label_present", True):
                raise ModelInputError("QUICK_UP_Ratio: is required by the macro")
            c.QUICK_UP_Ratio = _optional_float(data.get("quick_up_ratio"))
    return c


# The one-dimensional finite-volume grid has NC equal cells, centred at (i - 0.5) dx.
def build_grid(data: dict[str, Any]) -> Grid:
    # Values live at cell centres: x_i = (i - 0.5) * dx.
    g = Grid()
    g.Length = _float(data.get("length_m"), "Channel length")
    g.NC = _int(data.get("cells"), "Number of cells")
    g.RiverWidth = _float(data.get("width_m"), "River width")
    g.waterdepth = _float(data.get("depth_m"), "Water depth")
    if g.NC < 1:
        raise ZeroDivisionError("VBA divides channel length by the cell count")
    g.dx = g.Length / g.NC
    g.AreaVertical = g.waterdepth * g.RiverWidth
    g.AreaH = g.RiverWidth * g.dx
    g.xc = zeros(g.NC)
    for i in range(1, g.NC + 1):
        g.xc[i] = g.dx * (i - 0.5)
    return g


# Calculate velocity from flow rate when requested; otherwise keep the entered velocity.
def build_flow(data: dict[str, Any], gr: Grid, ctrl: Controls) -> FlowProperties:
    f = FlowProperties()
    if data.get("velocity_mode") == "flow_formula":
        rate = _float(data.get("flow_rate_m3_s"), "River flow rate")
        f.velocity = rate / (gr.RiverWidth * gr.waterdepth)
    else:
        f.velocity = _float(data.get("velocity_m_s"), "Flow velocity")
    diffusivity = data.get("diffusivity_m2_s")
    if isinstance(diffusivity, str) and diffusivity in ("standard", "Standard", "std", "STANDARD"):
        f.Diffusivity = 0.1 * abs(f.velocity) * gr.RiverWidth
    else:
        f.Diffusivity = _float(diffusivity, "Diffusivity")
    f.Riverslope = _float(data.get("slope_percent"), "River slope") / 100
    f.CourantNr = ctrl.dt * f.velocity / gr.dx if ctrl.Advection else 0.0
    f.DiffusionNr = ctrl.dt * f.Diffusivity / gr.dx ** 2 if ctrl.Diffusion else 0.0
    return f


# Fill every cell with its baseline first, then overwrite CELL or INTERVAL selections.
def _initialise(prop: PropertyState, init: dict[str, Any], gr: Grid) -> None:
    # CELL and INTERVAL are overrides of the baseline rather than zero-filled
    # sparse fields; unspecified cells retain the default concentration.
    default_cell = init.get("default_cell_value")
    initial_value = (_optional_float(default_cell) if init.get("default_cell_label_present", default_cell is not None)
                     else prop.Default)
    for i in range(1, gr.NC + 1):
        prop.NewValue[i] = initial_value
    kind = str(init.get("type", "DEFAULT")).upper()
    if kind == "DEFAULT":
        return
    if kind == "CELL":
        cells = init.get("cells") or []
        if not cells:
            raise ModelInputError(f"{prop.SheetName}: CELL initialization needs at least one cell")
        for entry in cells:
            cell = _int(entry.get("cell"), "Initial cell")
            if not 1 <= cell <= gr.NC:
                raise ModelInputError(f"{prop.SheetName}: cell {cell} outside 1..{gr.NC}")
            prop.NewValue[cell] = _optional_float(entry.get("value"))
        return
    if kind == "INTERVAL":
        intervals = init.get("intervals") or []
        if not intervals:
            raise ModelInputError(f"{prop.SheetName}: INTERVAL initialization needs at least one interval")
        interval_kind = str(init.get("interval_kind", "CELL")).upper()
        previous_end: float = 0
        for entry in intervals:
            if previous_end >= (gr.Length if interval_kind == "M" else gr.NC):
                break
            start = _float(entry.get("start"), "Interval start")
            end = _float(entry.get("end"), "Interval end")
            value = _float(entry.get("value"), "Interval value")
            if interval_kind == "M":
                if start < 0 or end > gr.Length or start < previous_end or end < start:
                    raise ModelInputError(f"{prop.SheetName}: invalid or overlapping metre interval")
                # VBA integer division rounds both operands to whole numbers
                # first, then divides the rounded integers.
                cell = round(start + gr.dx / 2) // round(gr.dx) + 1
                if not 1 <= cell <= gr.NC:
                    raise ModelInputError(f"{prop.SheetName}: interval begins outside grid")
                prop.NewValue[cell] = value
                x = start
                while x < end:
                    if cell > gr.NC:
                        raise ModelInputError(f"{prop.SheetName}: interval exceeds grid")
                    prop.NewValue[cell] = value
                    cell += 1
                    x += gr.dx
            else:
                first = _int(start, "Interval first cell")
                last = _int(end, "Interval last cell")
                # VBA compares the last cell with gr.Length, not gr.NC.
                if first < 1 or last > gr.Length or first < previous_end or last < first:
                    raise ModelInputError(f"{prop.SheetName}: invalid or overlapping cell interval")
                if last > gr.NC:
                    raise ModelInputError(f"{prop.SheetName}: interval exceeds cell array")
                for cell in range(first, last + 1):
                    prop.NewValue[cell] = value
                end = last
            previous_end = end
        return
    raise ModelInputError(f"{prop.SheetName}: unknown initialization method {kind}")


# Attach only the lateral sources belonging to this property, using one-based cell indices.
def _discharges_for(name: str, config: dict[str, Any], gr: Grid) -> DischargesList:
    result = DischargesList()
    rows = []
    for row in config.get("discharges") or []:
        if _int(row.get("id"), "Discharge ID") <= 0:
            break
        rows.append(row)
    result.NumberOfDischarges = len(rows)
    result.Discharge = [DischargeProperties()] + [DischargeProperties() for _ in rows]
    if not rows:
        return result
    use_cells = bool(config.get("discharge_cells_label_present", True))
    for i, row in enumerate(rows, start=1):
        item = result.Discharge[i]
        item.name = str(row.get("name") or "")
        if use_cells:
            item.cell = _int(row.get("cell"), "Discharge cell")
        else:
            item.cell = round(_float(row.get("location_km"), "Discharge location") * 1000 / gr.dx)
        if item.cell > gr.NC:
            result.NumberOfDischarges = i - 1
            break
        item.VolumeRate = _optional_float(row.get("flow_rate_m3_s"))
        item.SpecificValue = _optional_float(row.get(name))
    result.DischargesActive = bool(config.get("discharge_property_labels_present", {}).get(name, True))
    return result


# Translate one editor property into the VBA-shaped mutable arrays and boundary record.
def build_property(name: str, data: dict[str, Any], config: dict[str, Any],
                   gr: Grid, ctrl: Controls) -> PropertyState:
    p = PropertyState()
    p.SheetName = name
    p.name = str(data.get("name_cell") or "") if data.get("name_label_matches_vba") else name
    p.Active = bool(data.get("active", False))
    p.NewValue = zeros(gr.NC)
    p.Old = zeros(gr.NC)
    if name == "BOD":
        p.IntegratedAerobicResp = zeros(gr.NC)
        p.IntegratedAnaerobicResp = zeros(gr.NC)
    # On a present NO/blank PropertyActive label, VBA exits before all later
    # fields (including units and initialization), even for the Generic sheet.
    if data.get("active_label_present", True) and not p.Active:
        return p
    p.units = str(data.get("units") or "")
    p.Default = _optional_float(data.get("default"))
    p.maximum = _optional_float(data.get("maximum"))
    p.minimum = _optional_float(data.get("minimum"))
    p.MaximumActive = bool(data.get("maximum_label_present", True))
    # Matches CreateProperty's later assignment, including its label-presence bug.
    p.MinimumActive = bool(data.get("cyclic_label_present", False))
    p.Boundary.CyclicBoundary = bool(data.get("cyclic", False))
    p.Boundary.FreeSurfaceFlux = bool(data.get("free_surface_flux", False))
    p.Boundary.FreeSurfaceSensibleHeatFlux = bool(data.get("sensible_heat", True))
    p.Boundary.FreeSurfaceLatentHeatFlux = bool(data.get("latent_heat", True))
    p.Boundary.FreeSurfaceRadiativeHeatFlux = bool(data.get("radiative_heat", True))
    if ctrl.Advection:
        p.Boundary.LeftValue = _float(data.get("left_boundary"), f"{name} left boundary")
        p.Boundary.RightValue = _float(data.get("right_boundary"), f"{name} right boundary")
    p.Boundary.PropDischarges = _discharges_for(name, config, gr)
    if name == "Generic":
        rate = _optional_float(data.get("generic_rate"))
        units = data.get("generic_rate_units")
        if not data.get("generic_rate_label_present", True):
            p.Decayrate = 0.0
        elif units == "T90(Hours)":
            p.Decayrate = math.log(0.1) / (rate * 3600)
        elif units == "HalfLife(Days)":
            p.Decayrate = math.log(0.5) / (rate * 86400)
        elif units == "TDuplicate(Days)":
            p.Decayrate = math.log(2) / (rate * 86400)
        elif units == "Day-1":
            p.Decayrate = rate / 86400
        elif units == "RT":
            p.Decayrate = (rate - 1) / 7 / 86400
        else:
            raise ModelInputError("Generic rate units are not recognized by the macro")
        p.DecayOrGrowth = (p.Decayrate != 0 if data.get("generic_rate_label_present", True) else True)
    if name == "BOD":
        if not data.get("bod_decay_label_present", True):
            raise ModelInputError("BOD Decay_Rate: is required by the macro")
        p.Decayrate = _optional_float(data.get("bod_decay_per_day")) / 86400
        p.DecayOrGrowth = p.Decayrate != 0
        p.GrowthRate = _optional_float(data.get("bod_growth_per_day")) / 86400
        p.MaxValLogisticEquation = (_optional_float(data.get("bod_logistic_max"))
                                    if data.get("bod_logistic_max_label_present", True) else 10000000)
        p.GrazingKsat = _optional_float(data.get("bod_oxygen_half_saturation"))
        p.AnaerobicRespiration = bool(data.get("anaerobic_respiration", False))
    _initialise(p, data.get("initial") or {}, gr)
    return p


# The gas tables are copied into active property boundaries, as in CreateAtmosphere.
def build_atmosphere(data: dict[str, Any], pr: PropertiesList) -> Atmosphere:
    atm = Atmosphere()
    atm.Temperature = _float(data.get("air_temperature_c"), "Air temperature")
    atm.windspeed = _float(data.get("wind_speed_m_s"), "Wind speed")
    atm.h_min = (_optional_float(data.get("minimum_heat_transfer_coefficient"))
                 if data.get("h_min_label_present", True) else 6.9)
    atm.Humidity = _float(data.get("humidity_percent"), "Humidity") / 100
    atm.SolarConstant = _float(data.get("solar_constant_w_m2"), "Solar constant")
    atm.Latitude = _float(data.get("latitude_degrees"), "Latitude")
    atm.SkyTemperature = _optional_float(data.get("sky_temperature_c"))
    atm.SkyTemperatureImposed = bool(data.get("sky_temperature_imposed", False))
    atm.CloudCover = _float(data.get("cloud_cover_percent"), "Cloud cover") / 100
    atm.Tsr = _float(data.get("sunrise_hour"), "Sunrise hour")
    atm.Tss = _float(data.get("sunset_hour"), "Sunset hour")
    if not data.get("o2_pressure_label_present", True) and not data.get("co2_pressure_label_present", True):
        return atm
    rows = data.get("henry_constants") or []
    if not rows:
        raise ModelInputError("At least one Henry constant row is required")
    header_order = data.get("henry_header_order", ["O2", "CO2"])
    if set(header_order) != {"O2", "CO2"} or len(header_order) != 2:
        raise ModelInputError("Henry columns must be labeled O2 and CO2")
    counts: dict[str, int] = {}
    for label in ("O2", "CO2"):
        count = 0
        for row in rows:
            if row.get(label.lower()) is None:
                break
            count += 1
        counts[label] = count
    # The VBA stores gas labels in pressure order but assigns Henry lengths by
    # table header position. That produces a real swapped-header quirk.
    o2_pressure = _float(data.get("o2_partial_pressure_bar"), "O2 partial pressure")
    co2_pressure = (o2_pressure * 0.3 / 159
                    if data.get("co2_pressure_mode") == "o2_formula" else
                    _float(data.get("co2_partial_pressure_bar"), "CO2 partial pressure"))
    atm.Gas = [GasProperties(),
               GasProperties(Label="O2", partialpressure=o2_pressure),
               GasProperties(Label="CO2", partialpressure=co2_pressure)]
    for position, label in enumerate(header_order, start=1):
        atm.Gas[position].N_HenryConstants = counts[label]
    for position in (1, 2):
        gas = atm.Gas[position]
        gas.HenryConstValues = [[0.0] * (gas.N_HenryConstants + 1) for _ in range(3)]
        source_label = "O2" if position == 1 else "CO2"
        for i in range(1, gas.N_HenryConstants + 1):
            gas.HenryConstValues[1][i] = _float(rows[i - 1].get("temperature_c"), "Henry temperature")
            gas.HenryConstValues[2][i] = _optional_float(rows[i - 1].get(source_label.lower()))
    for label, target, weight in (("O2", pr.DO, 32000), ("CO2", pr.CO2, 44000)):
        if target.Active:
            position = header_order.index(label) + 1
            source = atm.Gas[position]
            target.Boundary.ValuesForFreeSurfaceGasExchange = GasProperties(
                Label=source.Label, partialpressure=source.partialpressure,
                N_HenryConstants=source.N_HenryConstants,
                HenryConstValues=[row.copy() for row in source.HenryConstValues],
                MolecularWeight=weight)
    return atm


# Initialization order matters: controls, grid, flow, coefficients, properties, then atmosphere.
def build_state(config: dict[str, Any]) -> tuple[Controls, Grid, FlowProperties,
                                                 PropertiesList, Atmosphere, EquationCoef]:
    """Reproduce the ``Channel`` initialization call sequence."""
    ctrl = build_controls(config.get("main") or {})
    gr = build_grid(config.get("river") or {})
    flow = build_flow(config.get("river") or {}, gr, ctrl)
    coef = EquationCoef.allocate(gr.NC)
    pr = PropertiesList()
    for name in PROPERTIES:
        setattr(pr, name, build_property(name, (config.get("properties") or {}).get(name) or {}, config, gr, ctrl))
    atm = build_atmosphere(config.get("atmosphere") or {}, pr)
    if pr.DO.Active and not pr.Temperature.Active:
        raise ModelInputError("DO active requires Temperature active")
    if pr.BOD.Active and not pr.DO.Active:
        raise ModelInputError("BOD active requires DO active")
    if pr.CO2.Active and not (pr.BOD.Active or pr.DO.Active):
        raise ModelInputError("CO2 active requires BOD or DO active")
    if (ctrl.Advection and ctrl.AdvType == "QUICK"
            and ctrl.TimeDiscretisation != "imp" and gr.NC < 4):
        raise ModelInputError("QUICK requires at least four cells")
    if gr.NC < 2 and ctrl.TimeDiscretisation != "imp" and ctrl.SimulationDuration > 0:
        raise ModelInputError("Explicit and semi implicit transport require at least two cells")
    return ctrl, gr, flow, pr, atm, coef
