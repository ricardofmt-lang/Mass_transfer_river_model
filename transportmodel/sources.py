"""Literal port of the river workbook's discharge and source/sink routines.

The callable names and one-based indexing follow ``Original VBA macro 1.txt``.
Several formulae differ from the accompanying PDF; these routines reproduce the
executed VBA, including the order of updates within each time step.
"""

from __future__ import annotations

from math import cos, exp, sin, trunc

from .state import Atmosphere, Controls, FlowProperties, Grid, PropertiesList, PropertyState


class CellCalculationError(ArithmeticError):
    """A failed source term with its one-based river cell and calculation."""

    def __init__(self, stage: str, cell: int, detail: str) -> None:
        self.stage = stage
        self.cell = cell
        self.detail = detail
        super().__init__(detail)


# Mix a lateral flow into one cell by volume balance before longitudinal transport.
def ApplyDischarges(Prop: PropertyState, gr: Grid, dt: float) -> None:
    """Apply each continuous discharge as a sequential cell-volume mix (VBA 933–949)."""
    discharges = Prop.Boundary.PropDischarges
    # Numerator is old river content plus incoming content; denominator is
    # the combined water volume during this time step.
    for i in range(1, discharges.NumberOfDischarges + 1):
        discharge = discharges.Discharge[i]
        ncell = discharge.cell
        if ncell < 1 or ncell > gr.NC:
            raise CellCalculationError(
                "Discharge mixing", ncell,
                f"Discharge {i} targets cell {ncell}, outside river cells 1 to {gr.NC}.",
            )
        volume_rate = discharge.VolumeRate
        specific_value = discharge.SpecificValue
        try:
            Prop.NewValue[ncell] = (
                Prop.NewValue[ncell] * (gr.AreaVertical * gr.dx)
                + (volume_rate * specific_value * dt)
            ) / (gr.AreaVertical * gr.dx + volume_rate * dt)
        except ArithmeticError as exc:
            raise CellCalculationError(
                "Discharge mixing", ncell,
                f"Discharge {i} failed while mixing at cell {ncell}: "
                f"flow = {volume_rate:g} m³/s, property value = {specific_value:g}, "
                f"time step = {dt:g} s. {exc}",
            ) from exc


# The empirical heat-transfer coefficient rises quadratically with wind speed.
def h(windspeed: float, h_min: float) -> float:
    """Surface heat exchange coefficient (VBA 1090–1092)."""
    return h_min + 0.345 * windspeed**2


# Compute saturation vapour pressure from temperature using the macro's Magnus expression.
def Es(T: float, *, source: str = "water") -> float:
    """Saturation vapour pressure in mbar (VBA 1093–1103)."""
    a = 6.112
    b = 17.67
    c = 243.5
    try:
        aux = b * T / (T + c)
    except ZeroDivisionError as exc:
        raise ZeroDivisionError(
            f"Saturation vapour pressure for {source} temperature {T:g} °C "
            "is undefined because T + 243.5 °C is zero."
        ) from exc
    try:
        return a * exp(aux)
    except OverflowError as exc:
        raise OverflowError(
            f"Saturation vapour pressure for {source} temperature {T:g} °C "
            f"overflowed while evaluating exp({aux:g})."
        ) from exc


# Net long-wave heat is sky emission minus water emission.
def FluxRadiative(watertemperature: float, atm: Atmosphere) -> float:
    """Net long-wave radiation, with optional imposed sky temperature (VBA 1104–1115)."""
    epson = 0.97
    sigma = 5.67 * 10**-8
    if atm.SkyTemperatureImposed:
        return epson * sigma * (
            (atm.SkyTemperature + 273.15) ** 4
            - (watertemperature + 273.15) ** 4
        )
    return epson * sigma * (
        (
            9.37
            * 10**-6
            * (atm.Temperature + 273.15) ** 6
            * (1 + 0.17 * atm.CloudCover**2)
        )
        - (watertemperature + 273.15) ** 4
    )


# A daylight sine curve, latitude, absorption, and cloud factor determine short-wave input.
def Solar_Radiation(timedays: float, atm: Atmosphere) -> float:
    """Daily sine solar forcing at the VBA's end-of-step time (VBA 997–1014)."""
    pi = 3.14159265358979
    atmospheric_absorption = 0.23
    im = atm.SolarConstant * cos(atm.Latitude * pi / 180) * (
        1 - atmospheric_absorption
    )
    # VBA Fix truncates toward zero, unlike Python's floor for negative values.
    hour = (timedays - trunc(timedays)) * 24
    if hour < atm.Tsr or hour > atm.Tss:
        return 0.0
    return (1 - 0.75 * atm.CloudCover**3) * im * sin(
        pi * (hour - atm.Tsr) / (atm.Tss - atm.Tsr)
    )


# Solar, sensible, latent, and long-wave terms update temperature sequentially.
def ExpFreeSurfaceHeatFluxes(
    Temperature: PropertyState, gr: Grid, ctrl: Controls, atm: Atmosphere
) -> None:
    """Apply solar, sensible, latent, then long-wave heat (VBA 950–996).

    The outer ``FreeSurfaceFlux`` switch is checked by ``Channel``, not here.
    Each successive flux observes the temperature updated by earlier fluxes.
    """
    ro = 1000
    cp = 4180
    cb = 0.62

    for i in range(1, gr.NC + 1):
        try:
            solar_flux = Solar_Radiation(ctrl.timedays, atm)
            Temperature.NewValue[i] = Temperature.NewValue[i] + ctrl.dt * solar_flux / (
                ro * cp * gr.waterdepth
            )
        except (ArithmeticError, ValueError) as exc:
            raise CellCalculationError(
                "Solar heat exchange", i,
                f"Solar heat calculation failed at cell {i} and "
                f"t = {ctrl.T:g} s: {exc}",
            ) from exc

    if Temperature.Boundary.FreeSurfaceSensibleHeatFlux:
        for i in range(1, gr.NC + 1):
            try:
                flux_conv = cb * h(atm.windspeed, atm.h_min) * (
                    atm.Temperature - Temperature.NewValue[i]
                )
                Temperature.NewValue[i] = Temperature.NewValue[i] + ctrl.dt * flux_conv / (
                    ro * cp * gr.waterdepth
                )
            except (ArithmeticError, ValueError) as exc:
                raise CellCalculationError(
                    "Sensible heat exchange", i,
                    f"Sensible heat calculation failed at cell {i}: "
                    f"water temperature = {Temperature.NewValue[i]:g} °C, "
                    f"air temperature = {atm.Temperature:g} °C, "
                    f"wind speed = {atm.windspeed:g} m/s. {exc}",
                ) from exc

    if Temperature.Boundary.FreeSurfaceLatentHeatFlux:
        for i in range(1, gr.NC + 1):
            try:
                flux_latent = -h(atm.windspeed, atm.h_min) * (
                    Es(Temperature.NewValue[i], source="water")
                    - atm.Humidity * Es(atm.Temperature, source="air")
                )
            except (ArithmeticError, ValueError) as exc:
                raise CellCalculationError(
                    "Latent heat exchange", i,
                    f"{exc} Water temperature = {Temperature.NewValue[i]:g} °C; "
                    f"air temperature = {atm.Temperature:g} °C.",
                ) from exc
            # VBA suppresses heat gain from condensation.
            if flux_latent > 0:
                flux_latent = 0.0
            Temperature.NewValue[i] = Temperature.NewValue[i] + ctrl.dt * flux_latent / (
                ro * cp * gr.waterdepth
            )

    if Temperature.Boundary.FreeSurfaceRadiativeHeatFlux:
        for i in range(1, gr.NC + 1):
            try:
                flux_radiation = FluxRadiative(Temperature.NewValue[i], atm)
                Temperature.NewValue[i] = Temperature.NewValue[i] + ctrl.dt * flux_radiation / (
                    ro * cp * gr.waterdepth
                )
            except (ArithmeticError, ValueError) as exc:
                raise CellCalculationError(
                    "Radiative heat exchange", i,
                    f"Long-wave radiation calculation failed at cell {i}: "
                    f"water temperature = {Temperature.NewValue[i]:g} °C. {exc}",
                ) from exc


# This helper exists for VBA parity but the executed gas routine does not call it.
def KGas(watertemperature: float, velocity: float, depth: float) -> float:
    """The alternative VBA gas coefficient (1040–1046), unused by ``Channel``."""
    k = 3.9 * ((abs(velocity) + 0.001) / depth) ** 0.5
    return k * 1.024 ** (watertemperature - 20)


# This historical helper is present but not used by the executed BOD routine.
def Kd_BOD(Oxygen: float, BOD: PropertyState, Temperature: float) -> float:
    """The VBA BOD-rate helper (1048–1058), unused by ``ExpSinksBOD``."""
    if Oxygen > 0:
        kd = BOD.Decayrate * (Oxygen / (BOD.GrazingKsat + Oxygen + 0.01))
        return kd * 1.047 ** (Temperature - 20)
    return 0.0


# Interpolate Henry's constant at water temperature, then multiply by pressure and molecular weight.
def getCsat_Henry(Prop: PropertyState, WaterTemp: float) -> float:
    """Interpolate or clamp Henry saturation data (VBA 1060–1089).

    ``HenryConstValues`` uses VBA-style row 1 for temperature and row 2 for
    Henry constant, with index zero unused in each row. The source's strict
    ``>`` comparison means a table-knot temperature uses that knot directly.
    """
    gas = Prop.Boundary.ValuesForFreeSurfaceGasExchange
    values = gas.HenryConstValues
    # Clamp outside the temperature table and interpolate between its knots.
    j = 1
    for _ in range(gas.N_HenryConstants):
        if WaterTemp > values[1][j]:
            j += 1

    if j == 1:
        return values[2][1] * gas.partialpressure * gas.MolecularWeight
    if j > gas.N_HenryConstants:
        return (
            values[2][gas.N_HenryConstants]
            * gas.partialpressure
            * gas.MolecularWeight
        )
    slope = (values[2][j] - values[2][j - 1]) / (
        values[1][j] - values[1][j - 1]
    )
    k_henry_interpolated = values[2][j - 1] + slope * (
        WaterTemp - values[1][j - 1]
    )
    return k_henry_interpolated * gas.partialpressure * gas.MolecularWeight


# Gas exchange relaxes each cell toward Henry saturation with an implicit one-cell update.
def ExpFreeSurfaceGasFluxes(
    Temperature: PropertyState,
    Gas: PropertyState,
    gr: Grid,
    ctrl: Controls,
    flow: FlowProperties,
    atm: Atmosphere,
) -> None:
    """Apply the active VBA implicit DO/CO2 free-surface exchange (1015–1037).

    The named ``KGas`` alternative and temperature multiplier in the PDF are
    not used by the VBA's active formula. Supersaturation accelerates exchange.
    """
    for i in range(1, gr.NC + 1):
        csat = getCsat_Henry(Gas, Temperature.NewValue[i])
        k_gas_exchange = (
            (1 / gr.waterdepth)
            * 0.142
            * (abs(atm.windspeed) + 0.1)
            * (flow.Riverslope + 0.00001)
        )
        # The workbook treats supersaturated water as faster to degas.
        if csat - Gas.NewValue[i] < 0:
            k_gas_exchange = k_gas_exchange * (
                1 + 20 * (Gas.NewValue[i] - csat) / (csat + Gas.NewValue[i])
            )
        Gas.NewValue[i] = (
            Gas.NewValue[i] + ctrl.dt * k_gas_exchange * csat
        ) / (1 + ctrl.dt * k_gas_exchange)


# An explicit first-order source changes Generic proportionally to its current value.
def ExpSinksGeneric(Generic: PropertyState, NC: int, dt: float) -> None:
    """Explicit generic growth/decay and optional floor (VBA 1116–1130)."""
    if Generic.MinimumActive:
        for i in range(1, NC + 1):
            Generic.NewValue[i] = max(
                Generic.NewValue[i] * (1 + Generic.Decayrate * dt), Generic.minimum
            )
    else:
        for i in range(1, NC + 1):
            Generic.NewValue[i] = Generic.NewValue[i] * (
                1 + Generic.Decayrate * dt
            )


# BOD consumption lowers DO, raises CO2, and records separate aerobic and anaerobic totals.
def ExpSinksBOD(pr: PropertiesList, gr: Grid, dt: float) -> None:
    """Couple BOD, DO, CO2 and accumulated respiration (VBA 1131–1187)."""
    for i in range(1, gr.NC + 1):
        if pr.BOD.NewValue[i] < pr.BOD.MaxValLogisticEquation:
            pr.BOD.NewValue[i] = pr.BOD.NewValue[i] * (
                1
                + dt
                * (pr.BOD.Decayrate + pr.BOD.GrowthRate)
                * (1 - pr.BOD.NewValue[i] / pr.BOD.MaxValLogisticEquation) ** 0.5
            )

        # Growth has already modified BOD; the subsequent reaction couples
        # aerobic BOD loss, oxygen consumption, and carbon-dioxide production.
        bod_before_decay = pr.BOD.NewValue[i]
        try:
            temperature_acceleration = 1.047 ** (pr.Temperature.NewValue[i] - 20)
        except OverflowError as exc:
            raise CellCalculationError(
                "BOD temperature acceleration", i,
                "The BOD temperature factor "
                f"1.047 ** ({pr.Temperature.NewValue[i]:g} - 20) exceeded "
                "the floating-point range.",
            ) from exc
        decay_rate = (
            temperature_acceleration
            * pr.BOD.Decayrate
            * (
                pr.DO.NewValue[i]
                / (pr.BOD.GrazingKsat + pr.DO.NewValue[i] + 0.01)
            )
        )
        bod_decay = pr.BOD.NewValue[i] * decay_rate * dt

        # Preserve the VBA conditions verbatim, including their +minimum tests.
        if bod_decay > bod_before_decay + pr.BOD.minimum:
            bod_decay = bod_before_decay - pr.BOD.minimum
        if bod_decay > pr.DO.NewValue[i] + pr.DO.minimum:
            bod_decay = pr.DO.NewValue[i] - pr.DO.minimum

        aerobic_decay_rate = bod_decay / pr.BOD.NewValue[i] / dt
        pr.BOD.NewValue[i] = bod_before_decay - bod_decay
        pr.DO.NewValue[i] = pr.DO.NewValue[i] - bod_decay
        pr.CO2.NewValue[i] = pr.CO2.NewValue[i] + bod_decay * (44 / 32)
        pr.BOD.IntegratedAerobicResp[i] = (
            pr.BOD.IntegratedAerobicResp[i]
            + bod_decay * gr.dx * gr.RiverWidth * gr.waterdepth / 1000
        )

        if pr.BOD.AnaerobicRespiration:
            anaerobic_decay_rate = (
                temperature_acceleration * pr.BOD.Decayrate - aerobic_decay_rate
            )
            aux = pr.BOD.NewValue[i]
            pr.BOD.NewValue[i] = pr.BOD.NewValue[i] / (
                1 + anaerobic_decay_rate * dt
            )
            anaerobic_decay = aux - pr.BOD.NewValue[i]
            pr.BOD.IntegratedAnaerobicResp[i] = (
                pr.BOD.IntegratedAnaerobicResp[i]
                + anaerobic_decay * gr.dx * gr.RiverWidth * gr.waterdepth / 1000
            )
            pr.CO2.NewValue[i] = pr.CO2.NewValue[i] + 0.5 * (
                anaerobic_decay * (44 / 32)
            )


# The generic source uses implicit or centred time weighting when transport is not explicit.
def ImpTypeSinksGeneric(Generic: PropertyState, NC: int, ctrl: Controls) -> None:
    """Implicit or semi-implicit generic source (VBA 1191–1211)."""
    for i in range(1, NC + 1):
        Generic.Old[i] = Generic.NewValue[i]
    if ctrl.TimeDiscretisation == "imp":
        for i in range(1, NC + 1):
            Generic.NewValue[i] = Generic.Old[i] / (1 - Generic.Decayrate * ctrl.dt)
    else:
        for i in range(1, NC + 1):
            Generic.NewValue[i] = Generic.NewValue[i] * (
                1 + Generic.Decayrate * ctrl.dt / 2
            ) / (1 - Generic.Decayrate * ctrl.dt / 2)
