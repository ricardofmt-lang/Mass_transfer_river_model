"""The VBA ``Channel`` time loop and its complete printed numerical output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import build_state
from .sources import (ApplyDischarges, ExpFreeSurfaceGasFluxes,
                      ExpFreeSurfaceHeatFluxes, ExpSinksBOD, ExpSinksGeneric,
                      ImpTypeSinksGeneric, CellCalculationError)
from .transport import coef_transport, transport, zero_coeffs
from .workbook import PROPERTIES


class SimulationError(RuntimeError):
    """A failed model calculation with its position in the time loop.

    ``__cause__`` retains the original Python exception and traceback for
    diagnosis. The attributes are safe to show beside a plain-language error.
    """

    def __init__(
        self, *, stage: str, property_name: str | None, step: int | None,
        time_seconds: float | None, cell: int | None, detail: str,
        hint: str | None, original_type: str, original_message: str,
    ) -> None:
        self.stage = stage
        self.property_name = property_name
        self.step = step
        self.time_seconds = time_seconds
        self.cell = cell
        self.detail = detail
        self.hint = hint
        self.original_type = original_type
        self.original_message = original_message
        where = f"{stage} for {property_name}" if property_name else stage
        if cell is not None:
            where += f" in cell {cell}"
        if step is not None and time_seconds is not None:
            where += f" at step {step} (t = {time_seconds:g} s)"
        super().__init__(f"{where} failed: {detail}")


def _cell_in_traceback(exc: Exception, cell_count: int) -> int | None:
    """Recover a model cell from a failed one-based calculation loop."""
    if cell_count <= 0:
        return None
    cell = None
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__", "").startswith("transportmodel."):
            discharge = frame.f_code.co_name == "ApplyDischarges"
            candidate = (frame.f_locals.get("ncell") if discharge
                         else frame.f_locals.get("i"))
            if isinstance(candidate, int) and (discharge or 1 <= candidate <= cell_count):
                cell = candidate
        traceback = traceback.tb_next
    return cell


def _simulation_error(
    exc: Exception, *, stage: str, property_name: str | None,
    step: int | None, time_seconds: float | None, cell_count: int,
) -> SimulationError:
    source = exc
    while source.__cause__ is not None and isinstance(source.__cause__, Exception):
        source = source.__cause__
    if isinstance(exc, CellCalculationError):
        stage = exc.stage
        cell = exc.cell
        detail = exc.detail
    else:
        cell = _cell_in_traceback(exc, cell_count)
        detail = str(exc)
        if isinstance(source, ZeroDivisionError) and stage == "River diagnostics":
            detail = "Residence time could not be calculated because velocity + 0.000001 m/s is zero."
        elif isinstance(source, OverflowError):
            detail = "A calculation exceeded the range of floating-point numbers."
        elif not detail:
            detail = f"{type(source).__name__} occurred."
    if isinstance(source, OverflowError) and stage == "Latent heat exchange":
        hint = ("The saturation vapour-pressure formula overflowed at the reported "
                "water or air temperature. Check temperature values and units in "
                "the initial distribution, boundaries, discharges, and atmosphere. "
                "If the water temperature changed sharply before this step, review "
                "the time step and heat-flux inputs.")
    elif isinstance(source, OverflowError) and stage == "BOD temperature acceleration":
        hint = ("The BOD reaction-rate temperature factor overflowed. Check the "
                "water temperature and its units in the initial distribution, "
                "boundaries, discharges, and heat-exchange inputs.")
    elif isinstance(source, OverflowError):
        hint = ("Check the initial, boundary, and discharge values for this property. "
                "If values grow rapidly during the run, reduce the time step or "
                "use a more stable transport method.")
    elif isinstance(source, ZeroDivisionError) and stage == "River diagnostics":
        hint = "Check the river velocity; -0.000001 m/s makes the residence-time denominator zero."
    elif isinstance(source, ZeroDivisionError):
        hint = "Check for a zero depth, zero rate denominator, or a singular transport coefficient."
    elif isinstance(source, IndexError):
        hint = "Check that discharge cells and initial-distribution cells lie inside the river grid."
    else:
        hint = None
    return SimulationError(
        stage=stage, property_name=property_name, step=step,
        time_seconds=time_seconds, cell=cell, detail=detail, hint=hint,
        original_type=type(source).__name__, original_message=str(source),
    )


@dataclass
class SimulationResult:
    """Values from rows 26 onward, with coordinates from row 23 of Excel."""

    time_seconds: list[float] = field(default_factory=list)
    coordinates_m: list[float] = field(default_factory=list)
    values: dict[str, list[list[float]]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_seconds": self.time_seconds,
            "coordinates_m": self.coordinates_m,
            "values": self.values,
            "diagnostics": self.diagnostics,
        }


def run_model(config: dict[str, Any], *, max_cells_times_steps: int | None = None) -> SimulationResult:
    """Run the exact Channel sequence with the same print-time threshold.

    ``max_cells_times_steps`` is an optional caller limit; the app leaves it
    unset so it does not narrow the workbook's input range.
    """
    try:
        ctrl, gr, flow, pr, atm, coef = build_state(config)
    except Exception as exc:
        raise _simulation_error(
            exc, stage="Input setup", property_name=None, step=None,
            time_seconds=None, cell_count=0,
        ) from exc
    try:
        steps = int(ctrl.SimulationDuration // ctrl.dt) + 1 if ctrl.SimulationDuration > 0 else 0
    except Exception as exc:
        raise _simulation_error(
            exc, stage="Time-step planning", property_name=None,
            step=0, time_seconds=ctrl.T, cell_count=gr.NC,
        ) from exc
    if max_cells_times_steps is not None and gr.NC * steps > max_cells_times_steps:
        raise ValueError("This run is too large for the local app; reduce cells or duration")
    result = SimulationResult(coordinates_m=gr.xc[1:])
    active = [name for name in PROPERTIES if getattr(pr, name).Active]
    result.values = {name: [] for name in active}
    integrated_print = pr.BOD.Active and pr.BOD.name == "BOD"
    if integrated_print:
        result.values["IntegratedAeroResp(kg)"] = []
        result.values["IntegratedAnaeroResp(kg)"] = []

    def print_values() -> None:
        result.time_seconds.append(ctrl.T)
        for name in active:
            result.values[name].append(getattr(pr, name).NewValue[1:].copy())
        if integrated_print:
            result.values["IntegratedAeroResp(kg)"].append(pr.BOD.IntegratedAerobicResp[1:].copy())
            result.values["IntegratedAnaeroResp(kg)"].append(pr.BOD.IntegratedAnaerobicResp[1:].copy())

    try:
        print_values()  # The macro always prints time zero, even for a zero-duration run.
    except Exception as exc:
        raise _simulation_error(
            exc, stage="Initial output", property_name=None, step=0,
            time_seconds=ctrl.T, cell_count=gr.NC,
        ) from exc
    ctrl.Printnumber += 1
    ctrl.printtime += ctrl.DtPrint
    ctrl.PrintLine += 1

    step = 0

    def execute(stage: str, property_name: str | None, action: Any, *args: Any) -> Any:
        try:
            return action(*args)
        except Exception as exc:
            raise _simulation_error(
                exc, stage=stage, property_name=property_name, step=step,
                time_seconds=ctrl.T, cell_count=gr.NC,
            ) from exc

    while ctrl.T < ctrl.SimulationDuration:
        step += 1
        ctrl.T += ctrl.dt
        ctrl.timedays = ctrl.T / 86400
        execute("Transport coefficient reset", None, zero_coeffs, coef, gr)
        for name in PROPERTIES:
            prop = getattr(pr, name)
            if prop.Active and prop.Boundary.PropDischarges.DischargesActive:
                execute("Discharge mixing", name, ApplyDischarges, prop, gr, ctrl.dt)
            if name == "Temperature" and prop.Active and prop.Boundary.CyclicBoundary:
                try:
                    if flow.velocity > 0:
                        prop.Boundary.LeftValue = prop.NewValue[gr.NC]
                    else:
                        prop.Boundary.RightValue = prop.NewValue[1]
                except Exception as exc:
                    raise _simulation_error(
                        exc, stage="Cyclic boundary update", property_name=name,
                        step=step, time_seconds=ctrl.T, cell_count=gr.NC,
                    ) from exc
        execute("Transport coefficient calculation", None,
                coef_transport, flow, gr.NC, ctrl, coef)
        for name in PROPERTIES:
            prop = getattr(pr, name)
            if prop.Active:
                execute("Transport", name, transport, prop, coef, gr.NC, ctrl)
        if pr.Temperature.Active and pr.Temperature.Boundary.FreeSurfaceFlux:
            execute("Surface heat exchange", "Temperature", ExpFreeSurfaceHeatFluxes,
                    pr.Temperature, gr, ctrl, atm)
        if pr.DO.Active and pr.DO.Boundary.FreeSurfaceFlux:
            execute("Surface gas exchange", "DO", ExpFreeSurfaceGasFluxes,
                    pr.Temperature, pr.DO, gr, ctrl, flow, atm)
        if pr.CO2.Active and pr.CO2.Boundary.FreeSurfaceFlux:
            execute("Surface gas exchange", "CO2", ExpFreeSurfaceGasFluxes,
                    pr.Temperature, pr.CO2, gr, ctrl, flow, atm)
        if pr.BOD.Active:
            execute("BOD reactions", "BOD", ExpSinksBOD, pr, gr, ctrl.dt)
        if pr.Generic.Active and pr.Generic.DecayOrGrowth:
            if ctrl.TimeDiscretisation == "exp":
                execute("Generic growth or decay", "Generic", ExpSinksGeneric,
                        pr.Generic, gr.NC, ctrl.dt)
            else:
                execute("Generic growth or decay", "Generic", ImpTypeSinksGeneric,
                        pr.Generic, gr.NC, ctrl)
        if ctrl.T >= ctrl.printtime:
            execute("Output table", None, print_values)
            if ctrl.PrintLine >= 32767:
                exc = OverflowError("VBA Integer overflow in PrintLine")
                raise _simulation_error(
                    exc, stage="Output table", property_name=None,
                    step=step, time_seconds=ctrl.T, cell_count=gr.NC,
                ) from exc
            ctrl.Printnumber += 1
            ctrl.printtime += ctrl.DtPrint
            ctrl.PrintLine += 1

    try:
        result.diagnostics = {
            "courant_number": round(flow.CourantNr, 2),
            "diffusion_number": round(flow.DiffusionNr, 2),
            "grid_reynolds_number": (round(flow.CourantNr / flow.DiffusionNr, 2)
                                      if flow.DiffusionNr > 0 else "infinity"),
            "residence_time_river_days": round(gr.Length / (flow.velocity + 0.000001) / 3600 / 24, 1),
            "residence_time_cell_seconds": round(gr.dx / (flow.velocity + 0.000001), 2),
        }
    except Exception as exc:
        raise _simulation_error(
            exc, stage="River diagnostics", property_name=None,
            step=step, time_seconds=ctrl.T, cell_count=gr.NC,
        ) from exc
    return result
