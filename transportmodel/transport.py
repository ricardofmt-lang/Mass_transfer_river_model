"""Literal one-based port of the VBA transport routines.

These functions mutate the shared state, as the original ByRef procedures do.
The order of assignments, including the source's boundary indexing mistakes,
is intentional: it affects numerical parity with the macro.
"""

from __future__ import annotations

from .state import Controls, EquationCoef, FlowProperties, Grid, PropertyState


def zero_coeffs(coef: EquationCoef, grid: Grid) -> None:
    """Port of ``ZerosCoeffs``; the three ``*_up`` arrays are not cleared."""
    for i in range(1, grid.NC + 1):
        coef.A[i] = 0.0
        coef.B[i] = 0.0
        coef.e[i] = 0.0
        coef.f[i] = 0.0
        coef.g[i] = 0.0
        coef.L_LeftCoef[i] = 0.0
        coef.LeftCoef[i] = 0.0
        coef.CenterCoef[i] = 0.0
        coef.RigthCoef[i] = 0.0
        coef.R_RigthCoef[i] = 0.0
        coef.TiCoef[i] = 0.0


def coef_transport(flow: FlowProperties, nc: int, ctrl: Controls, coef: EquationCoef) -> None:
    """Port of ``CoefTransport``. Call ``zero_coeffs`` first each time step."""
    if ctrl.Diffusion:
        coef.A[1] = 0.0
        coef.B[1] = 0.0
        coef.e[1] = -flow.DiffusionNr
        coef.f[1] = flow.DiffusionNr
        coef.g[1] = 0.0

        coef.A[nc] = 0.0
        coef.B[nc] = flow.DiffusionNr
        coef.e[nc] = -flow.DiffusionNr
        coef.f[nc] = 0.0
        coef.g[nc] = 0.0

        for i in range(2, nc):
            coef.A[i] = 0.0
            coef.B[i] = flow.DiffusionNr
            coef.e[i] = -2 * flow.DiffusionNr
            coef.f[i] = flow.DiffusionNr
            coef.g[i] = 0.0

    if ctrl.Advection:
        if ctrl.AdvType == "upwind":
            if flow.velocity > 0:
                for i in range(1, nc + 1):
                    coef.B[i] = coef.B[i] + flow.CourantNr
                    coef.e[i] = coef.e[i] - flow.CourantNr
                    coef.f[i] = coef.f[i]
            else:
                for i in range(1, nc + 1):
                    coef.B[i] = coef.B[i]
                    coef.e[i] = coef.e[i] + flow.CourantNr
                    coef.f[i] = coef.f[i] - flow.CourantNr

        elif ctrl.AdvType == "QUICK":
            if ctrl.QUICK_UP:
                if flow.velocity > 0:
                    for i in range(1, nc + 1):
                        coef.b_up[i] = coef.B[i] + flow.CourantNr
                        coef.e_up[i] = coef.e[i] - flow.CourantNr
                        coef.f_up[i] = coef.f[i]
                else:
                    for i in range(1, nc + 1):
                        coef.b_up[i] = coef.B[i]
                        coef.e_up[i] = coef.e[i] + flow.CourantNr
                        coef.f_up[i] = coef.f[i] - flow.CourantNr

            if flow.velocity > 0:
                for i in range(2, nc):
                    coef.A[i] = coef.A[i] - 1 / 8 * flow.CourantNr
                    coef.B[i] = coef.B[i] + 6 / 8 * flow.CourantNr
                    coef.e[i] = coef.e[i] + 3 / 8 * flow.CourantNr
                    coef.B[i] = coef.B[i] + 1 / 8 * flow.CourantNr
                    coef.e[i] = coef.e[i] - 6 / 8 * flow.CourantNr
                    coef.f[i] = coef.f[i] - 3 / 8 * flow.CourantNr

                coef.B[1] = coef.B[1] + 9 / 8 * flow.CourantNr
                coef.e[1] = coef.e[1] - 6 / 8 * flow.CourantNr
                coef.f[1] = coef.f[1] - 3 / 8 * flow.CourantNr

                coef.A[nc] = coef.A[nc] - 1 / 8 * flow.CourantNr
                coef.B[nc] = coef.B[nc] + 6 / 8 * flow.CourantNr
                coef.e[nc] = coef.e[nc] - 5 / 8 * flow.CourantNr
                coef.f[nc] = 0.0
            else:
                for i in range(2, nc):
                    coef.B[i] = coef.B[i] + 3 / 8 * flow.CourantNr
                    coef.e[i] = coef.e[i] + 6 / 8 * flow.CourantNr
                    coef.f[i] = coef.f[i] - 1 / 8 * flow.CourantNr
                    coef.e[i] = coef.e[i] - 3 / 8 * flow.CourantNr
                    coef.f[i] = coef.f[i] - 6 / 8 * flow.CourantNr
                    coef.g[i] = coef.g[i] + 1 / 8 * flow.CourantNr

                coef.B[1] = 0.0
                coef.e[1] = coef.e[1] + 5 / 8 * flow.CourantNr
                coef.f[1] = coef.f[1] - 6 / 8 * flow.CourantNr
                coef.g[1] = coef.g[1] + 1 / 8 * flow.CourantNr

                coef.B[nc] = coef.B[nc] + 3 / 8 * flow.CourantNr
                coef.e[nc] = coef.e[nc] + 6 / 8 * flow.CourantNr
                coef.f[nc] = coef.f[nc] - 9 / 8 * flow.CourantNr

        elif ctrl.AdvType == "central":
            for i in range(2, nc):
                coef.B[i] = coef.B[i] + flow.CourantNr / 2
                coef.f[i] = coef.f[i] - flow.CourantNr / 2

            if flow.velocity > 0:
                coef.B[1] = coef.B[1] + flow.CourantNr
                coef.e[1] = coef.e[1] - flow.CourantNr
                coef.B[nc] = coef.B[nc] + flow.CourantNr
                coef.e[nc] = coef.e[nc] - flow.CourantNr
            else:
                coef.e[1] = coef.e[1] + flow.CourantNr
                coef.f[1] = coef.f[1] - flow.CourantNr
                coef.e[nc] = coef.e[nc] + flow.CourantNr
                coef.f[nc] = coef.f[nc] - flow.CourantNr
        # VBA displays a message and continues when AdvType is unknown.


def transport(prop: PropertyState, coef: EquationCoef, nc: int, ctrl: Controls) -> None:
    """Dispatch exactly as ``Transport``; AdvType is tested even if advection is off."""
    if ctrl.AdvType == "QUICK":
        if ctrl.TimeDiscretisation == "exp":
            _exp_quick_transport(prop, coef, nc, ctrl)
        elif ctrl.TimeDiscretisation == "imp":
            _imp_quick_transport(prop, coef, nc, ctrl)
        else:
            _semi_imp_quick_transport(prop, coef, nc, ctrl)
    else:
        if ctrl.TimeDiscretisation == "exp":
            _exp_transport(prop, coef, nc)
        elif ctrl.TimeDiscretisation == "imp":
            _imp_transport(prop, coef, nc)
        else:
            _semi_imp_transport(prop, coef, nc)


def _exp_transport(prop: PropertyState, coef: EquationCoef, nc: int) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = prop.NewValue[i]
        coef.LeftCoef[i] = coef.B[i]
        coef.CenterCoef[i] = 1 + coef.e[i]
        coef.RigthCoef[i] = coef.f[i]

    prop.NewValue[1] = coef.LeftCoef[1] * prop.Boundary.LeftValue + coef.CenterCoef[1] * prop.Old[1] + coef.RigthCoef[1] * prop.Old[2]
    for i in range(2, nc):
        prop.NewValue[i] = coef.LeftCoef[i] * prop.Old[i - 1] + coef.CenterCoef[i] * prop.Old[i] + coef.RigthCoef[i] * prop.Old[i + 1]
    prop.NewValue[nc] = coef.LeftCoef[nc] * prop.Old[nc - 1] + coef.CenterCoef[nc] * prop.Old[nc] + coef.RigthCoef[nc] * prop.Boundary.RightValue


def _exp_quick_transport(prop: PropertyState, coef: EquationCoef, nc: int, ctrl: Controls) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = prop.NewValue[i]
        coef.L_LeftCoef[i] = coef.A[i]
        coef.LeftCoef[i] = coef.B[i]
        coef.CenterCoef[i] = 1 + coef.e[i]
        coef.RigthCoef[i] = coef.f[i]
        coef.R_RigthCoef[i] = coef.g[i]

    if ctrl.QUICK_UP:
        for i in range(2, nc):
            c = max(abs(prop.Old[i + 1] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            a = max(abs(prop.Old[i] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            b = max(abs(prop.Old[i + 1] - prop.Old[i]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            if abs(a - c) > ctrl.QUICK_UP_Ratio * a or abs(b - c) > ctrl.QUICK_UP_Ratio * b:
                coef.L_LeftCoef[i] = 0.0
                coef.LeftCoef[i] = coef.b_up[i]
                coef.CenterCoef[i] = 1 + coef.e_up[i]
                coef.RigthCoef[i] = coef.f_up[i]
                coef.R_RigthCoef[i] = 0.0

    prop.NewValue[1] = coef.LeftCoef[1] * prop.Boundary.LeftValue + coef.CenterCoef[1] * prop.Old[1] + coef.RigthCoef[1] * prop.Old[2] + coef.R_RigthCoef[1] * prop.Old[3]
    # The VBA uses cell-1 coefficients for the last two terms of cell 2.
    prop.NewValue[2] = coef.L_LeftCoef[2] * prop.Boundary.LeftValue + coef.LeftCoef[2] * prop.Old[1] + coef.CenterCoef[2] * prop.Old[2] + coef.RigthCoef[1] * prop.Old[3] + coef.R_RigthCoef[1] * prop.Old[4]
    for i in range(3, nc - 1):
        prop.NewValue[i] = coef.L_LeftCoef[i] * prop.Old[i - 2] + coef.LeftCoef[i] * prop.Old[i - 1] + coef.CenterCoef[i] * prop.Old[i] + coef.RigthCoef[i] * prop.Old[i + 1] + coef.R_RigthCoef[i] * prop.Old[i + 2]
    # The VBA uses Old(NC-1) twice, omitting Old(NC) here.
    prop.NewValue[nc - 1] = coef.L_LeftCoef[nc - 1] * prop.Old[nc - 3] + coef.LeftCoef[nc - 1] * prop.Old[nc - 2] + coef.CenterCoef[nc - 1] * prop.Old[nc - 1] + coef.RigthCoef[nc - 1] * prop.Old[nc - 1] + coef.R_RigthCoef[nc - 1] * prop.Boundary.RightValue
    prop.NewValue[nc] = coef.L_LeftCoef[nc] * prop.Old[nc - 2] + coef.LeftCoef[nc] * prop.Old[nc - 1] + coef.CenterCoef[nc] * prop.Old[nc] + coef.RigthCoef[nc] * prop.Boundary.RightValue


def _imp_transport(prop: PropertyState, coef: EquationCoef, nc: int) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = prop.NewValue[i]
        coef.TiCoef[i] = prop.Old[i]
        coef.LeftCoef[i] = -coef.B[i]
        coef.CenterCoef[i] = 1 - coef.e[i]
        coef.RigthCoef[i] = -coef.f[i]

    coef.TiCoef[1] = coef.TiCoef[1] - coef.LeftCoef[1] * prop.Boundary.LeftValue
    coef.TiCoef[nc] = coef.TiCoef[nc] - coef.RigthCoef[nc] * prop.Boundary.RightValue

    for i in range(2, nc + 1):
        factor = coef.LeftCoef[i] / coef.CenterCoef[i - 1]
        coef.LeftCoef[i] = coef.LeftCoef[i] - factor * coef.CenterCoef[i - 1]
        coef.CenterCoef[i] = coef.CenterCoef[i] - factor * coef.RigthCoef[i - 1]
        coef.TiCoef[i] = coef.TiCoef[i] - factor * coef.TiCoef[i - 1]

    prop.NewValue[nc] = coef.TiCoef[nc] / coef.CenterCoef[nc]
    for iaux in range(1, nc):
        i = nc - iaux
        prop.NewValue[i] = (coef.TiCoef[i] - coef.RigthCoef[i] * prop.NewValue[i + 1]) / coef.CenterCoef[i]


def _semi_imp_transport(prop: PropertyState, coef: EquationCoef, nc: int) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = prop.NewValue[i]
        coef.TiCoef[i] = prop.Old[i]
        coef.LeftCoef[i] = -coef.B[i] / 2
        coef.CenterCoef[i] = 1 - coef.e[i] / 2
        coef.RigthCoef[i] = -coef.f[i] / 2

    coef.TiCoef[1] = coef.TiCoef[1] - coef.LeftCoef[1] * prop.Boundary.LeftValue + 0.5 * (coef.B[1] * prop.Boundary.LeftValue + coef.e[1] * prop.Old[1] + coef.f[1] * prop.Old[2])
    coef.TiCoef[nc] = coef.TiCoef[nc] - coef.RigthCoef[nc] * prop.Boundary.RightValue + 0.5 * (coef.B[nc] * prop.Old[nc - 1] + coef.e[nc] * prop.Old[nc] + coef.f[nc] * prop.Boundary.RightValue)
    for i in range(2, nc):
        coef.TiCoef[i] = coef.TiCoef[i] + 0.5 * (coef.B[i] * prop.Old[i - 1] + coef.e[i] * prop.Old[i] + coef.f[i] * prop.Old[i + 1])

    for i in range(2, nc + 1):
        factor = coef.LeftCoef[i] / coef.CenterCoef[i - 1]
        coef.LeftCoef[i] = coef.LeftCoef[i] - factor * coef.CenterCoef[i - 1]
        coef.CenterCoef[i] = coef.CenterCoef[i] - factor * coef.RigthCoef[i - 1]
        coef.TiCoef[i] = coef.TiCoef[i] - factor * coef.TiCoef[i - 1]

    prop.NewValue[nc] = coef.TiCoef[nc] / coef.CenterCoef[nc]
    for iaux in range(1, nc):
        i = nc - iaux
        prop.NewValue[i] = (coef.TiCoef[i] - coef.RigthCoef[i] * prop.NewValue[i + 1]) / coef.CenterCoef[i]


def _imp_quick_transport(prop: PropertyState, coef: EquationCoef, nc: int, ctrl: Controls) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = prop.NewValue[i]
        coef.TiCoef[i] = prop.Old[i]
        coef.L_LeftCoef[i] = -coef.A[i]
        coef.LeftCoef[i] = -coef.B[i]
        coef.CenterCoef[i] = 1 - coef.e[i]
        coef.RigthCoef[i] = -coef.f[i]
        coef.R_RigthCoef[i] = -coef.g[i]

    if ctrl.QUICK_UP:
        for i in range(2, nc):
            c = max(abs(prop.Old[i + 1] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            a = max(abs(prop.Old[i] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            b = max(abs(prop.Old[i + 1] - prop.Old[i]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            if abs(a - c) > ctrl.QUICK_UP_Ratio * a or abs(b - c) > ctrl.QUICK_UP_Ratio * b:
                coef.L_LeftCoef[i] = 0.0
                coef.LeftCoef[i] = -coef.b_up[i]
                coef.CenterCoef[i] = 1 - coef.e_up[i]
                coef.RigthCoef[i] = -coef.f_up[i]
                coef.R_RigthCoef[i] = 0.0

    coef.TiCoef[1] = coef.TiCoef[1] - coef.LeftCoef[1] * prop.Boundary.LeftValue
    coef.TiCoef[2] = coef.TiCoef[2] - coef.L_LeftCoef[2] * prop.Boundary.LeftValue
    coef.TiCoef[nc - 1] = coef.TiCoef[nc - 1] - coef.R_RigthCoef[nc - 1] * prop.Boundary.RightValue
    coef.TiCoef[nc] = coef.TiCoef[nc] - coef.RigthCoef[nc] * prop.Boundary.RightValue

    for i in range(3, nc + 1):
        factor = coef.LeftCoef[i - 1] / coef.CenterCoef[i - 2]
        coef.LeftCoef[i - 1] = coef.LeftCoef[i - 1] - factor * coef.CenterCoef[i - 2]
        coef.CenterCoef[i - 1] = coef.CenterCoef[i - 1] - factor * coef.RigthCoef[i - 2]
        coef.RigthCoef[i - 1] = coef.RigthCoef[i - 1] - factor * coef.R_RigthCoef[i - 2]
        coef.TiCoef[i - 1] = coef.TiCoef[i - 1] - factor * coef.TiCoef[i - 2]

        factor = coef.L_LeftCoef[i] / coef.CenterCoef[i - 2]
        coef.L_LeftCoef[i] = coef.L_LeftCoef[i] - factor * coef.CenterCoef[i - 2]
        coef.LeftCoef[i] = coef.LeftCoef[i] - factor * coef.RigthCoef[i - 2]
        coef.CenterCoef[i] = coef.CenterCoef[i] - factor * coef.R_RigthCoef[i - 2]
        coef.TiCoef[i] = coef.TiCoef[i] - factor * coef.TiCoef[i - 2]

    factor = coef.LeftCoef[nc] / coef.CenterCoef[nc - 1]
    coef.LeftCoef[nc] = coef.LeftCoef[nc] - factor * coef.CenterCoef[nc - 1]
    coef.CenterCoef[nc] = coef.CenterCoef[nc] - factor * coef.RigthCoef[nc - 1]
    coef.RigthCoef[nc] = coef.RigthCoef[nc] - factor * coef.R_RigthCoef[nc - 1]
    coef.TiCoef[nc] = coef.TiCoef[nc] - factor * coef.TiCoef[nc - 1]

    prop.NewValue[nc] = coef.TiCoef[nc] / coef.CenterCoef[nc]
    prop.NewValue[nc - 1] = (coef.TiCoef[nc - 1] - coef.RigthCoef[nc - 1] * prop.NewValue[nc]) / coef.CenterCoef[nc - 1]
    for iaux in range(2, nc):
        i = nc - iaux
        prop.NewValue[i] = (coef.TiCoef[i] - coef.RigthCoef[i] * prop.NewValue[i + 1] - coef.R_RigthCoef[i] * prop.NewValue[i + 2]) / coef.CenterCoef[i]


def _semi_imp_quick_transport(prop: PropertyState, coef: EquationCoef, nc: int, ctrl: Controls) -> None:
    for i in range(1, nc + 1):
        prop.Old[i] = max(prop.NewValue[i], prop.minimum)
        coef.TiCoef[i] = prop.Old[i]
        coef.L_LeftCoef[i] = -coef.A[i] / 2
        coef.LeftCoef[i] = -coef.B[i] / 2
        coef.CenterCoef[i] = 1 - coef.e[i] / 2
        coef.RigthCoef[i] = -coef.f[i] / 2
        coef.R_RigthCoef[i] = -coef.g[i] / 2

    if ctrl.QUICK_UP:
        for i in range(2, nc):
            c = max(abs(prop.Old[i + 1] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            a = max(abs(prop.Old[i] - prop.Old[i - 1]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            b = max(abs(prop.Old[i + 1] - prop.Old[i]), ctrl.QUICK_UP_Ratio * (1 + prop.minimum))
            if abs(a - c) > ctrl.QUICK_UP_Ratio * a or abs(b - c) > ctrl.QUICK_UP_Ratio * b:
                coef.L_LeftCoef[i] = 0.0
                coef.LeftCoef[i] = -coef.b_up[i] / 2
                coef.CenterCoef[i] = 1 - coef.e_up[i] / 2
                coef.RigthCoef[i] = -coef.f_up[i] / 2
                coef.R_RigthCoef[i] = 0.0

    coef.TiCoef[1] = coef.TiCoef[1] - 2 * coef.LeftCoef[1] * prop.Boundary.LeftValue - (coef.CenterCoef[1] - 1) * prop.Old[1] - coef.RigthCoef[1] * prop.Old[2] - coef.R_RigthCoef[1] * prop.Old[3]
    coef.TiCoef[2] = coef.TiCoef[2] - 2 * coef.L_LeftCoef[2] * prop.Boundary.LeftValue - coef.LeftCoef[2] * prop.Old[1] - (coef.CenterCoef[2] - 1) * prop.Old[2] - coef.RigthCoef[2] * prop.Old[3] - coef.R_RigthCoef[2] * prop.Old[4]
    coef.TiCoef[nc - 1] = coef.TiCoef[nc - 1] - coef.L_LeftCoef[nc - 1] * prop.Old[nc - 3] - coef.LeftCoef[nc - 1] * prop.Old[nc - 2] - (coef.CenterCoef[nc - 1] - 1) * prop.Old[nc - 1] - coef.RigthCoef[nc - 1] * prop.Old[nc] - 2 * coef.R_RigthCoef[nc - 1] * prop.Boundary.RightValue
    coef.TiCoef[nc] = coef.TiCoef[nc] - coef.L_LeftCoef[nc] * prop.Old[nc - 2] - coef.LeftCoef[nc] * prop.Old[nc - 1] - (coef.CenterCoef[nc] - 1) * prop.Old[nc] - 2 * coef.RigthCoef[nc] * prop.Boundary.RightValue

    for i in range(3, nc - 1):
        coef.TiCoef[i] = coef.TiCoef[i] - coef.L_LeftCoef[i] * prop.Old[i - 2] - coef.LeftCoef[i] * prop.Old[i - 1] - (coef.CenterCoef[i] - 1) * prop.Old[i] - coef.RigthCoef[i] * prop.Old[i + 1] - coef.R_RigthCoef[i] * prop.Old[i + 2]

    for i in range(3, nc + 1):
        factor = coef.LeftCoef[i - 1] / coef.CenterCoef[i - 2]
        coef.LeftCoef[i - 1] = coef.LeftCoef[i - 1] - factor * coef.CenterCoef[i - 2]
        coef.CenterCoef[i - 1] = coef.CenterCoef[i - 1] - factor * coef.RigthCoef[i - 2]
        coef.RigthCoef[i - 1] = coef.RigthCoef[i - 1] - factor * coef.R_RigthCoef[i - 2]
        coef.TiCoef[i - 1] = coef.TiCoef[i - 1] - factor * coef.TiCoef[i - 2]

        factor = coef.L_LeftCoef[i] / coef.CenterCoef[i - 2]
        coef.L_LeftCoef[i] = coef.L_LeftCoef[i] - factor * coef.CenterCoef[i - 2]
        coef.LeftCoef[i] = coef.LeftCoef[i] - factor * coef.RigthCoef[i - 2]
        coef.CenterCoef[i] = coef.CenterCoef[i] - factor * coef.R_RigthCoef[i - 2]
        coef.TiCoef[i] = coef.TiCoef[i] - factor * coef.TiCoef[i - 2]

    factor = coef.LeftCoef[nc] / coef.CenterCoef[nc - 1]
    coef.LeftCoef[nc] = coef.LeftCoef[nc] - factor * coef.CenterCoef[nc - 1]
    coef.CenterCoef[nc] = coef.CenterCoef[nc] - factor * coef.RigthCoef[nc - 1]
    coef.RigthCoef[nc] = coef.RigthCoef[nc] - factor * coef.R_RigthCoef[nc - 1]
    coef.TiCoef[nc] = coef.TiCoef[nc] - factor * coef.TiCoef[nc - 1]

    prop.NewValue[nc] = coef.TiCoef[nc] / coef.CenterCoef[nc]
    prop.NewValue[nc - 1] = (coef.TiCoef[nc - 1] - coef.RigthCoef[nc - 1] * prop.NewValue[nc]) / coef.CenterCoef[nc - 1]
    for iaux in range(2, nc):
        i = nc - iaux
        prop.NewValue[i] = (coef.TiCoef[i] - coef.RigthCoef[i] * prop.NewValue[i + 1] - coef.R_RigthCoef[i] * prop.NewValue[i + 2]) / coef.CenterCoef[i]
