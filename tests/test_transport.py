"""Focused checks for the VBA transport port and its intentional quirks."""

from __future__ import annotations

import pytest

from transportmodel.state import Controls, EquationCoef, FlowProperties, Grid, PropertyState, zeros
from transportmodel.transport import coef_transport, transport, zero_coeffs


def _prop(values: list[float], left: float = 0.0, right: float = 0.0) -> PropertyState:
    n = len(values)
    prop = PropertyState(Old=zeros(n), NewValue=[0.0, *values])
    prop.Boundary.LeftValue = left
    prop.Boundary.RightValue = right
    return prop


def _solve_dense(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Independent dense Gaussian solver for tiny test cases."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= div
        for row in range(col + 1, n):
            factor = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        out[row] = a[row][n] - sum(a[row][j] * out[j] for j in range(row + 1, n))
    return out


def _quick_dense_expected(prop: PropertyState, coef: EquationCoef, nc: int, method: str) -> list[float]:
    """Build the full finite-difference equations without using VBA elimination."""
    initial = prop.NewValue[1:].copy()
    old = [max(value, prop.minimum) for value in initial] if method == "semi" else initial
    theta = 1.0 if method == "imp" else 0.5
    matrix = [[0.0] * nc for _ in range(nc)]
    rhs = old.copy()
    for i in range(1, nc + 1):
        for offset, coefficient in (
            (-2, coef.A[i]), (-1, coef.B[i]), (0, coef.e[i]),
            (1, coef.f[i]), (2, coef.g[i]),
        ):
            target = i + offset
            if target < 1:
                rhs[i - 1] += coefficient * prop.Boundary.LeftValue
            elif target > nc:
                rhs[i - 1] += coefficient * prop.Boundary.RightValue
            else:
                matrix[i - 1][target - 1] -= theta * coefficient
                if method == "semi":
                    rhs[i - 1] += (1 - theta) * coefficient * old[target - 1]
        matrix[i - 1][i - 1] += 1.0
    return _solve_dense(matrix, rhs)


def test_zero_coeffs_preserves_quick_up_backup() -> None:
    coef = EquationCoef.allocate(4)
    for name in coef.__dataclass_fields__:
        getattr(coef, name)[1] = 7.0
    zero_coeffs(coef, Grid(NC=4))
    for name in ("b_up", "e_up", "f_up"):
        assert getattr(coef, name)[1] == 7.0
    for name in ("A", "B", "e", "f", "g", "L_LeftCoef", "LeftCoef", "CenterCoef", "RigthCoef", "R_RigthCoef", "TiCoef"):
        assert getattr(coef, name)[1] == 0.0


@pytest.mark.parametrize(
    ("velocity", "courant", "expected"),
    [
        (1.0, 0.4, ((0.4, -0.6, 0.2), (0.6, -0.8, 0.2), (0.6, -0.6, 0.0))),
        (-1.0, -0.4, ((0.0, -0.6, 0.6), (0.2, -0.8, 0.6), (0.2, -0.6, 0.4))),
    ],
)
def test_upwind_with_diffusion(velocity: float, courant: float, expected: tuple) -> None:
    coef = EquationCoef.allocate(5)
    ctrl = Controls(Advection=True, Diffusion=True, AdvType="upwind")
    flow = FlowProperties(velocity=velocity, CourantNr=courant, DiffusionNr=0.2)
    coef_transport(flow, 5, ctrl, coef)
    for i, triple in zip((1, 3, 5), expected):
        assert (coef.B[i], coef.e[i], coef.f[i]) == pytest.approx(triple)


@pytest.mark.parametrize("velocity,courant", [(1.0, 0.4), (-1.0, -0.4)])
def test_central_uses_upwind_at_boundaries(velocity: float, courant: float) -> None:
    coef = EquationCoef.allocate(5)
    ctrl = Controls(Advection=True, AdvType="central")
    coef_transport(FlowProperties(velocity=velocity, CourantNr=courant), 5, ctrl, coef)
    assert coef.B[3] == pytest.approx(courant / 2)
    assert coef.f[3] == pytest.approx(-courant / 2)
    if velocity > 0:
        assert (coef.B[1], coef.e[1], coef.f[1]) == pytest.approx((courant, -courant, 0))
        assert (coef.B[5], coef.e[5], coef.f[5]) == pytest.approx((courant, -courant, 0))
    else:
        assert (coef.B[1], coef.e[1], coef.f[1]) == pytest.approx((0, courant, -courant))
        assert (coef.B[5], coef.e[5], coef.f[5]) == pytest.approx((0, courant, -courant))


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("exp", [1.1, 2.0, 2.9]),
        ("imp", [1 + 1 / 11, 2.0, 3 - 1 / 11]),
        ("semi", [1 + 2 / 21, 2.0, 3 - 2 / 21]),
    ],
)
def test_three_cell_diffusion(method: str, expected: list[float]) -> None:
    coef = EquationCoef.allocate(3)
    ctrl = Controls(Diffusion=True, TimeDiscretisation=method)
    coef_transport(FlowProperties(DiffusionNr=0.1), 3, ctrl, coef)
    prop = _prop([1, 2, 3])
    transport(prop, coef, 3, ctrl)
    assert prop.NewValue[1:] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("velocity", "courant", "first", "middle", "last"),
    [
        (1.0, 0.8, (0.0, 0.9, -0.6, -0.3, 0.0), (-0.1, 0.7, -0.3, -0.3, 0.0), (-0.1, 0.6, -0.5, 0.0, 0.0)),
        (-1.0, -0.8, (0.0, 0.0, -0.5, 0.6, -0.1), (0.0, -0.3, -0.3, 0.7, -0.1), (0.0, -0.3, -0.6, 0.9, 0.0)),
    ],
)
def test_quick_coefficients(velocity: float, courant: float, first: tuple, middle: tuple, last: tuple) -> None:
    coef = EquationCoef.allocate(5)
    ctrl = Controls(Advection=True, AdvType="QUICK")
    coef_transport(FlowProperties(velocity=velocity, CourantNr=courant), 5, ctrl, coef)
    for i, expected in ((1, first), (3, middle), (5, last)):
        assert (coef.A[i], coef.B[i], coef.e[i], coef.f[i], coef.g[i]) == pytest.approx(expected)


def test_explicit_quick_preserves_two_vba_index_mistakes() -> None:
    nc = 5
    coef = EquationCoef.allocate(nc)
    coef.f[1], coef.g[1] = 0.1, 0.2
    coef.f[2], coef.g[2] = 1.0, 2.0
    coef.f[4], coef.g[4] = 0.3, 0.4
    prop = _prop([10, 20, 30, 40, 50], left=100, right=200)
    transport(prop, coef, nc, Controls(AdvType="QUICK", TimeDiscretisation="exp"))
    assert prop.NewValue[2] == pytest.approx(20 + 0.1 * 30 + 0.2 * 40)
    assert prop.NewValue[4] == pytest.approx(40 + 0.3 * 40 + 0.4 * 200)


@pytest.mark.parametrize("method", ["imp", "semi"])
@pytest.mark.parametrize("velocity,courant", [(1.0, 0.2), (-1.0, -0.2)])
def test_implicit_quick_matches_independent_dense_equations(method: str, velocity: float, courant: float) -> None:
    nc = 6
    coef = EquationCoef.allocate(nc)
    ctrl = Controls(Advection=True, Diffusion=True, AdvType="QUICK", TimeDiscretisation=method)
    flow = FlowProperties(velocity=velocity, CourantNr=courant, DiffusionNr=0.07)
    coef_transport(flow, nc, ctrl, coef)
    prop = _prop([1.1, 2.7, 1.5, 3.4, 2.2, 1.8], left=4.5, right=0.8)
    expected = _quick_dense_expected(prop, coef, nc, method)
    transport(prop, coef, nc, ctrl)
    assert prop.NewValue[1:] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_quick_up_selects_upwind_at_steep_gradient() -> None:
    nc = 6
    coef = EquationCoef.allocate(nc)
    ctrl = Controls(Advection=True, AdvType="QUICK", TimeDiscretisation="exp", QUICK_UP=True, QUICK_UP_Ratio=2)
    coef_transport(FlowProperties(velocity=1, CourantNr=0.8), nc, ctrl, coef)
    prop = _prop([0, 0, 0, 100, 100, 100])
    transport(prop, coef, nc, ctrl)
    assert coef.LeftCoef[3] == pytest.approx(coef.b_up[3])
    assert coef.CenterCoef[3] == pytest.approx(1 + coef.e_up[3])
    assert coef.RigthCoef[3] == pytest.approx(coef.f_up[3])
    assert coef.L_LeftCoef[3] == 0
    assert coef.R_RigthCoef[3] == 0
    assert coef.LeftCoef[2] == pytest.approx(coef.B[2])


@pytest.mark.parametrize(
    ("method", "expected"),
    [("exp", 20.0), ("imp", 20 / 1.2), ("semi", 20 / 1.1)],
)
def test_quick_up_replacement_in_all_time_methods(method: str, expected: float) -> None:
    nc = 6
    coef = EquationCoef.allocate(nc)
    # The physical QUICK operator is zero. A nonzero backup affects cell 3
    # only when the steep-gradient branch replaces its coefficients.
    coef.e_up[3] = -0.2
    coef.f_up[3] = 0.2
    prop = _prop([0, 0, 0, 100, 100, 100])
    ctrl = Controls(AdvType="QUICK", TimeDiscretisation=method, QUICK_UP=True, QUICK_UP_Ratio=2)
    transport(prop, coef, nc, ctrl)
    assert prop.NewValue[3] == pytest.approx(expected)


def test_semi_implicit_quick_clamps_old_even_when_minimum_inactive() -> None:
    nc = 4
    prop = _prop([-2.0, 1.0, -3.0, 4.0])
    prop.minimum = 0.0
    prop.MinimumActive = False
    transport(prop, EquationCoef.allocate(nc), nc, Controls(AdvType="QUICK", TimeDiscretisation="semi"))
    assert prop.Old[1:] == [0.0, 1.0, 0.0, 4.0]
    assert prop.NewValue[1:] == [0.0, 1.0, 0.0, 4.0]
