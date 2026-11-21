"""Ready-to-run teaching cases for the river model.

Durations and locations are stored in the same units as saved app inputs.
Each load creates fresh inputs, so editing a case never changes its baseline.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import exp, nextafter

from .workbook import PROPERTIES


DAY = 86400
ATMOSPHERE = {
    "air_temperature_c": 20,
    "wind_speed_m_s": 0,
    "minimum_heat_transfer_coefficient": 6.9,
    "humidity_percent": 60,
    "solar_constant_w_m2": 1370,
    "latitude_degrees": 38,
    "sky_temperature_c": -40,
    "sky_temperature_imposed": True,
    "cloud_cover_percent": 0,
    "sunrise_hour": 6,
    "sunset_hour": 18,
    "o2_partial_pressure_bar": 0.2095,
    "co2_pressure_mode": "o2_formula",
    "co2_partial_pressure_bar": 0.2095 * 0.3 / 159,
    "henry_constants": [
        {"temperature_c": t, "o2": o2, "co2": co2}
        for t, o2, co2 in [
            (0, 0.0021812, 0.076425), (5, 0.0019126, 0.063532),
            (10, 0.0016963, 0.053270), (15, 0.0015236, 0.045463),
            (20, 0.0013840, 0.039172), (25, 0.0012630, 0.033363),
        ]
    ],
}


@dataclass(frozen=True)
class TeachingCase:
    key: str
    title: str
    topic: str
    lecture: str
    focus: str
    config: dict | None = None  # None loads the original standalone workbook example.


def _property(value=0.0, *, units="mg/L", left=None, right=None, intervals=(), **options):
    return {
        "active": True,
        "units": units,
        "default": value,
        "minimum": 0,
        "maximum": 1e12,
        "left_boundary": value if left is None else left,
        "right_boundary": value if right is None else right,
        "initial": {
            "default_cell_value": value,
            "type": "INTERVAL" if intervals else "DEFAULT",
            "interval_kind": "CELL",
            "intervals": [{"start": a, "end": b, "value": c} for a, b, c in intervals],
        },
        **options,
    }


def _configuration(*, length, cells, dt, stop, output, velocity=0.0, diffusivity=0.0,
                   scheme="upwind", time="semi", width=5, depth=0.5, slope=0.0,
                   atmosphere=None, properties, discharges=(), stations=None,
                   result_property=None):
    # The loop tests seconds before advancing. A duration just below an exact
    # step multiple avoids a spurious extra step from round-off in day inputs.
    duration = nextafter(stop / DAY, 0.0)
    return {
        "main": {
            "simulation_duration_days": duration, "time_step_seconds": dt,
            "print_interval_seconds": output, "time_method": time,
            "advection": velocity != 0, "advection_type": scheme, "quick_up_ratio": 4,
            "diffusion": diffusivity != 0,
        },
        "river": {
            "length_m": length, "cells": cells, "width_m": width, "depth_m": depth,
            "velocity_mode": "direct", "velocity_m_s": velocity,
            "flow_rate_m3_s": velocity * width * depth, "slope_percent": slope,
            "diffusivity_m2_s": diffusivity,
        },
        "atmosphere": {**deepcopy(ATMOSPHERE), **(atmosphere or {})},
        "discharges": [{"id": i, **row} for i, row in enumerate(discharges, start=1)],
        "properties": {name: properties.get(name, {"active": False}) for name in PROPERTIES},
        "graph": {
            "time_series_cells": list(stations or dict.fromkeys((1, (cells + 1) // 2, cells))),
            "longitudinal_profiles_percent": [0, 10, 20, 50, 80, 100],
        },
        "preferred_property": result_property,
    }


def _tracer(*, initial=0.0, intervals=(), left=0, right=0, rate=0.0,
            units="mg/L", rate_units="Day-1", **options):
    prop = _property(initial, units=units, intervals=intervals, left=left, right=right,
                     generic_rate=rate, generic_rate_units=rate_units)
    return _configuration(properties={"Generic": prop}, **options)


def _thermal(*, initial=15, intervals=(), left=None, cyclic=False, sensible=True,
             latent=True, longwave=True, **options):
    prop = _property(initial, units="°C", intervals=intervals, left=left, cyclic=cyclic,
                     cyclic_label_present=True, free_surface_flux=True,
                     sensible_heat=sensible, latent_heat=latent, radiative_heat=longwave)
    return _configuration(properties={"Temperature": prop}, **options)


def _water_quality(*, initial_bod=50, initial_do=10, initial_co2=1, temperature=15,
                   inflow_bod=None, temperature_forcing=True, rate=0.2, growth=0,
                   logistic_max=0, anaerobic=True, bod_patch=(), indicator=False, **options):
    props = {
        "Temperature": _property(temperature, units="°C", free_surface_flux=temperature_forcing),
        "DO": _property(initial_do, free_surface_flux=True),
        "BOD": _property(initial_bod, left=inflow_bod, intervals=bod_patch,
                         anaerobic_respiration=anaerobic, bod_decay_per_day=rate,
                         bod_growth_per_day=growth, bod_logistic_max=logistic_max,
                         bod_oxygen_half_saturation=0.1),
        "CO2": _property(initial_co2, free_surface_flux=True),
    }
    if indicator:
        props["Generic"] = _property(1, left=1e6, units="CFU/100 mL",
                                     generic_rate=3, generic_rate_units="T90(Hours)")
    return _configuration(properties=props, result_property="DO", **options)


def teaching_cases() -> tuple[TeachingCase, ...]:
    """Return the catalogue in teaching order, grouped by physical topic."""
    cases = []

    def add(key, title, topic, lecture, focus, config=None):
        cases.append(TeachingCase(key, title, topic, lecture, focus, config))

    add("workbook", "Workbook example", "Getting started", "Original workbook",
        "A short run with Generic and Temperature active for learning the controls and output tabs.")
    add("worksheet-intro", "Indicator concentration with growth", "Getting started", "18",
        "Reconstruct the worksheet walkthrough with a 10 km reach, an upstream value of 20 and a 0.05 day doubling time.",
        _tracer(length=10000, cells=1000, dt=500, stop=2*DAY, output=2500,
                velocity=0.02, diffusivity=4, left=20, rate=0.05,
                rate_units="TDuplicate(Days)", units="CFU/100 mL"))
    transport = "Tracer transport and decay"
    closed = dict(length=1000, cells=10, dt=600, stop=12*3600, output=1800)
    add("t90", "Decay in a closed volume", transport, "17, 24",
        "Start at 8,000 CFU/100 mL with T90 = 5 h. Compare the 12-hour output with the exponential solution.",
        _tracer(initial=8000, rate=5, rate_units="T90(Hours)", units="CFU/100 mL", **closed))
    add("half-life", "Half-life in still water", transport, "17",
        "Start at 30,000 with a four-day half-life as in the archived Pool_decay case. Use the five halving intervals.",
        _tracer(length=1, cells=3, width=3, dt=300, stop=20*DAY, output=12*3600,
                initial=30000, rate=4, rate_units="HalfLife(Days)"))
    for time, label in (("exp", "Explicit"), ("imp", "Implicit"), ("semi", "Crank–Nicolson")):
        add(f"decay-step-{time}", f"One decay step · {label}", transport, "17, 23",
            "Apply a 12-hour step to 100 mg/L with a signed rate of −0.2 day⁻¹. Compare with the analytical concentration.",
            _tracer(length=100, cells=4, dt=43200, stop=43200, output=43200,
                    time=time, initial=100, rate=-0.2))

    reach = dict(length=20000, cells=500, dt=400, output=3600, width=5, depth=0.5)
    pulse = ((101, 125, 15000),)
    outlets = tuple({"name": label, "cell": cell, "flow_rate_m3_s": 0.1, "Generic": 100000}
                    for label, cell in (("First discharge", 125), ("Second discharge", 375)))
    lit19 = [
        ("diffuse-pulse", "Instantaneous release · diffusion", 2, 0, 4, 0, pulse, (),
         "Watch a 15,000-unit patch spread and its peak fall as mass redistributes."),
        ("diffuse-decay-pulse", "Instantaneous release · diffusion and decay", 2, 0, 4, 10, pulse, (),
         "Add T90 = 10 h to the same patch and compare the remaining mass after two days."),
        ("decay-pulse", "Instantaneous release · decay only", 2, 0, 0, 10, pulse, (),
         "With transport off, the patch loses amplitude while preserving its footprint."),
        ("diffuse-load", "Two continuous discharges · diffusion", 4, 0, 4, 0, (), outlets,
         "Compare the two peaks with the growth of concentrations between the source cells."),
        ("diffuse-decay-load", "Two continuous discharges · diffusion and decay", 4, 0, 4, 10, (), outlets,
         "T90 = 10 h introduces a loss that can balance the continuing load."),
        ("advect-diffuse-pulse", "Instantaneous release · advection and diffusion", 2, 0.2, 4, 0, pulse, (),
         "Follow the patch downstream. Compare translation and spreading with the diffusion-only case."),
        ("advect-diffuse-load", "Two continuous discharges · advection and diffusion", 2, 0.2, 4, 0, (), outlets,
         "Look for the step in concentration at each discharge and spreading upstream of each step."),
        ("advect-load", "Two continuous discharges · advection only", 2, 0.2, 0, 0, (), outlets,
         "Removing physical diffusion removes the upstream spread in this positive-flow upwind case."),
        ("advect-diffuse-decay-load", "Two continuous discharges · transport and decay", 2, 0.2, 4, 10, (), outlets,
         "Compare the two downstream tails at 0.2 m/s with the conservative result."),
    ]
    for key, title, days, u, d, decay, patch, sources, focus in lit19:
        add(key, title, transport, "19, 27", focus,
            _tracer(**reach, stop=days*DAY, velocity=u, diffusivity=d, rate=decay,
                    rate_units="T90(Hours)" if decay else "Day-1", units="CFU/100 mL",
                    intervals=patch, discharges=sources))
    add("boundary-growth", "Upstream supply and exponential growth", transport, "19",
        "Supply 20 at the inlet, apply a 0.05 day doubling time, and inspect the two-day growth.",
        _tracer(**reach, stop=2*DAY, velocity=0.2, diffusivity=4,
                left=20, rate=0.05, rate_units="TDuplicate(Days)", units="CFU/100 mL"))
    for n in (1, 2):
        add(f"field-discharge-{n}", f"Homework · {n} lateral {'source' if n == 1 else 'sources'}", transport, "19, 27",
            "Use a 20 km reach, 0.2 m/s flow, 1 m²/s diffusivity and T90 = 3 h. Each source supplies 0.1 m³/s at 100,000 CFU/100 mL.",
            _tracer(**reach, stop=2*DAY, velocity=0.2, diffusivity=1, rate=3,
                    rate_units="T90(Hours)", units="CFU/100 mL", discharges=outlets[:n]))
    for flow, entry in [(0.0, "still water"), (0.2, "flowing water")]:
        add(f"homework-pulse-{flow:g}", f"Homework · instantaneous pulse in {entry}", transport, "19, 27",
            "Run the conservative 20 km homework pulse. Set T90 to 3 h to compare with decay.",
            _tracer(**reach, stop=2*DAY, velocity=flow, diffusivity=1, intervals=pulse,
                    units="CFU/100 mL"))
    add("smooth-pulse", "Smooth release for scheme comparison", transport, "27",
        "Start from a resolved Gaussian pulse and compare upwind, central and QUICK at the same grid and time step.",
        _tracer(length=10000, cells=100, dt=250, stop=5000, output=1000, velocity=0.2,
                scheme="quick", intervals=[(i, i, exp(-((i-50)/8)**2))
                                          for i in range(20, 81)]))
    add("travelling-pulse", "Conservative pulse and arrival time", transport, "24",
        "Translate a 450 m pulse at 0.25 m/s with Courant number 1. The pulse width corresponds to a 30-minute inlet event.",
        _tracer(length=24000, cells=160, dt=600, stop=79200, output=600, velocity=0.25,
                time="exp", intervals=((1, 3, 40),), stations=[120, 121]))
    add("decaying-inflow", "Downstream attenuation from a steady inlet", transport, "24",
        "Apply 25 mg/L at the inlet, 0.4 m/s flow and an eight-hour half-life. Inspect the profile near 30 km.",
        _tracer(length=40000, cells=200, dt=500, stop=120000, output=2500,
                velocity=0.4, time="exp", left=25, rate=8/24, rate_units="HalfLife(Days)",
                stations=[50, 150, 151]))
    add("breakthrough", "Breakthrough at 2 and 6 km", transport, "24",
        "Compare when the transported pulse arrives and how its peak and width change. Use the multi-cell time-series chart.",
        _tracer(length=10000, cells=200, dt=45, stop=18000, output=300, velocity=0.5,
                diffusivity=6, intervals=((1, 4, 40),), stations=[41, 121]))
    add("lateral-mixing", "Steady lateral mixing", transport, "23, 24",
        "A 10 m³/s river at 10 mg/L receives 0.1 m³/s at 100 mg/L. Compare with the lecture’s instantaneous mixing balance.",
        _tracer(length=600, cells=12, width=10, depth=1, dt=10, stop=1800, output=60,
                velocity=1, initial=10, left=10, right=10, stations=[5, 6, 7],
                discharges=[{"name": "Side inflow", "cell": 6, "flow_rate_m3_s": 0.1, "Generic": 100}]))


    heat = "Temperature and surface exchange"
    still = dict(length=4000, cells=100, dt=600, stop=10*DAY, output=7200)
    daily_solar = {"solar_constant_w_m2": 300}
    add("daily-cycle", "Daily warming and cooling", heat, "21, 22",
        "Reconstruct the 20 °C air and 60% humidity example with an effective solar input of 300 W/m². Compare day and night in Cell time series.",
        _thermal(**still, longwave=False, atmosphere=daily_solar))
    add("longwave", "Long-wave radiation on a clear night", heat, "21, 22, 24",
        "Turn on long-wave exchange with a sky at −40 °C. Compare with Daily warming and cooling.",
        _thermal(**still, atmosphere=daily_solar))
    for key, title, air, opts, focus in [
        ("dry-air", "Dry air and evaporative cooling", {"humidity_percent": 30}, {"longwave": False},
         "Lower humidity to 30% with the other forcing unchanged. Compare the mean temperature."),
        ("wind", "Wind and surface convection", {"wind_speed_m_s": 5}, {"longwave": False},
         "Increase wind speed to 5 m/s. The same transfer coefficient affects sensible and latent exchange."),
        ("shallow", "Shallow water under daily forcing", {}, {"longwave": False, "depth": 0.25},
         "At 0.25 m depth, the same surface flux acts on a smaller thermal mass."),
        ("deep", "Deep water under daily forcing", {}, {"longwave": False, "depth": 2},
         "At 2 m depth, compare the diurnal amplitude with the shallow case."),
        ("summer", "Summer daylight and solar heating", {"sunrise_hour": 5, "sunset_hour": 21}, {"longwave": False},
         "Compare 16 hours of daylight with the matched winter case."),
        ("winter", "Winter daylight and solar heating", {"sunrise_hour": 8, "sunset_hour": 16}, {"longwave": False},
         "Compare eight hours of daylight with the matched summer case."),
        ("latitude", "High latitude under equal daylight", {"latitude_degrees": 60}, {"longwave": False},
         "Keep day length fixed and increase latitude. Compare the model’s cosine factor for solar heating."),
        ("cloud", "Cloud cover and the estimated sky flux", {"sky_temperature_imposed": False, "cloud_cover_percent": 80}, {},
         "Clouds reduce short-wave radiation and change the sky estimate for long-wave exchange. Try 0% cloud cover."),
        ("sensible", "Sensible heat exchange only", {"solar_constant_w_m2": 0}, {"latent": False, "longwave": False},
         "With no radiation or evaporation, 15 °C water approaches the 20 °C air temperature."),
        ("latent", "Latent heat exchange only", {"solar_constant_w_m2": 0}, {"sensible": False, "longwave": False},
         "Evaporation cools water in dry air. Increase humidity to investigate the vapour-pressure difference."),
        ("shortwave", "Short-wave heating only", {}, {"sensible": False, "latent": False, "longwave": False},
         "Solar heat warms the water during daylight and leaves it unchanged at night in this isolated test."),
    ]:
        forcing = daily_solar | air if key != "shortwave" else air
        add(key, title, heat, "21, 22, 24", focus, _thermal(**still, atmosphere=forcing, **opts))
    add("cool-inlet", "Cool inflow and warm water along a reach", heat, "22",
        "Use a 15 °C inflow and a 20 °C background, with colder and warmer initial intervals. Follow advection and surface exchange.",
        _thermal(length=12000, cells=300, dt=600, stop=4*DAY, output=7200,
                 initial=20, left=15, intervals=((20, 40, 15), (50, 60, 10)),
                 velocity=0.1, diffusivity=4, depth=2))
    add("thermal-cycle", "Cyclic temperature boundary", heat, "22",
        "Follow a warm patch around a recirculating channel. Compare with a fixed left boundary.",
        _configuration(length=4000, cells=80, dt=200, stop=40000, output=1000, velocity=0.2,
                       time="exp", properties={
                           "Temperature": _property(18, units="°C", intervals=((20, 30, 25),),
                                                    cyclic=True, cyclic_label_present=True)
                       }))
    add("warm-discharge", "Warm lateral inflow", heat, "22, 24",
        "Mix a 1.2 m³/s discharge at 28 °C into an 8 m³/s reach at 16 °C. Compare the downstream result with the mixing balance.",
        _configuration(length=1000, cells=20, width=20, depth=2, dt=50, stop=7200, output=200,
                       velocity=0.2, properties={"Temperature": _property(16, units="°C")},
                       discharges=[{"name": "Warm inflow", "cell": 7, "flow_rate_m3_s": 1.2, "Temperature": 28}],
                       stations=[6, 7, 8]))

    numerical = "Numerical methods"
    sharp = dict(length=10100, cells=101, output=1000, velocity=0.2, intervals=((51, 51, 1),))
    for key, title, method, scheme, d, dt, stop, focus in [
        ("exact-upwind", "Upwind shift at Courant 1", "exp", "upwind", 0, 500, 5000,
         "The explicit stencil moves the single-cell pulse by one cell each step."),
        ("smeared-upwind", "Upwind spreading at Courant 0.5", "exp", "upwind", 0, 250, 5000,
         "The lower peak and wider pulse show numerical diffusion with physical diffusion off."),
        ("unstable-upwind", "Upwind beyond its explicit stability limit", "exp", "upwind", 0, 600, 6000,
         "At Courant 1.2 the explicit upwind stencil creates growing oscillations. Inspect the sign changes."),
        ("diffusion-limit", "Explicit diffusion at its limit", "exp", "upwind", 10, 500, 5000,
         "Diffusion number 0.5 is the marginal explicit limit. Raise the step to 600 s and compare."),
        ("central-limit", "Central differences with balancing diffusion", "exp", "central", 10, 500, 5000,
         "At Courant 1 and diffusion number 0.5, the stencil reduces to a one-cell shift."),
        ("central-unstable", "Unstable explicit central pulse", "exp", "central", 2, 1000, 6000,
         "The peak and negative undershoot grow even though the diffusion number is only 0.2."),
        ("central-diffusive", "Stable explicit central pulse", "exp", "central", 20, 200, 5000,
         "Physical diffusion offsets the central stencil’s negative weight at these settings."),
        ("central-implicit", "Bounded oscillations with implicit central", "imp", "central", 2, 1000, 6000,
         "Compare the small negative tail with the unstable explicit run at the same grid and step."),
    ]:
        add(key, title, numerical, "25–28, 32", focus,
            _tracer(**(sharp | ({"velocity": 0} if key=="diffusion-limit" else {})
                         | ({"output": 600} if key=="unstable-upwind" else {})),
                    diffusivity=d, dt=dt, stop=stop, time=method, scheme=scheme))
    for scheme, lecture in [("upwind","26"), ("central","27")]:
        add(f"four-cell-{scheme}", f"Four-cell hand calculation · {scheme}", numerical, lecture,
            "Start with [1, 1, 0, 0], Δx = 1 m and Δt = 0.5 s. Compare the first row after time zero with the exercise.",
            _tracer(length=4, cells=4, dt=0.5, stop=0.5, output=0.5,
                    velocity=1, left=1, intervals=((1, 2, 1),), time="exp", scheme=scheme))
    add("quick-face", "QUICK near a sharp trailing edge", numerical, "28, 32",
        "A sharp step gives QUICK a negative face reconstruction. Compare the resulting profile with upwind and QUICK_UP.",
        _tracer(length=12, cells=12, dt=0.5, stop=3, output=0.5, velocity=1,
                intervals=((2, 4, 2),), scheme="quick", time="exp"))
    for scheme in ("upwind","central","quick"):
        for time, label in (("exp", "explicit"), ("semi", "semi-implicit"), ("imp", "implicit")):
            add(f"six-cell-{scheme}-{time}", f"Six-cell exercise · {scheme.upper() if scheme=='quick' else scheme}, {label}",
                numerical, "32", "Six cells at Δx = 1 m with a two-cell patch and Δt = 0.8 s. Compare the three steps using the app’s VBA boundary stencils.",
                _tracer(length=6, cells=6, dt=0.8, stop=2.4, output=0.8, velocity=1,
                        intervals=((3, 4, 1),), scheme=scheme, time=time))
    add("six-cell-large-step", "Six-cell exercise · Courant 1.4", numerical, "32",
        "Repeat the explicit six-cell upwind case with a larger step. Look for an overshoot and new negative values.",
        _tracer(length=6, cells=6, dt=1.4, stop=4.2, output=1.4, velocity=1,
                intervals=((3, 4, 1),), time="exp"))
    add("reverse-flow", "Tracer moving towards the left boundary", numerical, "15, 16, 26",
        "With negative velocity, the upstream side is on the right. At |Cr| = 1 the pulse moves one cell left per step.",
        _tracer(**(sharp | {"velocity": -0.2}), dt=500, stop=5000, time="exp"))
    add("quick-up", "QUICK with a local upwind switch", numerical, "28",
        "Compare QUICK_UP with QUICK and upwind for the same sharp front. The workbook’s gradient-ratio rule controls the switch.",
        _tracer(length=12000, cells=300, dt=200, stop=24000, output=800, velocity=0.2,
                intervals=((49, 69, 1), (100, 120, 1)), scheme="quick_up"))

    oxygen = "Oxygen, BOD and carbon dioxide"
    tank = dict(length=10000, cells=10, width=0.6/0.55, depth=0.55, dt=800,
                stop=20*DAY, output=10800, stations=[1,4,10])
    # The slide's slope labels disagree. Keep the decimals as workbook inputs
    # so the two high-wind cases give the similar response shown in the slides.
    for key,label,wind,slope,velocity,growth,maximum,discharges,focus in [
        ("bod-low-wind", "BOD decay under low wind", 0.5,0.00001,0.08,0,0,(),
         "Wind is 0.5 m/s. Follow oxygen depletion, carbon dioxide production and late recovery."),
        ("bod-high-wind", "BOD decay under high wind", 5,0.00001,0.08,0,0,(),
         "Wind is 5 m/s. Compare the oxygen minimum and recovery with the low-wind run."),
        ("bod-steep", "BOD decay with higher slope and velocity", 5,0.00005,0.18,0,0,(),
         "The slope and recorded velocity increase. Advection remains off so the comparison isolates the slope in the gas-transfer calculation."),
        ("bod-growth", "BOD decay and logistic growth", 5,0.00001,0.08,0.3,9,(),
         "The growth input is 0.3 day⁻¹. A 9 mg/L logistic maximum gives a nonzero long-term BOD level."),
        ("bod-load", "BOD with a continuing discharge", 2,0.00001,0.08,0,0,
         tuple({"name": "Local sustained load","cell": i,"flow_rate_m3_s": 20/DAY,
                "BOD": 45, "DO": 0, "CO2": 1, "Temperature": 15} for i in (1,4)),
         "A 20 m³/day load at 45 mg/L in each plotted cell sustains oxygen demand. Compare cells 1 and 4 with an unloaded cell."),
    ]:
        config=_water_quality(**tank, slope=slope, velocity=velocity, growth=growth, logistic_max=maximum,
                              discharges=discharges, atmosphere={"wind_speed_m_s":wind})
        config["main"]["advection"]=False
        add(key,label,oxygen,"30, 32",focus,config)
    add("aerobic-only", "BOD with aerobic respiration only", oxygen, "29, 30",
        "Compare both accumulated respiration outputs with the low-wind run. Without anaerobic loss, low DO limits continued BOD removal.",
        _water_quality(**tank, slope=0.00001, anaerobic=False, atmosphere={"wind_speed_m_s":0.5}))
    gas_props={"Temperature":_property(17,units="°C"),
               "DO":_property(5, free_surface_flux=True),
               "CO2":_property(5, free_surface_flux=True)}
    add("saturation", "DO uptake and CO₂ outgassing", oxygen, "29",
        "Start DO below air equilibrium and CO₂ above it. Follow both gases as they approach their Henry-law saturation concentrations.",
        _configuration(length=10000,cells=10,dt=300,stop=5*DAY,output=3600,slope=0.01,
                       atmosphere={"wind_speed_m_s":2},properties=gas_props,result_property="DO"))
    add("reactive-plume", "Reactive BOD pulse with longitudinal transport", oxygen, "30, 32",
        "Follow the BOD pulse and oxygen depression downstream. Change the spatial scheme or disable diffusion and compare.",
        _water_quality(length=24000,cells=120,dt=300,stop=3*DAY,output=3600,
                       velocity=0.15,diffusivity=4,slope=0.01,initial_bod=0.2,
                       bod_patch=((28,34,40),),atmosphere={"wind_speed_m_s":2}))
    for pop in (5000,10000):
        add(f"pond-{pop}",f"Three ponds · {pop:,} inhabitants",oxygen,"31",
            "Represent three 200 m tanks in series with 150 m width and 0.5 m depth. Compare DO and BOD for the two hydraulic loads.",
            _water_quality(length=600,cells=3,dt=600,stop=60*DAY,output=DAY,
                           width=150,depth=0.5,velocity=(pop*0.3/DAY)/(150*0.5),time="imp",
                           initial_bod=2,inflow_bod=333,indicator=True,temperature_forcing=False,
                           slope=0.001,atmosphere={"wind_speed_m_s":1}))
    add("pond-reaeration", "Three ponds with stronger surface exchange", oxygen, "31, 32",
        "Increase wind in the 5,000-person pond case to illustrate stronger atmospheric exchange. Compare DO at the same BOD load.",
        _water_quality(length=600,cells=3,dt=600,stop=60*DAY,output=DAY,
                       width=150,depth=0.5,velocity=(1500/DAY)/(150*0.5),time="imp",
                       initial_bod=2,inflow_bod=333,indicator=True,temperature_forcing=False,
                       slope=0.001,atmosphere={"wind_speed_m_s":5}))
    for width,depth,label in [(50,2,"Large river"),(3,0.5,"Small stream")]:
        add(f"receiving-{width}",f"{label} · same wastewater discharge",oxygen,"31, 32",
            "Discharge 1,500 m³/day at 333 mg/L BOD. Compare the oxygen response in the two receiving waters at 0.2 m/s.",
            _water_quality(length=24000,cells=120,dt=300,stop=5*DAY,output=7200,width=width,
                           depth=depth,velocity=0.2,diffusivity=2,slope=0.001,initial_bod=1,initial_do=9,
                           atmosphere={"wind_speed_m_s":0.5},stations=[9,10,30,60],
                           discharges=[{"name":"Wastewater discharge","cell":10,"flow_rate_m3_s":1500/DAY,
                                        "BOD":333,"DO":0,"CO2":1,"Temperature":15}]))
    return tuple(cases)
