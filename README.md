# River transport model

The [live app](https://ist-river-transport.streamlit.app/) is available on
Streamlit Community Cloud. The illustrated
[student guide](docs/River_Transport_Model_Student_Guide.docx) explains the
inputs, numerical methods, results, and code with screenshots and exercises.

This project reproduces the `Channel` VBA macro in
`Channel_TempOD&BOD&CO2AdvectionSourceCode2025_V3.xlsm` as a Python model
and Streamlit app. The reference PDF is `Transport Model for a River.pdf`.
Those original binary files are kept locally and are not needed to run the app
or its saved parity tests. `Original VBA macro 1.txt` and
`Original VBA macro 2.txt` preserve the source code for review. The numerical
port follows the **executed VBA** when the PDF and code differ.

## Start the app

In PowerShell, from `C:\Work\Transport`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.cmd
```

Open the URL printed by `run.cmd` (normally <http://127.0.0.1:8502/>). If this
river app is already running, the launcher reports its URL and does not start a
second copy. If another app owns port 8502, it selects the next free local port.
`run.cmd` works even if PowerShell script execution is disabled. The app starts
with blank inputs. Use **Load workbook
example** for a short, 20-cell case, or **Load saved inputs JSON** to restore
your own inputs. The app neither uploads nor requires an Excel file. A second
example, `examples/workbook_original_inputs.json`, captures the original
1,000-cell workbook setup.

Inputs cover time controls; river geometry, flow, velocity and dispersion;
atmospheric heat and gas parameters; Henry tables; any number of lateral
discharges; Generic, Temperature, DO, BOD and CO2 properties; boundary values;
and DEFAULT, CELL and INTERVAL initialization. Graph selectors match the
workbook. Results include numerical tables, interactive profiles, time series,
space-time maps, an animated 3D river view, and the workbook chart layouts and
series data. The 3D view colors each model cell by the selected property as
printed time advances. Its six faces form a channel with the entered width and
depth; each one-dimensional cell value is uniform across the full cross-section.
The river length is shortened visually so the channel remains visible, while
all axis coordinates retain the entered dimensions. The color scale stays fixed
throughout the animation. Inputs and results can be downloaded as JSON; each
numerical table can be downloaded as CSV.

If a run fails, the app reports the calculation, property, simulation step and
time, and river cell when available. It shows the failed formula or input value,
suggests specific checks, and retains the original exception chain under
**Technical details**. The report stays visible while inputs are edited and
clears after a successful run or when new inputs are loaded.

## Deploy to Streamlit Community Cloud

Deploy this repository with `app.py` as the entrypoint and Python 3.12. The
root `requirements.txt` lists all runtime dependencies. The hosted app runs
on Linux and needs no Excel installation, uploaded workbook, Windows helper,
or secrets. `run.cmd` and `launch_local.py` are only for local startup. The
bundled JSON examples and parity fixtures are safe to publish and remain
available in a clean deployment. Streamlit's [deployment
guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
describes how to select the GitHub repository, branch, entrypoint, and Python
version.

## Verify parity

Install the development dependencies and run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests compare **every printed cell** of 89 fresh Excel cases (44,500 values),
including six coupled extreme cases, rather than selected locations. For charts
sourced from properties calculated in both programs, they compare titles,
chart types, series names, Excel formulas, and all plotted X/Y arrays. The
supplied 1,000-cell saved workbook result provides another 140,000-value check.
Floating-point checks allow `1e-14` relative or `1e-12` absolute difference
because VBA and Python can differ in their last binary rounding bits. A
standalone test copies only the Python app and bundled JSON example, with no
XLSM, then runs the app and its charts successfully.

To record a fresh case from Excel on a Windows computer, first place the
original `Channel_TempOD&BOD&CO2AdvectionSourceCode2025_V3.xlsm` in the
project root. It is intentionally excluded from Git. Then run:

```powershell
.\.venv\Scripts\python.exe tests\record_excel_oracle.py --list
.\.venv\Scripts\python.exe tests\record_excel_oracle.py --case baseline_20
.\.venv\Scripts\python.exe tests\record_excel_oracle.py --stress --missing
```

The recorder copies the XLSM byte for byte into a temporary directory, runs
`Channel` in a separate Excel process, captures all active output sheets and
charts, closes the copy without saving, and verifies the source workbook hash.
It never edits the user's open workbook. Graph-gap probes are separate because
the macro can display a validation dialog for a blank graph entry.

## Important macro behaviors preserved

- Each time step applies discharges once **before** transport, despite a VBA
  comment describing two half steps. Transport then runs in Generic,
  Temperature, DO, BOD, CO2 order, followed by surface fluxes and reactions.
- The time loop can step beyond the requested end time; results print at the
  first step meeting each print threshold.
- Generic's minimum enforcement depends on the presence of the
  `CyclicBoundary:` label, due to a macro assignment error. Temperature's
  cyclic boundary updates only the inflow value.
- Heat exchange adds solar radiation whenever the main surface-flux switch is
  on. The component switches affect sensible, latent and long-wave terms.
- The gas exchange routine uses the implemented wind/slope/depth expression;
  the VBA `KGas` helper is present but is not called.
- Excel's chart routine selects the CO2 worksheet whenever DO is active, even
  when CO2 is inactive. It may then plot old CO2 worksheet values; conversely,
  it can omit calculated CO2 when DO is inactive. The app deliberately fixes
  this chart-selection bug: workbook-style charts use only the current Python
  simulation results, so they need no Excel workbook or prior worksheet data.
  The chart tests retain exact comparisons for the charts calculated in both
  runs and explicitly test the two divergent CO2 selection cases.
- The original workbook has `PropertyName: ` with a trailing space. VBA's
  exact label search does not read that cell; the Python parser preserves this.

The core modules are `transportmodel/config.py` (input-to-state conversion),
`transportmodel/transport.py` (six transport schemes),
`transportmodel/sources.py` (heat, gas, BOD, Generic and discharge terms),
`transportmodel/numerical.py` (time loop), `transportmodel/charts.py` (graph
series), and `transportmodel/workbook.py` (reference workbook input extraction).
