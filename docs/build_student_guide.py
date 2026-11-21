"""Build the illustrated student handbook from the committed app screenshots."""

from __future__ import annotations

from pathlib import Path
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SHOTS = ROOT / "screenshots"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "River_Transport_Model_Student_Guide.docx"
sys.path.insert(0, str(ROOT.parent))
from transportmodel.presets import teaching_cases


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    item = OxmlElement("w:shd")
    item.set(qn("w:fill"), fill)
    tc_pr.append(item)


def add_page_number(paragraph) -> None:
    paragraph.add_run("Page ")
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, value, end):
        run._r.append(element)


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.73)
section.bottom_margin = Inches(0.68)
section.left_margin = Inches(0.75)
section.right_margin = Inches(0.75)

styles = doc.styles
styles["Normal"].font.name = "Aptos"
styles["Normal"].font.size = Pt(10)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Title"].font.name = "Aptos Display"
styles["Title"].font.size = Pt(26)
styles["Title"].font.bold = True
styles["Title"].font.color.rgb = RGBColor(0, 0, 0)
title_properties = styles["Title"]._element.get_or_add_pPr()
title_border = title_properties.find(qn("w:pBdr"))
if title_border is not None:
    title_properties.remove(title_border)
for name, size in (("Heading 1", 16), ("Heading 2", 12), ("Heading 3", 10.5)):
    styles[name].font.name = "Aptos Display"
    styles[name].font.size = Pt(size)
    styles[name].font.bold = True
    styles[name].font.color.rgb = RGBColor(35, 48, 63)
    styles[name].paragraph_format.space_before = Pt(12)
    styles[name].paragraph_format.space_after = Pt(5)

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.style = "Caption"
add_page_number(footer)


def p(text: str = "", *, bold_lead: str | None = None) -> None:
    paragraph = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        paragraph.add_run(bold_lead).bold = True
        paragraph.add_run(text[len(bold_lead):])
    else:
        paragraph.add_run(text)


def h(text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def bullets(items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def steps(items: list[str]) -> None:
    for number, item in enumerate(items, start=1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.paragraph_format.first_line_indent = Inches(-0.22)
        paragraph.add_run(f"{number}.  ").bold = True
        paragraph.add_run(item)


def table(headers: list[str], rows: list[list[str]], widths: tuple[float, ...] | None = None) -> None:
    result = doc.add_table(rows=1, cols=len(headers))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.style = "Light Shading Accent 1"
    if widths:
        result.autofit = False
        for column, width in zip(result.columns, widths):
            column.width = Inches(width)
    result.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for cell, value in zip(result.rows[0].cells, headers):
        cell.text = value
        shade(cell, "DDE5EC")
        cell.paragraphs[0].paragraph_format.keep_with_next = True
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        for cell, value in zip(result.add_row().cells, row):
            cell.text = value
    for row in result.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if widths:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.line_spacing = 1.02
                    for run in paragraph.runs:
                        run.font.size = Pt(9.5)
    doc.add_paragraph()


def figure(name: str, caption: str, focus: str) -> None:
    path = SHOTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    picture_paragraph = doc.add_paragraph()
    picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image = picture_paragraph.add_run().add_picture(str(path), width=Inches(6.65))
    image._inline.docPr.set("descr", caption + " " + focus)
    picture_paragraph.paragraph_format.keep_with_next = True
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.add_run(caption).bold = True
    paragraph.add_run("  " + focus)


def code(text: str, explanation: str) -> None:
    container = doc.add_table(rows=1, cols=1)
    row_properties = container.rows[0]._tr.get_or_add_trPr()
    row_properties.append(OxmlElement("w:cantSplit"))
    cell = container.cell(0, 0)
    shade(cell, "F0F3F6")
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text.strip("\n"))
    run.font.name = "Consolas"
    run.font.size = Pt(8)
    doc.add_paragraph(explanation)


doc.add_paragraph("River Transport Model Student Guide", style="Title")
p("Code, numerical methods, and a step by step guide to the Streamlit app")
p("For students of energy and mass transfer  |  21 November 2026")
p("This handbook explains the one dimensional river model from input to result. It follows the executed Python port of the Channel VBA macro. Work through the app tutorial first, then use the code chapters to connect each control to an equation and a program step. The app runs without Excel.")
p("Live app: https://ist-river-transport.streamlit.app/")
p("Source code: https://github.com/ricardofmt-lang/Mass_transfer_river_model")
h("What you should be able to do", 1)
bullets([
    "Set up a river reach with a meaningful grid, time step, transport scheme, and property distribution.",
    "Explain advection, longitudinal diffusion, lateral mixing, surface exchange, and reactions in the calculation order used by the code.",
    "Read the numerical table, spatial profiles, time series, space time map, and animated 3D channel without confusing visualization with a three dimensional calculation.",
    "Use the Courant and diffusion numbers to reason about resolution and numerical behavior.",
    "Trace a result from a Streamlit input through configuration, mutable state, solver, and chart modules.",
])
h("Model boundary and scope", 1)
p("Every river cell represents a whole vertical cross section. Width and depth affect flow, mixing volume, and surface exchange, but each property has only one value per cell. The 3D view gives those 1D values a channel shape; it does not calculate vertical or cross river gradients.")
p("The calculation follows the executed Channel macro. The revised theoretical notes explain its implemented equations. The closing catalogue identifies the teaching cases by lecture and shows which result to examine.")

doc.add_page_break()
h("Part 1 Using the app", 1)
h("A first complete run", 2)
steps([
    "Open the app. For a local run, use run.cmd and open the address it prints. The editor starts with blank numerical inputs.",
    "In Teaching cases, choose Getting started and Workbook example, then select Load case. This fills a short 20-cell demonstration.",
    "Read Controls, River, Atmosphere, Discharges, Properties, and Graphs from left to right. Change one value at a time during your first experiments.",
    "Select Run simulation. The Results section appears below the input tabs.",
    "Choose a property, compare its numerical table with a profile and time series, and then open the 3D river animation.",
    "Save the configuration with Download saved inputs JSON. To reopen it, expand Open saved inputs or clear the editor, upload the file and select Apply JSON inputs.",
])
figure("01_controls_blank.jpg", "Figure 1  Empty starting state.", "Select a topic and case at the top of the app. The duration, time step and output interval stay empty until you load a case or enter values.")
figure("02_controls_example.jpg", "Figure 2  Example controls.", "The example uses 0.005 day, a 60 second step, a 120 second output interval, semi implicit transport, diffusion on, and advection off.")

h("Controls and numerical resolution", 2)
table(["Input", "Meaning", "How to interpret it"], [
    ["Duration", "Requested end time in days.", "The loop can finish one step beyond this time; inspect the printed time column."],
    ["Time step dt", "Seconds advanced by each calculation step.", "Smaller steps usually improve time resolution and increase run time."],
    ["Output interval", "Target spacing between saved rows.", "The first step reaching a print threshold is saved; values are not interpolated."],
    ["Time method", "exp, imp, or semi (Crank Nicolson).", "This changes longitudinal transport weighting; source terms retain their macro specific updates."],
    ["Advection", "Turns transport by mean flow on or off.", "Off makes displayed Courant number zero even if velocity is entered."],
    ["Diffusion", "Turns longitudinal spreading on or off.", "Off makes displayed diffusion number zero even if diffusivity is entered."],
])
p("For equal cells, dx = length / number of cells. Cr = velocity × dt / dx and D = diffusivity × dt / dx² when the respective process is enabled. The app rounds diagnostic numbers to two decimals: a displayed 0.0 can mean a small nonzero value. In the example, dx = 500 m and D = 4 × 60 / 500² = 0.00096, which displays as 0.0.")

h("River geometry and flow", 2)
figure("03_river.jpg", "Figure 3  River input tab.", "The River tab shows cell length and the resulting velocity and flow rate. Choose an entered velocity, conversion from flow rate, or a Manning estimate.")
table(["Input", "Role in calculation", "Student check"], [
    ["Length and cells", "Fix the one dimensional grid and cell centre positions.", "Cell 1 is centred at dx/2, not at x = 0."],
    ["Width and depth", "Set cell volume and exchange area; depth also enters gas exchange.", "A deeper channel dilutes the same lateral load more."],
    ["Flow velocity", "Sets the mean longitudinal velocity used for advection.", "The sign determines the upstream end."],
    ["Diffusivity", "Sets longitudinal mixing in m²/s.", "The standard option uses 0.1 × abs(velocity) × width."],
    ["Slope", "Sets the slope in gas exchange and in the selected Manning estimate.", "Enter percent. Multiply an m/m slope by 100."],
    ["Manning coefficient", "Specifies channel roughness in Manning mode.", "Compare velocities at different roughness values with the same geometry and slope."],
])

p("In Manning mode, enter the roughness coefficient and slope in percent. For example, a slope of 0.0001 m/m is 0.01%. The estimate uses hydraulic radius R = width × depth / (width + 2 × depth) for a rectangular channel.")
h("Atmospheric forcing and solubility", 2)
figure("04_atmosphere.jpg", "Figure 4  Atmosphere tab.", "Air temperature, wind, humidity, sun, and sky settings drive heat exchange. Gas pressures and the Henry table appear farther down the same tab.")
p("Solar forcing follows a daylight sine between sunrise and sunset, reduced by latitude, atmospheric absorption, and cloud cover. Sensible heat responds to the air water temperature difference. Latent heat responds to vapour pressure and humidity. Long wave radiation uses an imposed sky temperature or the macro correlation. These processes run only when atmospheric heat exchange is enabled for Temperature.")
p("DO and CO2 surface exchange use Henry constants interpolated at the current water temperature. Enter one temperature per table row, with O2 and CO2 constants in the associated columns. The example CO2 pressure is calculated from the O2 pressure. Gas exchange requires the relevant property and surface exchange switch.")

h("Continuous lateral sources", 2)
figure("05_discharges.jpg", "Figure 5  Discharge editor.", "Choose cell number or distance for each source, then provide its flow rate and each property value in the same table.")
p("A continuous discharge is mixed into its target cell before that step's longitudinal transport. The update is a volume weighted average of old river water and incoming source water. A positive source flow adds water each step in the macro's mixing calculation. Instantaneous releases belong in the initial distribution, not in this table.")

h("Properties, boundaries, and initial values", 2)
figure("06_properties_generic.jpg", "Figure 6  Generic property.", "The property tabs select Generic, Temperature, DO, BOD, or CO2. The visible fields depend on the active property and initialization choice.")
figure("07_temperature.jpg", "Figure 7  Temperature property.", "Atmospheric heat switches sit with Temperature. Solar input follows the overall atmosphere exchange switch even when the three optional heat components are off.")
table(["Property", "What it represents", "Coupling or important controls"], [
    ["Generic", "A tracer or pollutant in user chosen units.", "Optional first order decay or growth; T90 and half life inputs are converted to rates."],
    ["Temperature", "Water temperature in degrees Celsius.", "Surface heat exchange and optional cyclic boundary."],
    ["DO", "Dissolved oxygen.", "Gas exchange and aerobic BOD consumption; needs Temperature."],
    ["BOD", "Biological oxygen demand.", "Growth, aerobic and optional anaerobic decay; needs DO."],
    ["CO2", "Dissolved carbon dioxide.", "Gas exchange and carbon produced by BOD respiration."],
])
p("Left and right boundary values describe water entering from either end. Positive velocity enters from the left; negative velocity enters from the right. The cyclic option is implemented for Temperature in the macro and should be interpreted as that specific code behavior, not a generic periodic boundary for every property.")
p("DEFAULT applies a background value everywhere. CELL first fills the background, then overrides named cells. INTERVAL first fills the background, then overrides ranges. Cells not listed remain at the background value. Minimum and maximum fields follow the macro's property rules; do not assume every field is a universal post processing clamp.")

p("For a conservative Generic tracer, enter zero with Day-1 or enter 1 with RT. A negative Day-1 rate gives decay. T90, half-life and doubling time expect positive durations. In BOD, a zero logistic maximum skips the logistic step for nonnegative values while leaving oxidation active.")
h("Graph selections", 2)
figure("08_graph_selectors.jpg", "Figure 8  Workbook graph selectors.", "Time series selectors identify cell numbers. Profile selectors identify percentages of the printed run. Use standard graph selection to populate a teaching example.")
p("These selectors update Workbook graphs for the last completed run. Use Output snapshot in the interactive profile to inspect an individual saved time. When you change a simulation input, the app tells you to rerun before interpreting the results.")

doc.add_page_break()
h("Part 2 Reading and exporting results", 1)
h("Numerical tables", 2)
figure("09_results_table.jpg", "Figure 9  Numerical result table.", "Each row is a printed time; each cell column is located at the stated centre coordinate. Horizontal scrolling reveals the remaining cells.")
p("The example asks for 0.005 day = 432 seconds but the final printed row is 480 seconds. This follows the macro's step and print threshold logic. Never label the last row as exactly the requested duration without checking Time (s).")
p("Choose the property above the result tabs. Download table CSV gives the selected table; Download all results JSON retains every active property, printed time, coordinate, and diagnostic. Saved inputs JSON is the file to use when you want to reproduce or share the setup.")
h("Profiles and time series", 2)
figure("11_profile_temperature.jpg", "Figure 10  Temperature profile.", "A longitudinal profile fixes a printed time and shows the value at each cell centre. The cold interval in this example remains visible after diffusion.")
p("A cell time series instead fixes a position and shows the saved values through time. A space time map puts position on one axis, printed time on the other, and uses color for the selected property. Compare all three views before drawing conclusions about movement or spreading.")
h("Animated 3D channel", 2)
figure("12_3d_fullscreen.jpg", "Figure 11  3D river and GIF export.", "The channel height is the entered 0.5 m depth and its width is 5 m. Play or scrub through the saved times. Download 3D GIF records the selected property from the first to the last saved time.")
p("Rotate or zoom the channel before export to choose the camera angle. The GIF uses the same frames, playback speed and fixed color scale as Play. A whole cross-section shares one color. The length is compressed visually, and the axes and hover readouts retain the model’s dimensions.")
h("Workbook style graphs and diagnostics", 2)
figure("13_workbook_graph.jpg", "Figure 12  Workbook style chart.", "These charts use the current Python run rather than old Excel worksheet data. Select a chart above the plot and inspect its series data if needed.")
p("Courant number compares travel in one time step with cell length. Diffusion number compares longitudinal spreading with cell length squared. Grid Reynolds number is Cr/D when D is positive. Residence time estimates travel time from length or cell size and mean velocity; the macro adds a small denominator offset, so treat extreme zero velocity cases cautiously.")
p("The app flags explicit upwind and central settings that exceed the applicable stability or monotonicity limits. Finite results remain visible so you can inspect the oscillations. Try Unstable explicit central pulse, then compare it with Bounded oscillations with implicit central.")
h("When a simulation fails", 2)
steps([
    "Read the named calculation stage, property, time step, and cell in the error panel.",
    "Check the particular input or formula mentioned there: cell index, depth, denominator, Henry table, or a rate that grows too fast.",
    "Open Technical details only when you need the original Python exception chain.",
    "Correct an input and rerun. Keep the failed setup as saved inputs JSON if you are studying why it failed.",
])

doc.add_page_break()
h("Part 3 How the code works", 1)
h("Module map and data flow", 2)
table(["Module", "Responsibility", "Read it after"], [
    ["app.py", "Streamlit editors, results views, downloads, error panel.", "The app tutorial"],
    ["transportmodel/config.py", "Turn editor JSON into the macro shaped mutable state.", "state.py"],
    ["transportmodel/state.py", "One based arrays and records for grid, flow, properties, atmosphere, coefficients.", "The input guide"],
    ["transportmodel/numerical.py", "Execute the time loop and snapshot printed rows.", "config.py"],
    ["transportmodel/transport.py", "Build stencils and solve explicit, implicit, or semi implicit transport.", "numerical.py"],
    ["transportmodel/sources.py", "Discharges, heat, gas, Generic, and BOD DO CO2 reactions.", "numerical.py"],
    ["transportmodel/charts.py", "Create workbook style chart specifications from current results.", "The results guide"],
    ["transportmodel/visualization3d.py", "Build the animated channel from one dimensional printed values.", "The results guide"],
    ["transportmodel/workbook.py", "Optional Excel input extraction for local reference work.", "config.py"],
])
p("The central flow is editor fields -> JSON shaped configuration -> build_state -> mutable model arrays -> run_model -> SimulationResult -> tables and charts. The hosted app needs neither the original workbook nor Excel.")

p("presets.py supplies the teaching inputs through the same configuration path as saved JSON. animation_download.py loads the browser exporter, which reads the Plotly frames and encodes the selected property as a GIF.")
h("One based state and cell geometry", 2)
code("""def zeros(n: int) -> list[float]:
    return [0.0] * (n + 1)

g.dx = g.Length / g.NC
for i in range(1, g.NC + 1):
    g.xc[i] = g.dx * (i - 0.5)""",
     "Index zero is deliberately unused. Cell i has centre x_i = (i - 0.5) dx. The extra slot keeps formulas and loops close to the VBA source and makes Excel parity easier to audit.")
p("The vertical area is width × depth. A cell volume is width × depth × dx. These dimensions explain why equal discharge mass produces a smaller concentration change in deeper or wider water.")

h("Configuration and validation", 2)
code("""c.dt = _float(data.get("time_step_seconds"), "Time step")
c.SimulationDuration = _float(
    data.get("simulation_duration_days"), "Duration"
) * 24 * 3600
c.DtPrint = _float(data.get("print_interval_seconds"), "Print interval")""",
     "The app exposes duration in days while the numerical loop uses seconds. Validation rejects nonfinite values, impossible indices, and cases that would leave the VBA loop running forever.")
code("""f.velocity = rate / (gr.RiverWidth * gr.waterdepth)
f.CourantNr = ctrl.dt * f.velocity / gr.dx if ctrl.Advection else 0.0
f.DiffusionNr = ctrl.dt * f.Diffusivity / gr.dx ** 2 if ctrl.Diffusion else 0.0""",
     "Flow rate divided by cross sectional area yields mean velocity. The process switches determine whether the corresponding dimensionless transport number participates in the calculation.")

h("The step sequence", 2)
code("""while ctrl.T < ctrl.SimulationDuration:
    ctrl.T += ctrl.dt
    zero_coeffs(coef, gr)
    # For each active property: apply lateral discharges.
    coef_transport(flow, gr.NC, ctrl, coef)
    # For each active property: solve longitudinal transport.
    # Then heat, gas, BOD, and Generic sources.
    if ctrl.T >= ctrl.printtime:
        print_values()""",
     "This abridged excerpt shows the order, not a replacement implementation. Source mixing precedes transport. Heat and gas exchange and biochemical reactions follow it. The solver prints time zero before the loop and then at the first step meeting each output threshold.")
p("The property order is Generic, Temperature, DO, BOD, CO2. The shared transport stencil is calculated once per step. Property arrays change in place, so process order matters when a later source reads an earlier updated value.")

h("Longitudinal transport schemes", 2)
p("Diffusion adds equal neighbour contributions D and an interior centre contribution -2D. Upwind advection chooses the upstream side of each face; central differences divide the contribution between neighbours. QUICK estimates face values with a quadratic upstream stencil and therefore reaches two cells away. QUICK_UP can replace that stencil locally with upwind near a sharp gradient.")
code("""if ctrl.AdvType == "upwind":
    if flow.velocity > 0:
        coef.B[i] += flow.CourantNr
        coef.e[i] -= flow.CourantNr
    else:
        coef.e[i] += flow.CourantNr
        coef.f[i] -= flow.CourantNr""",
     "Positive flow takes information from the left cell; negative flow reverses the direction. The implemented code uses one based coefficient arrays and separately handles boundary cells.")
p("Explicit transport evaluates the entire next field from a frozen Old field. Implicit transport places the new neighbour values in a tridiagonal linear system solved by forward elimination and back substitution. Semi implicit transport splits the stencil between Old and New. QUICK variants require a five diagonal elimination because their stencil spans two extra cells.")
p("A stable looking plot is not proof of physical accuracy. Compare a no diffusion advected patch with the original shape, inspect undershoots and overshoots, and repeat at smaller dx and dt.")

h("Lateral mixing and surface exchange", 2)
code("""Prop.NewValue[ncell] = (
    Prop.NewValue[ncell] * (gr.AreaVertical * gr.dx)
    + volume_rate * specific_value * dt
) / (gr.AreaVertical * gr.dx + volume_rate * dt)""",
     "The numerator adds old property content and incoming property content. The denominator is the combined volume for this step. Multiple discharges into one cell are applied sequentially in list order.")
code("""Temperature.NewValue[i] += (
    ctrl.dt * solar_flux / (ro * cp * gr.waterdepth)
)""",
     "A surface heat flux divided by water density, heat capacity, and depth becomes a temperature change rate for the entire represented water column. The macro then applies sensible, latent, and long wave terms in sequence.")
code("""csat = getCsat_Henry(Gas, Temperature.NewValue[i])
Gas.NewValue[i] = (
    Gas.NewValue[i] + ctrl.dt * k_gas_exchange * csat
) / (1 + ctrl.dt * k_gas_exchange)""",
     "Henry interpolation gives the equilibrium gas concentration. The implicit update relaxes the cell toward equilibrium; the executed coefficient depends on wind, slope, depth, and any supersaturation enhancement.")

h("Coupled BOD, oxygen, and carbon", 2)
code("""bod_decay = pr.BOD.NewValue[i] * decay_rate * dt
pr.BOD.NewValue[i] = bod_before_decay - bod_decay
pr.DO.NewValue[i] = pr.DO.NewValue[i] - bod_decay
pr.CO2.NewValue[i] = pr.CO2.NewValue[i] + bod_decay * (44 / 32)""",
     "Aerobic respiration removes BOD and the same mass unit of oxygen. The 44/32 molecular weight ratio converts the oxygen equivalent into CO2 mass. The routine limits consumption by available oxygen and BOD before these assignments.")
p("When anaerobic respiration is enabled, further BOD is removed without consuming DO. The simplified macro counts half of that loss toward CO2. Integrated aerobic and anaerobic respiration outputs sum each cell's reacted mass through time. These are deliberately simplified biochemistry, not a full aquatic ecosystem model.")

h("Charts, animation, and errors", 2)
p("charts.py reads the current SimulationResult and builds serializable workbook style chart specifications. This prevents old CO2 worksheet values from appearing in a new run, a known macro chart bug. visualization3d.py repeats each one dimensional cell value across top, side, and end faces, and holds the color scale fixed across animation frames.")
p("A SimulationError attaches stage, property, step, time, and cell to the original exception. The app converts this into a readable explanation, suggested checks, and a Technical details expander. The original cause is preserved for debugging.")

h("Evidence that the port matches the workbook", 2)
p("The test suite checks every printed cell in 89 recorded Excel cases, including coupled and extreme settings, and also compares the saved 1,000 cell workbook run. Chart tests compare series data for cases calculated by both systems. Floating point parity uses a relative tolerance of 1e-14 or absolute tolerance of 1e-12 to allow the last binary rounding bits to differ. The standalone test runs the app without the XLSM.")
p("Install requirements.txt and requirements-dev.txt, then run python -m pytest -q. The test suite also runs the teaching presets, checks selected analytical and numerical comparisons, and exercises the Streamlit case loader.")

h("Reading the revised theoretical notes", 2)
bullets([
    "The revised notes use the workbook’s wind, slope and depth dependence for gas exchange, including the supersaturation correction.",
    "The solar term includes 23% atmospheric absorption and the cloud factor 1 − 0.75 F³.",
    "The printed CO₂ Henry constant at 20 °C is 0.039172 mol/(L·atm).",
    "The time loop mixes lateral discharges before transport. Aerobic oxidation supplies CO₂ and consumes oxygen.",
])

h("Study exercises", 2)
steps([
    "Turn off both advection and diffusion. Explain which terms can still change each active property.",
    "Increase depth while keeping the same lateral source and other inputs. Predict and verify the concentration response.",
    "Enable advection and change its sign. Identify the upstream boundary and compare profile movement.",
    "Run the same initial patch with upwind, central, QUICK, and QUICK_UP. Compare spreading and overshoot.",
    "Halve dt and dx separately. Calculate the new Cr and D before running, then compare the results.",
    "Enable DO, BOD, and CO2 in a saved case. Trace one aerobic decay step through all three arrays.",
    "Read one row of the numerical table and locate its time and cell in a profile, time series, map, and 3D frame.",
])

h("Glossary", 2)
table(["Term", "Meaning"], [
    ["Cell", "One longitudinal river segment representing its whole cross section."],
    ["Printed time", "A saved result row; not necessarily every computation step."],
    ["Courant number", "Mean advective travel per step divided by cell length."],
    ["Diffusion number", "Longitudinal diffusivity times dt divided by dx squared."],
    ["Boundary value", "Prescribed value used at a reach end in the transport stencil."],
    ["Henry constant", "Temperature dependent proportionality between gas pressure and equilibrium dissolved concentration."],
    ["Operator splitting", "Applying transport, exchange, and reaction in successive updates within a step."],
])
doc.add_page_break()
h("Part 4 Teaching case catalogue", 1)
p("Choose the topic, select a case, then select Load case and Run simulation. Each case opens with editable inputs. Load it again to restore its baseline. Save each changed configuration before comparing two sets of results.")
figure("14_case_picker.jpg", "Figure 13  Selecting a lecture case.",
       "The case description gives the lecture and the feature to look for in the results.")
p("The lecture numbers identify the teaching material. Some slides give a plotted result and only part of the input. Those presets reconstruct the demonstration. For the six-cell numerical exercise, compare the interior values first. The workbook’s end-cell stencils differ from the ghost-cell assumptions used in the hand calculation.")
p("In Lecture 30, the printed decimal and centimetres-per-kilometre slope values disagree. The presets use the decimals as workbook entries, reflecting the similar high-wind curves shown in the slides. The pond example in Lecture 31 uses three 200 m tanks. The two receiving-water examples interpret 50 m and 3 m as channel widths because those widths give the stated flow rates.")
p("The model has a fixed cross-section. Exercises on shrinking water volume, vertical stratification and mechanical aeration require additional physics. The stronger pond exchange case varies wind to illustrate how an increase in atmospheric exchange affects DO.")
figure("15_bod_timeseries.jpg", "Figure 14  Oxygen recovery in the high-wind BOD case.",
       "Select DO, then open Interactive graphs and Cell time series. Compare the minimum and recovery with the low-wind case.")
cases = teaching_cases()
for topic in dict.fromkeys(case.topic for case in cases):
    h(topic, 2)
    table(["Case", "Lecture", "What to look for"],
          [[case.title, case.lecture, case.focus] for case in cases if case.topic == topic],
          widths=(2.15, 0.88, 3.72))

doc.save(OUTPUT)
print(OUTPUT)
