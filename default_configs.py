# This file contains the raw CSV content from the provided Excel file.
# We cleaned up the excessive trailing commas to prevent parsing errors.

DEFAULT_CSVS = {
    "Main": """Simulation Options,
,ExecutableFile:,C:\\EXCEL_TEM\\Channel_TempOD&BOD&CO2SourceCode2020.xlsm
SimulationDuration(Days):,1,Days
TimeStep(Seconds):,200,s
Dtprint(seconds):,3600,s
TimeDiscretisation:,semi,semi/imp/exp
Advection:,Yes,yes/no
AdvectionType:,QUICK,upwind /central/QUICK/QUICK_UP
QUICK_UP_Ratio:,4,
Diffusion:,yes,Typically between 2 and 10
,,yes/no""",

    "River": """River & Flow Properties,
ChannelLength:,12000,m
NumberOfCells:,300,-
RiverWidth:,100,m,"Flow and Velocity are computed..."
WaterDepth:,0.5,m
RiverSlope:,0.0001,m/m
ManningCoef(n):,0.025,m(-1/3)s
Discharge:,12.515,m3/s
FlowVelocity:,0.2503,m/s
Diffusivity:,1.0012,m2/s
Hydraulic Radius:,0.495,
Equation for,diffusivity,: 0.01+velocity*width""",

    "Atmosphere": """Atmospheric Data,
,
AirTemperature:,20,C
WindSpeed:,0,m/s
h_min:,6.9,
AirHumidity:,80,%
SolarConstant:,1370,W/m2
Latitude:,38,Degrees
SkyTemperature:,-40,C
CloudCover:,0,%
SunRizeHour:,6,hours
SunSetHour:,18,hours
,
O2PartialPressure:,0.2095,bar,MolecularWeightO2,32000,mg/mole
CO2PartialPressure:,0.000395,bar,MolecularWeightCO2,44000,
,
HenryConstants:,Temperature,O2,CO2,Units
,0,0.0021812,0.076425,M/atm
,5,0.0019126,0.063532,M/atm
,10,0.0016963,0.05327,M/atm
,15,0.0015236,0.045463,M/atm
,20,0.001384,0.039172,M/atm
,25,0.001263,0.033363,M/atm""",

    "Discharges": """DischargeNumbers:,1,2,3,4
DischargeNames:,Descarga 1,Descarga2,Descarga3,Descarga4
DischargeCells:,1,60,100,140
!!!DischargeLocations(km):,5,20,40,80
DischargeFlowRates(m3/s):,0,0,0,0
DischargeTemperatures:,30,30,50,50,ºC
DischargeConcentrations_DO:,0,0,0,0,mg/L
DischargeConcentrations_BOD:,100,100,200,200,mg/L
DischargeConcentrations_CO2:,1,1,1,1,mg/L
DischargeGeneric:,100000,100000,100000,100000,""",

    "Temperature": """PropertyName: ,Temperature,
PropertyUnits:,ºC,
PropertyActive:,Yes,Yes/No
DefaultValue:,15,
MaximumValue:,45,""",

    "DO": """PropertyName: ,DO,
PropertyUnits:,mg/L,
PropertyActive:,Yes,Yes/No
DefaultValue:,0,""",

    "BOD": """PropertyName: ,BOD,
PropertyUnits:,mg/L,
PropertyActive:,Yes,Yes/No
DefaultValue:,5,
MaximumValue:,400,""",

    "CO2": """PropertyName: ,CO2,
PropertyUnits:,mg/L,
PropertyActive:,Yes,Yes/No
DefaultValue:,0.7,"""
}
