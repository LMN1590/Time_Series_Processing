import gradio as gr
import pandas as pd
import numpy as np  
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


from model_module import PatientModelModule
from preprocess.preprocess_data import preprocess_time_data
from preprocess.const import LOG_COLS,STANDARD_SCALE_CONST,SCALE_COLS,ROBUST_SCALE_CONST
from sklearn.preprocessing import StandardScaler,RobustScaler

model = PatientModelModule.load_from_checkpoint(
    checkpoint_path="inference_utils/Switch_Former_epoch(34)_step(1050)_val_0.1495.ckpt",
    hparams_file=f"inference_utils/hparams.yaml"
).eval()

precision_df = pd.read_csv("./data/scale_result_filtered_2.5.csv")
precision = list(precision_df["precision"])

LOWER_SCALE_VAL = 5
UPPER_SCALE_VAL = 11

FLOAT_COLS = [
    "Pulse",
    "Systolic",
    "Diastolic",
    "Temperature",
    "BMI",
    "RespiratoryRate",
    "OxygenSaturation",
    "OxygenConcentration",
    
    "CancelledRateThePast6Months",
    "RescheduledRateThePast6Months",
    "CancelledAppointmentsSinceLastEncounter",
    "RescheduledAppointmentsSinceLastEncounter",
    "Average Duration PCP Visit"
]

def inference(df:pd.DataFrame, sex:str,ethnicity:str,dob:str,expected_output:int):
    df["Sex"] = [sex]*df.shape[0]
    df["Ethnicity"]=[ethnicity]*df.shape[0]
    df["DOB"] = [dob]*df.shape[0]
    df["EncounterId"] = [0.0]*4
    df["PatientId"] = [0.0]*4
    df["TimeSinceLastVisit(Day)"] = [0.0] * 4
    
    df[FLOAT_COLS] = df[FLOAT_COLS].replace(r'^\s*$', np.nan, regex=True).astype(float)

    input_data_preprocessed = preprocess_time_data(df)
    input_data_preprocessed[LOG_COLS] = np.log(input_data_preprocessed[LOG_COLS]+1)

    # region Robust Scaler
    scaler = RobustScaler()
    scaler.center_ = ROBUST_SCALE_CONST["center"]
    scaler.scale_ = ROBUST_SCALE_CONST["scale"]
    # endregion

    input_data_preprocessed[list(SCALE_COLS.keys())] = scaler.transform(input_data_preprocessed[list(SCALE_COLS.keys())])
    input_data_preprocessed = input_data_preprocessed.drop(list(input_data_preprocessed.columns[[0,1,2]]) + ["Target","Target_shift_1"],axis=1).values.astype(float)
    input_data_preprocessed = torch.from_numpy(input_data_preprocessed).unsqueeze(0).float()
    
    with torch.no_grad():
        res = model(input_data_preprocessed)


    mean = res[0]
    var = res[1]
    # # region Standard Scaler
    # center = STANDARD_SCALE_CONST["mean"]
    # scale = STANDARD_SCALE_CONST["scale"]
    # # endregion

    # region Robust Scaler
    center = ROBUST_SCALE_CONST["center"][-1]
    scale = ROBUST_SCALE_CONST["scale"][-1]
    # endregion
    mean_unnorm = mean*scale+center
    var_unnorm = var*(scale**2)

    mean_unlog = torch.exp(mean_unnorm + (var_unnorm)/2)-1
    var_unlog = torch.exp(2*mean_unnorm)*(torch.exp(2*var_unnorm) - torch.exp(var_unnorm))
    
    mean_final = mean_unlog[0].item()
    std_final = torch.sqrt(var_unlog)[0].item()
    
    lower_lim = 0
    upper_lim = max(expected_output,mean_final + std_final * UPPER_SCALE_VAL) + 10
    plt.xlim(lower_lim,upper_lim)
    plt.ylim(0,1)
    range_val = upper_lim - lower_lim

    warning_flag = (mean_final + std_final * LOWER_SCALE_VAL - lower_lim)/range_val
    lost_flag = (mean_final + std_final * UPPER_SCALE_VAL - lower_lim)/range_val

    LW = 5
    
    image_path = "legend.png"  # Replace with your image path
    img = mpimg.imread(image_path)

    plt.imshow(img, extent=[0, upper_lim*0.55, 0.65, 1], aspect='auto', alpha=1)
    plt.axhline(y=0.5,xmin=0,xmax=warning_flag,color="g",lw=LW)
    plt.axhline(y=0.5,xmin=warning_flag,xmax=lost_flag,color="gold",lw=LW)
    plt.axhline(y=0.5,xmin = lost_flag, xmax = 1,color="r",lw=LW)

    plt.axvline(x=expected_output,ymin=0.4,ymax=0.6,lw=LW,color = "lime",label="Current Time")
    plt.axvline(x=mean_final + std_final * LOWER_SCALE_VAL - lower_lim,ymin=0.4,ymax=0.6,lw=LW, color = "b",label="Barriers")
    plt.axvline(x=mean_final + std_final * UPPER_SCALE_VAL - lower_lim,ymin=0.4,ymax=0.6,lw=LW, color = "b")

    plt.legend()
    plt.xlabel("Days since Last Visit")      
    
    plt.savefig("plot.png")
    plt.clf()
    
    scale = (expected_output-mean_final)/std_final
    lower_scale = math.floor(scale)
    upper_scale = math.ceil(scale)
    if(lower_scale >= len(precision)-1): precision_final = 1
    elif(scale<0): precision_final =  precision[0]
    else: precision_final = precision[lower_scale] + (scale-lower_scale)*(precision[upper_scale]-precision[lower_scale])
    return precision_final*100,std_final*LOWER_SCALE_VAL,std_final*(UPPER_SCALE_VAL-LOWER_SCALE_VAL),"plot.png"


gr.set_static_paths(paths=["./legend.png"])
demo = gr.Interface(
    inference,
    [
        gr.Dataframe(
            headers=[
                "EncounterDt",
                "CurrentVisitICDs",
                "ThePast6MonthsICDs",
                "Allegies",
                "CustomAllegies",
                "Primary Claim Type",
                "Secondary Claim Type",
                "Vaccinations",
                "Pulse",
                "Systolic",
                "Diastolic",
                "Temperature",
                "BMI",
                "RespiratoryRate",
                "OxygenSaturation",
                "OxygenConcentration",
                
                "CancelledRateThePast6Months",
                "RescheduledRateThePast6Months",
                "CancelledAppointmentsSinceLastEncounter",
                "RescheduledAppointmentsSinceLastEncounter",
                "Average Duration PCP Visit",
                "EncounterType",
                "LocationType"
            ],
            datatype=[
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                
                "number",
                "number",
                "number",
                "number",
                "number",
                
                "str",
                "str"
            ],
            row_count=4,
            col_count=(23, "fixed"),
            type="pandas"
        ),
        gr.Radio(["M","F","U"], label="Sex", info="Patient's Sex"),
        gr.Radio([
            'Declined','Hispanic or Latino', 
            'Not Hispanic or Latino', 
            'Not asked'
        ], label="Ethnicity", info="Patient's Ethnicity"),
        gr.Textbox(label="Date of Birth"),
        gr.Number(label="Time since Last Visit")
    ],
    
    [
        gr.Number(label="Losing Patient Percentage(%)"),
        gr.Number(label="Patient Safe Time(Days)"),
        gr.Number(label="Patient Warning Time(Days)"),
        gr.Image(label="Plot", type="filepath")
    ],
    examples=[
        [
            pd.DataFrame({
                "EncounterDt":[
                    "2022-2-3 15:00:00",
                    "2022-2-8 15:00:00",
                    "2022-3-11 13:15:00",
                    "2022-5-26 11:15:00"
                ],
                "CurrentVisitICDs":[
                    "A09",
                    "A09; E1143; E119; E785; H269; I10; K2970; K5900; M5440; N529",
                    "E119; G20",
                    "E119; E559; E785; G20; I10; M5440; N529; Z0001"
                ],
                "ThePast6MonthsICDs":[
                    "A09; E1143; E119; E785; H269; I10; K2970; K5900; M5440; N529; Z029; Z1211; Z760",
                    "A09; E1143; E119; E785; H269; I10; K2970; K5900; M5440; N529; Z029; Z1211; Z760",
                    "A09; E1143; E119; E785; G20; H269; I10; K2970; K5900; M5440; N529; Z029; Z1211; Z760",
                    "A09; E1143; E119; E559; E785; G20; H269; I10; K2970; K5900; M5440; N529; Z0001"

                ],
                "Allegies":["Empty"]*4,
                "CustomAllegies":["Empty"]*4,
                "Primary Claim Type":[
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk"
                ],
                "Secondary Claim Type":["Medicaid"]*4,
                "Vaccinations":["Empty"]*4,
                "Pulse":[None,None,91,81],
                "Systolic"	:[None,None,121,81],
                "Diastolic"	:[None,None,67,60],
                "Temperature": [None,None,None,97.7],
                "BMI":[None,None,29.1,29.1],
                "RespiratoryRate":[None,None,16,15],
                "OxygenSaturation":[None,None,97,98],
                "OxygenConcentration":[None,None,21,21],

                "CancelledRateThePast6Months":[
                    0.428571429,
                    0.375,
                    0.428571429,
                    0.375,
                ],
                "RescheduledRateThePast6Months":[0,0,0,0],
                "CancelledAppointmentsSinceLastEncounter":[0,1,2,0],
                "RescheduledAppointmentsSinceLastEncounter":[0,0,0,0],
                "Average Duration PCP Visit":[
                    28.33333333,
                    22.5,
                    24.2,
                    39.4
                ],
                "EncounterType":[
                    "Follow up",
                    "Follow up",
                    "Follow up",
                    "Annual"
                ],
                "LocationType":[
                    "Tele",
                    "Tele",
                    "Office",
                    "Office"
                ]
            }),
            "M",
            "Hispanic or Latino",
            "7/3/1945",196
        ],
        [
            pd.DataFrame({
                "EncounterDt":[
                    "2022-5-23 11:45:00",
                    "2022-6-1 12:15:00",
                    "2022-9-14 15:15:00",
                    "2022-9-30 12:00:00"
                ],
                "CurrentVisitICDs":[
                    "C641; K769; Z0000",
                    "C641; F0390; N182",
                    "E039; F0390; N182",
                    "N390"
                ],
                "ThePast6MonthsICDs":[
                    "C641; E039; E7800; F0390; I10; K769; Z0000",
                    "C641; E039; E7800; F0390; I10; K769; N182; Z0000",
                    "C641; E039; E7800; F0390; I10; K769; N182; Z0000",
                    "C641; E039; E7800; F0390; I10; K769; N182; N390; Z0000"

                ],
                "Allegies":["Empty"]*4,
                "CustomAllegies":["Empty"]*4,
                "Primary Claim Type":[
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk"
                ],
                "Secondary Claim Type":["Medicaid"]*4,
                "Vaccinations":["Empty"]*4,
                "Pulse":[65,None,None,None],
                "Systolic"	:[91,None,None,None],
                "Diastolic"	:[54,None,None,None],
                "Temperature": [98.2,None,None,None],
                "BMI":[24,None,None,None],
                "RespiratoryRate":[12,None,None,None],
                "OxygenSaturation":[98,None,None,None],
                "OxygenConcentration":[21,None,None,None],

                "CancelledRateThePast6Months":[
                    0.25,
                    0.2,
                    0.142857143,
                    0.125,
                ],
                "RescheduledRateThePast6Months":[0,0,0,0],
                "CancelledAppointmentsSinceLastEncounter":[0,0,0,0],
                "RescheduledAppointmentsSinceLastEncounter":[0,0,0,0],
                "Average Duration PCP Visit":[
                    64.66666667,
                    50.75,
                    37.66666667,
                    34.57142857
                ],
                "EncounterType":[
                    "Follow up",
                    "Follow up",
                    "Follow up",
                    "Follow up"
                ],
                "LocationType":[
                    "Office",
                    "Tele",
                    "Tele",
                    "Tele",
                ]
            }),
            "F",
            "Hispanic or Latino",
            "8/27/1936",42
        ],
        [
            pd.DataFrame({
                "EncounterDt":[
                    "2022-3-17 13:00:00",
                    "2022-7-28 15:45:00",
                    "2022-9-15 15:00:00",
                    "2022-9-20 15:30:00"
                ],
                "CurrentVisitICDs":[
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R630; Z0000",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R630; Z0000; Z1239",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R2240; R630; Z0000; Z1239",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R2240; R630; Z0000; Z1239"
                ],
                "ThePast6MonthsICDs":[
                    "D649; E039; E538; E559; E782; E785; E871; E876; G4700; G629; I10; I2510; I639; I739; J209; J40; K219; K760; M109; M170; M2550; M519; N189; N3281; N3289; R413; R600; R630; R68; R740; Z0000; Z01818; Z09; Z712; Z96659",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R630; R68; Z0000; Z1239",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R2240; R630; R68; Z0000; Z1239",
                    "D649; E039; E538; E559; E785; G4700; G629; I10; I2510; I639; K219; M109; M170; M519; N3281; R053; R2240; R630; R68; Z0000; Z1239"
                ],
                "Allegies":["Empty"]*4,
                "CustomAllegies":["Empty"]*4,
                "Primary Claim Type":[
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk",
                    "Health Maintenance Organization (HMO) Medicare Risk"
                ],
                "Secondary Claim Type":["Medicaid"]*4,
                "Vaccinations":["Empty"]*4,
                "Pulse":[None,70,None,None],
                "Systolic"	:[None,126,122,None],
                "Diastolic"	:[None,80,76,None],
                "Temperature": [None,None,None,None],
                "BMI":[None,0,0,None],
                "RespiratoryRate":[None,12,None,None],
                "OxygenSaturation":[None,None,None,None],
                "OxygenConcentration":[None,None,None,None],

                "CancelledRateThePast6Months":[
                    0.285714286,
                    0.333333333,
                    0.25,
                    0.4,
                ],
                "RescheduledRateThePast6Months":[0,0,0,0],
                "CancelledAppointmentsSinceLastEncounter":[0,1,0,1],
                "RescheduledAppointmentsSinceLastEncounter":[0,0,0,0],
                "Average Duration PCP Visit":[
                    21.16666667,
                    47.5,
                    70,
                    62.33333333,
                ],
                "EncounterType":[
                    "Follow up",
                    "Annual",
                    "Follow up",
                    "Follow up"
                ],
                "LocationType":[
                    "Tele",
                    "Home",
                    "Home",
                    "Tele",
                ]
            }),
            "F",
            "Not asked",
            "8/4/1945",27
        ],
        [
            pd.DataFrame({
                "EncounterDt":[
                    "2022-11-14 16:00:00",
                    "2022-12-6 16:00:00",
                    "2022-12-20 15:00:00",
                    "2023-1-3 15:00:00"
                ],
                "CurrentVisitICDs":[
                    "F419",
                    "F329; F419",
                    "F419",
                    "F419",
                ],
                "ThePast6MonthsICDs":[
                    "E049; F329; F419; L709; M25511; M419; M7918; N83201; R000; Z0000; Z13220; Z712",
                    "E049; F329; F419; L709; M25511; M419; M7918; N83201; R000; Z0000; Z13220; Z712",
                    "E049; F329; F419; L709; M25511; M419; M7918; N83201; R000; Z0000; Z13220; Z712",
                    "E049; F329; F419; L709; M25511; M419; M7918; N83201; R000; Z0000; Z13220; Z712",
                ],
                "Allegies":["Empty"]*4,
                "CustomAllegies":["Empty"]*4,
                "Primary Claim Type":[
                    "Commercial Insurance Co."
                ]*4,
                "Secondary Claim Type":[None]*4,
                "Vaccinations":["Empty"]*4,
                "Pulse":[None,None,None,None],
                "Systolic"	:[None,None,None,None],
                "Diastolic"	:[None,None,None,None],
                "Temperature": [None,None,None,None],
                "BMI":[None,None,None,None],
                "RespiratoryRate":[None,None,None,None],
                "OxygenSaturation":[None,None,None,None],
                "OxygenConcentration":[None,None,None,None],

                "CancelledRateThePast6Months":[
                    0,0,0,0
                ],
                "RescheduledRateThePast6Months":[0,0.142857143,0.125,0.111111111],
                "CancelledAppointmentsSinceLastEncounter":[0,0,0,0],
                "RescheduledAppointmentsSinceLastEncounter":[0,1,0,0],
                "Average Duration PCP Visit":[
                    43.2,
                    39.66666667,
                    36,
                    33.25,
                ],
                "EncounterType":[
                    "Follow up",
                    "Follow up",
                    "Follow up",
                    "Follow up"
                ],
                "LocationType":[
                    "Tele",
                    "Tele",
                    "Tele",
                    "Tele",
                ]
            }),
            "F",
            "Not Hispanic or Latino",
            "5/14/2000",7
        ],
        [
            pd.DataFrame({
                "EncounterDt":[
                    "2022-2-17 11:15:00",
                    "2022-2-28 9:15:00",
                    "2022-3-11 13:45:00",
                    "2022-5-13 14:30:00"
                ],
                "CurrentVisitICDs":[
                    "E1169; E782; I10; R9431",
                    "B351; D259; D751; E1169; E55; E782; E8352; F419; I10; J302; M25512; M25562; M5382; M545; M546; M79642; M79651; M79652; N840; R059; R202; R229; R42; R928; R9431; Z0000",
                    "B351; D259; D751; E1169; E55; E782; E8352; F419; I10; J302; M25512; M25562; M5382; M545; M546; M79642; M79651; M79652; N840; R059; R202; R229; R42; R928; R9431; Z0000",
                    "B351; D259; D751; E1169; E55; E782; E8352; F419; I10; J302; M25512; M25562; M5382; M545; M546; M79642; M79651; M79652; N840; R059; R202; R229; R42; R928; R9431; Z0000",
                ],
                "ThePast6MonthsICDs":[
                    "B351; D259; D751; E1169; E119; E55; E782; E8352; F419; I10; J028; J029; J302; J40; L0591; M25512; M25562; M5382; M545; M546; M62838; M79642; M79651; M79652; N840; R05; R109; R202; R21; R229; R42; R7989; R928; R938; R9431; Z0000; Z01419; Z01810; Z0184; Z1159",
                    "B351; D259; D751; E1169; E119; E55; E782; E8352; F419; I10; J028; J029; J302; J40; L0591; M25512; M25562; M5382; M545; M546; M62838; M79642; M79651; M79652; N840; R05; R059; R109; R202; R21; R229; R42; R7989; R928; R938; R9431; Z0000; Z01419; Z01810; Z0184; Z1159",
                    "B351; D259; D751; E1169; E119; E55; E782; E8352; F419; I10; J028; J029; J302; J40; L0591; M25512; M25562; M5382; M545; M546; M62838; M79642; M79651; M79652; N840; R05; R059; R109; R202; R21; R229; R42; R7989; R928; R938; R9431; Z0000; Z01419; Z01810; Z0184; Z1159",
                    "B351; D259; D751; E1169; E55; E782; E8352; F419; I10; J302; M25512; M25562; M5382; M545; M546; M79642; M79651; M79652; N840; R059; R202; R229; R42; R928; R9431; Z0000"

                ],
                "Allegies":["Empty"]*4,
                "CustomAllegies":["Empty"]*4,
                "Primary Claim Type":[
                    "Health Maintenance Organization (HMO) Medicare Risk"
                ]*4,
                "Secondary Claim Type":["Medicaid"]*4,
                "Vaccinations":["Empty"]*4,
                "Pulse":[69,70,80,None],
                "Systolic"	:[135,120,130,None],
                "Diastolic"	:[75,64,81,None],
                "Temperature": [None,97.2,97.3,None],
                "BMI":[23.9,23.9,23.9,None],
                "RespiratoryRate":[16,16,16,None],
                "OxygenSaturation":[None,96,97,None],
                "OxygenConcentration":[None,21,21,None],

                "CancelledRateThePast6Months":[
                    0.272727273,
                    0.1,
                    0.090909091,
                    0.153846154,
                ],
                "RescheduledRateThePast6Months":[
                    0.090909091,
                    0.3,
                    0.272727273,
                    0.307692308,

                ],
                "CancelledAppointmentsSinceLastEncounter":[
                    0,
                    0,
                    0,
                    1
                ],
                "RescheduledAppointmentsSinceLastEncounter":[
                    0,
                    2,
                    0,
                    1
                ],
                "Average Duration PCP Visit":[
                    33.33333333,
                    27.75,
                    24.4,
                    37,
                ],
                "EncounterType":[
                    "Follow up",
                    "Follow up",
                    "Follow up",
                    "Follow up"
                ],
                "LocationType":[
                    "Office",
                    "Office",
                    "Office",
                    "Tele",
                ]
            }),
            "F",
            "Hispanic or Latino",
            "8/2/1952",742
        ]
    ],
    # description="<img src='/file=./legend.png'>",

)

demo.launch(server_name="0.0.0.0",server_port=7860,allowed_paths=["./legend.png"])
    