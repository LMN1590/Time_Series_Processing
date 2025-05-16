import pandas as pd
from datetime import date,datetime
from pyzipcode import ZipCodeDatabase
from functools import reduce
from datetime import datetime
import numpy as np

from .const import ICD_DICT,RACE,VACCINATION,SCALE_COLS
from .dummies_columns import DUMMY_COLS

class Temp:
    def __init__(self):
        self.state = "Unknown"

ZCDB = ZipCodeDatabase()

def preprocess_time_data(df:pd.DataFrame):
    try:
        df["EncounterDt"] = df["EncounterDt"].apply(lambda x:datetime.strptime(x, "%y-%m-%d %H:%M:%S"))
    except Exception as e:
        df["EncounterDt"] = df["EncounterDt"].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df = df.sort_values(by=["PatientId","EncounterDt"], ignore_index=True)

    cleaned_df = df[["EncounterId","PatientId","EncounterDt"]]

    # region Age
    df_age = df[["DOB","EncounterDt"]].apply(
        lambda row:(
            row['EncounterDt'].to_pydatetime()
            -datetime.strptime(row["DOB"],"%m/%d/%Y")
        ).days/365
    ,axis=1)
    cleaned_df = cleaned_df.join(df_age.rename("Age").to_frame())
    # endregion

    # # region Race
    # df_race = pd.DataFrame()
    # for race_group in RACE:
    #     display = race_group["display"]
    #     child_concepts = [race["code"] for race in race_group["child_concept"]] + [race_group["code"]]
        
    #     df_race[f'Race One-hot {display}'] = df["Race"].apply(lambda race_code: int(race_code in child_concepts))
        
    # df_race["Other Race"] = df_race.apply(lambda row:int(not np.any(row.values)),axis=1)

    # cleaned_df = cleaned_df.join(df_race)
    # # endregion

    # region Ethnicity
    df["Ethnicity"] = df["Ethnicity"].fillna("Not asked")
    # endregion

    # # region Zip Code
    # ZIP_CODE = {item: ZCDB.get(int(str(item).split("-")[0]),Temp()).state for item in df["Zip"].unique().tolist()}
    # df["Zip"] = df["Zip"].apply(lambda x:ZIP_CODE[x])
    # # endregion

    pt_ids = df["PatientId"].unique()
    df_pt = df.set_index("PatientId")
    for pt in pt_ids:
        try:
            target_shift_1 = df_pt.loc[pt,"TimeSinceLastVisit(Day)"].shift(-1)
            target_shift_1.iloc[-1] = (datetime.strptime("10/30/2024", "%m/%d/%Y") - df_pt.loc[pt,"EncounterDt"].iloc[-1].to_pydatetime()).days
            df_pt.loc[pt,"Target_shift_1"] = target_shift_1
            # print(df_pt.loc[pt,"Target_shift_1"])
        except Exception as e:
            print(e)
            df_pt.loc[pt,"Target_shift_1"] = np.nan
    df = df_pt.reset_index()
    # print(df["Target_shift_1"])
    cleaned_df["Target_shift_1"] = df["Target_shift_1"]




    # region Vitals
    vitals = df[[
        "Pulse",
        "Systolic",
        "Diastolic",
        "Temperature",
        "BMI",
        "RespiratoryRate",
        "OxygenSaturation",
        "OxygenConcentration"
    ]]
    vitals = vitals.clip(lower = vitals.quantile(0.05), upper = vitals.quantile(0.95),axis = 1)
    vitals = vitals.fillna({col:SCALE_COLS[col]["mean"] for col in vitals.columns})
    cleaned_df = cleaned_df.join(vitals)
    # endregion

    # region Scheduling Stats
    sched = df[[
        "CancelledRateThePast6Months",
        "RescheduledRateThePast6Months",
        "CancelledAppointmentsSinceLastEncounter",
        "RescheduledAppointmentsSinceLastEncounter"
    ]]
    cleaned_df = cleaned_df.join(sched)
    # endregion

    # region Ground-truth
    last_day = df["TimeSinceLastVisit(Day)"]
    cleaned_df = cleaned_df.join(last_day.rename("Target").to_frame())

    # endregion

    # region ICDs
    icd_groupings_current = pd.DataFrame()
    icds = df[["CurrentVisitICDs"]].apply(
        lambda row: 
            [code.strip()[:2] for code in row["CurrentVisitICDs"].split(';')] 
            if (isinstance(row["CurrentVisitICDs"],str)) 
            else [],
        axis=1
    )
    for group_name, group_prefixes in ICD_DICT.items():
        icd_groupings_current["Current "+group_name] = icds.apply(
            lambda x: int(any(prefix in group_prefixes for prefix in x))
        )
    icd_groupings_current["Current ICD Other"] = icds.apply(
        lambda x: int(any(prefix not in reduce(lambda arr,x:arr+x,ICD_DICT.values(),[]) for prefix in x))
    )
    icd_groupings_current["Current ICD Count"] = icds.apply(
        lambda x:len(x)
    )

    icd_groupings_6month = pd.DataFrame()
    icds = df[["ThePast6MonthsICDs"]].apply(
        lambda row: 
            [code.strip()[:2] for code in row["ThePast6MonthsICDs"].split(';')] 
            if (isinstance(row["ThePast6MonthsICDs"],str)) 
            else [],
        axis=1
    )
    for group_name, group_prefixes in ICD_DICT.items():
        icd_groupings_6month["6months "+group_name] = icds.apply(
            lambda x: int(any(prefix in group_prefixes for prefix in x))
        )
    icd_groupings_6month["6months ICD Other"] = icds.apply(
        lambda x: int(any(prefix not in reduce(lambda arr,x:arr+x,ICD_DICT.values(),[]) for prefix in x))
    )
    icd_groupings_6month["6months ICD Count"] = icds.apply(
        lambda x:len(x)
    )

    cleaned_df = cleaned_df.join(icd_groupings_current).join(icd_groupings_6month)
    # endregion

    # region Allergies
    alls_groupings = pd.DataFrame()
    alls = df["Allegies"].apply(
        lambda row: [
            item.strip().title() for item in row.split(";") if "no " not in item.strip().lower()
        ] if row!="Empty" else []
    )
    custom_alls = df["CustomAllegies"].apply(
        lambda row: [
            item.strip().title() for item in row.split(";") if "no " not in item.strip().lower()
        ] if row!="Empty" else []
    )
    count_alls = pd.DataFrame({
        "Allergy_Count":alls,
        "Custom_Allergy_Count":custom_alls
    })
    alls_groupings["allergies_count"] = count_alls.apply(
    lambda row:len(row["Allergy_Count"]) + len(row["Custom_Allergy_Count"]), axis = 1
    )

    cleaned_df = cleaned_df.join(alls_groupings)
    # endregion

    # region Vaccination
    vacc_groupings = pd.DataFrame()
    vaccs = df["Vaccinations"].apply(
        lambda row: [
            item.strip().lower() for item in row.split(";")
        ] if row!="Empty" else []
    )

    # for group_name, group_prefixes in VACCINATION.items():
    #     vacc_groupings[group_name] = vaccs.apply(
    #         lambda x: int(any(prefix in [grp["name"].lower() for grp in group_prefixes] for prefix in x))
    #     )
    # vacc_groupings["other_vaccinations"] = vaccs.apply(
    #     lambda x: int(any(prefix not in reduce(lambda arr,x:arr+[vacc["name"].lower() for vacc in x],VACCINATION.values(),[]) for prefix in x))
    # )
    vacc_groupings["vaccination_count"] = vaccs.apply(
        lambda x:len(x)
    )
    cleaned_df = cleaned_df.join(vacc_groupings)
    # endregion

    # region Insurance
    df["Primary Claim Type"] = df["Primary Claim Type"].fillna("No Insurance")
    df["Secondary Claim Type"] = df["Secondary Claim Type"].fillna("No Insurance")
    # endregion

    # region Average Visit Pattern
    cleaned_df["Average Visit Pattern"] = df["Average Duration PCP Visit"].fillna(SCALE_COLS["Average Visit Pattern"]["mean"])
    # endregion
    for key in DUMMY_COLS.keys():
        dummies = pd.get_dummies(
            pd.Categorical(df[key],categories=DUMMY_COLS[key].keys()),prefix=key
        )
        cleaned_df = cleaned_df.join(dummies)
    return cleaned_df

if __name__=="__main__":
    df = pd.read_csv("./data/Full_DataV3.csv")
    df = df.sort_values(by=["PatientId","EncounterDt"], ignore_index=True)
    cleaned_df = preprocess_time_data(df)
    print(cleaned_df.head())
    cleaned_df.to_csv("./data/unnorm_data_reduced_shift.csv")


