"""Shared data loading, cleaning and feature logic.

Both train.py and predict.py import from here so that training and serving
always prepare data in exactly the same way (the same cleaning rules as notebook.ipynb).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# Discharge dispositions for patients who died or went to hospice (they cannot be readmitted)
DEATH_OR_HOSPICE = [11, 13, 14, 19, 20, 21]

# Columns that hold codes, not quantities, so they are treated as categories
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

NUM_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
# Heavily right-skewed counts; log-transformed for the linear model
SKEWED = ["number_outpatient", "number_emergency", "number_inpatient"]

DRUGS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Columns removed before modelling (IDs, target columns and the raw ICD-9 codes,
# which are replaced by their grouped versions)
NON_FEATURES = [
    "encounter_id", "patient_nbr", "readmitted", "readmit_30",
    "diag_1", "diag_2", "diag_3",
]

# Fields a caller must supply to the prediction service
REQUIRED_FIELDS = [
    "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "diag_1", "change", "diabetesMed",
]
# Fields the caller may leave out; these defaults match how the training data encodes "unknown"
OPTIONAL_DEFAULTS = {
    "race": "Missing",
    "payer_code": "Missing",
    "medical_specialty": "Missing",
    "max_glu_serum": "None",   # test not performed
    "A1Cresult": "None",       # test not performed
}


def icd9_group(code):
    """Map an ICD-9 diagnosis code to a broad clinical group (Strack et al., 2014)."""
    if code is None or pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 250 <= v < 251:
        return "Diabetes"
    if (390 <= v <= 459) or v == 785:
        return "Circulatory"
    if (460 <= v <= 519) or v == 786:
        return "Respiratory"
    if (520 <= v <= 579) or v == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if (580 <= v <= 629) or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def load_raw(csv_path):
    """Read diabetic_data.csv, downloading and extracting it from UCI if it is not there yet."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{csv_path} not found, downloading from UCI...")
        with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(csv_path.parent)
    # "?" marks missing values. keep_default_na=False stops pandas turning the
    # string "None" (glucose / A1C test not performed) into a missing value.
    return pd.read_csv(csv_path, na_values=["?"], keep_default_na=False)


def clean(df_raw):
    """Apply the cleaning decisions from the notebook and add the binary target."""
    df = df_raw[~df_raw["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)]
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    df = df.drop(columns=["weight"])
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    for col in ID_COLS:
        df[col] = df[col].astype(str)
    for col in ["race", "payer_code", "medical_specialty"]:
        df[col] = df[col].fillna("Missing")
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].map(icd9_group)

    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    return df.reset_index(drop=True)


def find_rare_drugs(df, threshold=0.99):
    """Drug columns where more than `threshold` of patients have 'No' (almost no information)."""
    return [c for c in DRUGS if (df[c] == "No").mean() > threshold]


def build_features(df, rare_drugs):
    """Return the model input table: everything except IDs, targets, raw codes and rare drugs."""
    return df.drop(columns=NON_FEATURES + list(rare_drugs))


def split_feature_types(columns):
    num = [c for c in columns if c in NUM_FEATURES]
    cat = [c for c in columns if c not in NUM_FEATURES]
    return cat, num


def _normalise_category(value):
    """Make a JSON value look like the strings used in training (e.g. 1.0 -> '1')."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def record_to_frame(record, feature_columns):
    """Turn one patient record (a dict from JSON) into a one-row DataFrame for the model.

    Raises ValueError with a readable message if the record is unusable.
    """
    if not isinstance(record, dict):
        raise ValueError("Request body must be a JSON object describing one patient.")

    rec = dict(record)
    missing = [f for f in REQUIRED_FIELDS if rec.get(f) is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    for key, default in OPTIONAL_DEFAULTS.items():
        if rec.get(key) is None or rec.get(key) == "?":
            rec[key] = default
    for drug in DRUGS:
        if rec.get(drug) is None:
            rec[drug] = "No"
    for i in (1, 2, 3):
        rec[f"diag_{i}_group"] = icd9_group(rec.get(f"diag_{i}"))

    row = {}
    for col in feature_columns:
        value = rec[col]
        if col in NUM_FEATURES:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Field '{col}' must be a number, got {value!r}.")
            if not np.isfinite(number):
                raise ValueError(f"Field '{col}' must be a finite number.")
            row[col] = number
        else:
            row[col] = _normalise_category(value)
    return pd.DataFrame([row], columns=feature_columns)
