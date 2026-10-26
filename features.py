"""Midterm notebook logic (Untitled7.ipynb): cleaning, derived variables, predictor list."""
import re, pandas as pd

NUM = ["Participants"]
CAT = ["Facility_Type", "User_Type", "Purpose"]
FEATURES = ["Facility_Type", "User_Type", "Participants", "Purpose"]  # notebook's selected predictors, known at submission
TARGET = "Reservation_Outcome"   # Completed / Not Completed
POSITIVE = "Not Completed"       # cancellation / no-show

def _tidy(x):
    return re.sub(r"\s+", " ", x.strip()).casefold().title() if isinstance(x, str) else x

def add_derived(df):
    df = df.copy()
    for c in df.columns:                      # standardize category text, as in the notebook
        if not pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].map(_tidy)
    if "Participants" in df: df["Participants"] = pd.to_numeric(df["Participants"], errors="coerce").where(lambda s: s > 0)
    lead = "Lead_Time_Days" if "Lead_Time_Days" in df else "Days_Before_Event" if "Days_Before_Event" in df else None
    if lead:   # dashboard only; NOT a model input
        d = pd.to_numeric(df[lead], errors="coerce")
        df["Lead_Time_Category"] = pd.cut(d, [-1, 2, 7, 14, float("inf")], labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"])
    if TARGET not in df and "Usage_Status" in df:
        m = {"used": "Completed", "completed": "Completed", "cancelled": "Not Completed", "canceled": "Not Completed",
             "no-show": "Not Completed", "no show": "Not Completed", "noshow": "Not Completed"}
        df[TARGET] = df["Usage_Status"].astype(str).str.casefold().map(m)
    return df
