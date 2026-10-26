"""Rebuilds the midterm notebook's Decision Tree pipeline (same settings). Run once, commit models/."""
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from modules.features import *

df = pd.read_csv("data/facility_reservations.csv").drop_duplicates().drop_duplicates(subset=["Reservation_ID"])
df = add_derived(df).dropna(subset=[TARGET])
X, y = df[FEATURES], df[TARGET]
pre = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM),
                         ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                   ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
model = Pipeline([("preprocessor", pre), ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model.fit(Xtr, ytr); p = model.predict(Xte)
w = dict(average="weighted", zero_division=0)
names = model.named_steps["preprocessor"].get_feature_names_out(); imp = model.named_steps["classifier"].feature_importances_
top = [[str(n).split("__")[-1], round(float(v), 3)] for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:5]]
m = dict(version="v1.0-decision-tree", data_source="midterm facility_reservations.csv", accuracy=accuracy_score(yte, p),
         precision=precision_score(yte, p, **w), recall=recall_score(yte, p, **w), f1=f1_score(yte, p, **w),
         n_train=len(Xtr), n_test=len(Xte), top_predictors=top, midterm_reported=dict(accuracy=.412, precision=.484, recall=.412, f1=.430))
joblib.dump(model, "models/trained_model.joblib"); json.dump(m, open("models/metrics.json", "w"), indent=2); print(m)
