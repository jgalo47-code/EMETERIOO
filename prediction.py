import datetime as dt, joblib, pandas as pd, streamlit as st
from modules.data_processing import META, get_active_df
from modules.features import FEATURES, POSITIVE

@st.cache_resource
def load_model(): return joblib.load("models/trained_model.joblib")

def predict_df(df):
    """Returns (labels, P(Not Completed)) using the saved pipeline and exact training features."""
    m = load_model(); X = df[FEATURES]
    return m.predict(X), m.predict_proba(X)[:, list(m.classes_).index(POSITIVE)]

def risk_band(p): return "High" if p >= 0.6 else "Medium" if p >= 0.4 else "Low"

def show():
    st.header("Prediction: Will the reservation be completed?")
    df = get_active_df()
    with st.form("f"):
        c1, c2 = st.columns(2)
        fac = c1.selectbox("Facility type", sorted(df.Facility_Type.dropna().unique())); usr = c1.selectbox("User type", sorted(df.User_Type.dropna().unique()))
        pur = c2.selectbox("Purpose", sorted(df.Purpose.dropna().unique())); part = c2.number_input("Participants", 0, 5000, 30)
        go = st.form_submit_button("Predict")
    if go:
        if part <= 0: st.error("Participants must be greater than zero.")
        else:
            row = pd.DataFrame([dict(Facility_Type=fac, User_Type=usr, Participants=part, Purpose=pur)])[FEATURES]
            lab, p = predict_df(row); p = float(p[0])
            st.metric("Predicted outcome (model output)", str(lab[0])); st.metric("Probability of Not Completed", f"{p:.1%}")
            st.write("Risk level:", risk_band(p))
            st.caption(f"Model {META['version']} | Predicted {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
            st.warning("Prediction, not a fact. A facility administrator must review before any action; never auto-reject a reservation.")
    st.subheader("Model metrics")
    st.dataframe(pd.DataFrame({"This app (test set)": {k: round(META[k], 3) for k in ["accuracy", "precision", "recall", "f1"]},
                               "Midterm report": META["midterm_reported"]}))
    st.write("**Most influential predictors:**", ", ".join(f"{n} ({v})" for n, v in META["top_predictors"][:3]))
    st.caption(f"Data: {META['data_source']}; train n={META['n_train']}, test n={META['n_test']}. Limitations: low accuracy, small dataset, "
               "weighted averages hide per-class errors, tree probabilities are crude, and patterns may shift over time.")
