import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------------------------
# 1️⃣  MUST be first Streamlit command
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="🏥 Diabetes 30-Day Readmission Predictor",
    layout="centered"
)

# -------------------------------------------------------------------------
# 2️⃣  Load the trained model (pipeline)
# -------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/readmit_model.pkl")
    return model

model = load_model()

st.title("🏥 Diabetes Readmission Risk App")
st.write(
    "Predict whether a diabetic patient is **likely to be readmitted within 30 days** "
    "after discharge.\n\n"
    "Model trained on >100k hospital encounters."
)
st.markdown("---")

# -------------------------------------------------------------------------
# 3️⃣  Collect user inputs
# -------------------------------------------------------------------------
st.subheader("🔢 Enter Patient Details")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=6
        )
        gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])
        race = st.selectbox(
            "Race",
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
        )
        time_in_hospital = st.number_input(
            "Time in hospital (days)", min_value=1, max_value=14, value=4)
        number_diagnoses = st.number_input(
            "Number of diagnoses", min_value=1, max_value=16, value=6)

    with col2:
        admission_type_id = st.selectbox(
            "Admission type ID",
            [1, 2, 3, 4, 5, 6, 7, 8],
            help="1=Emergency, 2=Urgent, 3=Elective, etc."
        )
        discharge_disposition_id = st.selectbox(
            "Discharge disposition ID",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 28]
        )
        admission_source_id = st.selectbox(
            "Admission source ID",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 21, 22, 23, 24, 25]
        )
        num_lab_procedures = st.number_input(
            "Number of lab procedures", min_value=0, max_value=150, value=44)
        num_medications = st.number_input(
            "Number of medications", min_value=0, max_value=100, value=16)

    st.markdown("### 💊 Diabetes Treatment")

    col3, col4, col5 = st.columns(3)
    with col3:
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        change_ui = st.selectbox(
    "Any medication change?",
    ["No", "Yes"],   # what the user sees
)
    # map to model format
    change = "Ch" if change_ui == "Yes" else "No"

    with col4:
        diabetesMed = st.selectbox("On diabetes medication?", ["No", "Yes"])
        num_procedures = st.number_input(
            "Number of procedures", min_value=0, max_value=6, value=1)
    with col5:
        number_outpatient = st.number_input(
            "No. of outpatient visits", min_value=0, max_value=20, value=0)
        number_emergency = st.number_input(
            "No. of emergency visits", min_value=0, max_value=20, value=0)
        number_inpatient = st.number_input(
            "No. of inpatient visits", min_value=0, max_value=20, value=0)

    submitted = st.form_submit_button("🔮 Predict readmission risk")

# -------------------------------------------------------------------------
# 4️⃣  When user clicks PREDICT
# -------------------------------------------------------------------------
if submitted:
    patient_dict = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed,
        # default diagnosis placeholders
        "diag_1": "250",
        "diag_2": "250",
        "diag_3": "250"
    }

    input_df = pd.DataFrame([patient_dict])

    # ---------------------------------------------------------------------
    # ✅ Add missing columns expected by the model
    # ---------------------------------------------------------------------
    expected_cols = model.named_steps['preprocess'].feature_names_in_

    for col in expected_cols:
        if col not in input_df.columns:
            if any(drug in col for drug in [
                'metformin', 'glyburide', 'glipizide', 'repaglinide',
                'rosiglitazone', 'pioglitazone', 'nateglinide', 'acarbose',
                'miglitol', 'troglitazone', 'chlorpropamide', 'tolbutamide',
                'glimepiride', 'tolazamide', 'citoglipton', 'examide'
            ]):
                input_df[col] = 'No'
            else:
                input_df[col] = 0

    # ---------------------------------------------------------------------
    # 5️⃣  Predict
    # ---------------------------------------------------------------------
    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    # ---------------------------------------------------------------------
    # 6️⃣  Display results
    # ---------------------------------------------------------------------
    st.markdown("## 🧪 Prediction Result")
    st.write(f"**Predicted probability of 30-day readmission:** `{prob:.2%}`")

    if prob >= 0.50:
        risk = "🔴 HIGH RISK"
    elif prob >= 0.25:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    st.write(f"**Risk level:** {risk}")
    st.markdown("---")
    st.caption("Note: Model outputs probabilities based on historical data; always apply clinical judgment.")
