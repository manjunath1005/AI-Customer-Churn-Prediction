import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("churn_ann_model.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# LOAD DATASET FOR COLUMNS
# -----------------------------
@st.cache_data
def load_reference_data():

    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    df.drop(
        "customerID",
        axis=1,
        inplace=True
    )

    df["Churn"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    return df

df = load_reference_data()

# -----------------------------
# HEADER
# -----------------------------
st.title(
    "📉 AI-Powered Customer Churn Prediction Platform"
)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "ANN")

with col2:
    st.metric("Accuracy", "77.4%")

with col3:
    st.metric("Dataset", "7043 Customers")

st.markdown(
    "Predict whether a telecom customer is likely to churn."
)

# -----------------------------
# PROFILE PRESETS
# -----------------------------
customer_type = st.selectbox(
    "Select Customer Profile",
    [
        "Custom Input",
        "High Risk Customer",
        "Medium Risk Customer",
        "Low Risk Customer"
    ]
)

if customer_type == "High Risk Customer":

    tenure_default = 3
    monthly_default = 95.0
    total_default = 300.0

elif customer_type == "Medium Risk Customer":

    tenure_default = 18
    monthly_default = 70.0
    total_default = 1500.0

elif customer_type == "Low Risk Customer":

    tenure_default = 60
    monthly_default = 50.0
    total_default = 5000.0

else:

    tenure_default = 12
    monthly_default = 70.0
    total_default = 1000.0

# -----------------------------
# CUSTOMER INFO
# -----------------------------
st.header("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        tenure_default
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=monthly_default
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=total_default
    )

# -----------------------------
# SERVICES
# -----------------------------
st.header("🌐 Services")

c1, c2, c3 = st.columns(3)

with c1:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with c2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with c3:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame({

    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone_service],
    "MultipleLines":[multiple_lines],
    "InternetService":[internet_service],
    "OnlineSecurity":[online_security],
    "OnlineBackup":[online_backup],
    "DeviceProtection":[device_protection],
    "TechSupport":[tech_support],
    "StreamingTV":[streaming_tv],
    "StreamingMovies":[streaming_movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment_method],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges]

})

X = df.drop("Churn", axis=1)

X_encoded = pd.get_dummies(
    X,
    drop_first=True
)

input_encoded = pd.get_dummies(
    input_data,
    drop_first=True
)

input_encoded = input_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Churn"):

    scaled_data = scaler.transform(
        input_encoded
    )

    probability = model.predict(
        scaled_data,
        verbose=0
    )[0][0]

    st.header("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Retention Probability",
            f"{(1-probability)*100:.2f}%"
        )

    if probability > 0.7:

        st.error(
            f"🔴 HIGH RISK OF CHURN ({probability*100:.2f}%)"
        )

    elif probability > 0.4:

        st.warning(
            f"🟡 MEDIUM RISK OF CHURN ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"🟢 LOW RISK OF CHURN ({probability*100:.2f}%)"
        )

    st.progress(float(probability))

    # -----------------------------
    # RISK FACTORS
    # -----------------------------
    st.subheader("🔎 Risk Factors")

    risk_factors = []

    if tenure < 12:
        risk_factors.append("Low Tenure")

    if contract == "Month-to-month":
        risk_factors.append("Month-to-Month Contract")

    if online_security == "No":
        risk_factors.append("No Online Security")

    if tech_support == "No":
        risk_factors.append("No Technical Support")

    if monthly_charges > 70:
        risk_factors.append("High Monthly Charges")
    
    if senior == 1:
        risk_factors.append("Senior Citizen")

    if len(risk_factors) == 0:
        st.success("No major churn risk factors detected.")
    else:
        for factor in risk_factors:
            st.write("•", factor)

    # -----------------------------
    # BUSINESS RECOMMENDATION
    # -----------------------------
    st.subheader(
        "📌 Business Recommendation"
    )
    if probability > 0.7:

        st.markdown("""
        - Offer annual contract discounts
        - Provide free technical support
        - Promote online security plans
        - Contact retention team
        """)

    elif probability > 0.4:

        st.markdown("""
        - Offer loyalty rewards
        - Encourage contract upgrades
        - Provide retention offers
        """)

    else:

        st.markdown("""
        - Customer likely to stay
        - Continue normal engagement
        """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption(
    "Built with TensorFlow, Scikit-Learn and Streamlit | Telco Customer Churn Dataset"
)