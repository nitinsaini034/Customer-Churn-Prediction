import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.joblib')
    return model

# Prepare encoders and scaler
@st.cache_resource
def initialize_encoders_and_scaler():
    encoders = {}
    categorical_features = {
        'gender': ['Female', 'Male'],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['No', 'Yes', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    for feature, classes in categorical_features.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[feature] = le

    scaler = StandardScaler()
    dummy_data = np.array([
        [0, 0, 18.25, 0],
        [1, 72, 118.75, 10000]
    ])
    scaler.fit(dummy_data)

    return encoders, scaler

FEATURE_ORDER = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                 'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen', 'tenure',
                 'MonthlyCharges', 'TotalCharges']

def preprocess_input(data, encoders, scaler):
    """Preprocess user input data"""
    processed_data = data.copy()

    for feature in encoders.keys():
        if feature in processed_data.columns:
            processed_data[feature] = encoders[feature].transform(processed_data[feature])

    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    processed_data[numerical_features] = scaler.transform(processed_data[numerical_features])

    return processed_data

# Main app
def main():
    st.title("📊 Customer Churn Prediction")
    st.markdown("Predict customer churn risk using machine learning")
    st.markdown("---")

    # Load model and encoders
    model = load_model()
    encoders, scaler = initialize_encoders_and_scaler()

    # Create tabs for main features only
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "❓ How it Works"])

    with tab1:
        st.header("Make a Prediction")

        # Create columns for input
        demo_col, services_col, account_col = st.columns(3)

        with demo_col:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender_single")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior_single")
            partner = st.selectbox("Partner", ["No", "Yes"], key="partner_single")
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="dep_single")

        with services_col:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_single")
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multi_single")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="inet_single")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="osec_single")
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="obak_single")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="dev_single")
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_single")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="stv_single")
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="smov_single")

        with account_col:
            st.subheader("Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure_single")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=0.1, key="m_charge_single")
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, step=0.1, key="t_charge_single")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract_single")
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paper_single")
            payment_method = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"], key="pay_single")

        # Prepare input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        input_data = input_data[FEATURE_ORDER]

        # Real-time prediction
        try:
            processed_input = preprocess_input(input_data, encoders, scaler)
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]

            st.markdown("---")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK")
                else:
                    st.success("✅ LOW RISK")

            with col2:
                st.metric("Risk Score", f"{max(prediction_proba)*100:.2f}%")

            with col3:
                st.metric("Churn Probability", f"{prediction_proba[1]*100:.1f}%")

            st.markdown("---")

            # Visualization: Prediction breakdown
            fig_breakdown = go.Figure(data=[
                go.Bar(x=['Staying', 'Churned'], y=[prediction_proba[0]*100, prediction_proba[1]*100],
                       marker_color=['green', 'red'])
            ])
            fig_breakdown.update_layout(title="Prediction Probability",
                                       yaxis_title="Probability (%)",
                                       xaxis_title="Outcome",
                                       showlegend=False)
            st.plotly_chart(fig_breakdown, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("❓ How it Works")

        # How to Use Section
        st.subheader("📋 How to Use This Application")
        st.markdown("""
        **Step-by-step guide to make a prediction:**

        1. **Fill in Demographics**
           - Select the customer's gender (Male/Female)
           - Indicate if they are a senior citizen
           - Select partner status and dependents

        2. **Choose Services**
           - Select which services the customer has (Phone, Internet, Security, etc.)
           - Multiple options allow you to customize the customer profile

        3. **Enter Account Information**
           - **Tenure**: How long they've been a customer (in months)
           - **Monthly Charges**: The monthly bill amount
           - **Total Charges**: Lifetime charges (monthly charge × tenure)
           - **Contract**: Type of contract (Month-to-month, One year, Two year)
           - **Billing & Payment**: Paperless billing preference and payment method

        4. **Get Prediction**
           - The model automatically predicts and displays:
             - **Risk Status**: HIGH RISK or LOW RISK
             - **Risk Score**: Overall probability percentage
             - **Churn Probability**: Specific churn prediction percentage
             - **Visualization**: Bar chart showing staying vs. churned probabilities
        """)

        st.markdown("---")

        # How the Model Works
        st.subheader("🤖 How the Model Predicts")
        st.markdown("""
        **Algorithm: Random Forest Classifier**

        The model uses a Random Forest algorithm, which works by:

        1. **Creating Multiple Decision Trees**
           - Builds 100+ decision trees, each making independent predictions
           - Each tree looks at different combinations of customer features

        2. **Analyzing Key Indicators**
           - **Tenure** (months with company): Longer tenure = Lower churn risk
           - **Contract Type**: Multi-year contracts = Much lower churn risk
           - **Internet Service**: Fiber optic users show higher churn tendency
           - **Monthly Charges**: Higher charges = Higher churn risk
           - **Support Services**: Tech support and security = Lower churn risk

        3. **Voting & Consensus**
           - All trees "vote" on whether the customer will churn
           - Final prediction is the majority vote (more reliable than single decision)
           - Probability shows confidence level of the prediction

        **Prediction Output:**
        - **Churn Probability**: 0-100% likelihood of customer leaving
        - Risk level categorization:
          - **LOW RISK** (0-50%): Customer likely to stay
          - **HIGH RISK** (50-100%): Customer likely to churn
        """)

        st.markdown("---")

        # Model Information
        st.subheader("ℹ️ Important Information")
        st.markdown("""
        **Model Performance**
        - Training Accuracy: ~78.89%
        - Total Features Used: 20
        - Training Dataset: 7,032 telecom customer records
        - Algorithm: Ensemble Learning (Random Forest)

        **Features Tracked**
        - Demographics (4): Gender, Age, Partner, Dependents
        - Services (10): Phone, Internet, Security, Backup, Protection, Support, TV, Movies
        - Account (6): Tenure, Contract, Billing, Payment, Monthly/Total Charges, Senior Citizen

        **When to Use This Tool**
        - Identify at-risk customers before they leave
        - Plan targeted retention campaigns
        - Prioritize customer service efforts
        - Understand what factors drive churn predictions

        **Limitations & Disclaimers**
        - This is a predictive model, not 100% accurate
        - Real-world decisions should incorporate:
          - Customer feedback and satisfaction scores
          - Business context and market conditions
          - Support team insights
        - Historical customer data may have biases
        - Model should be retrained periodically with new data

        **Top Factors Influencing Churn (in order of importance)**
        1. **Tenure**: Whether customer is long-term or new
        2. **Contract Type**: Long-term vs. month-to-month
        3. **Internet Service**: Type of internet connection
        4. **Monthly Charges**: Price sensitivity indicator
        5. **Services**: Number and type of add-on services
        """)

        st.markdown("---")

        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        - **Accurate Data**: Enter customer information as accurately as possible
        - **Current Status**: Use current/real customer details for meaningful predictions
        - **Combine with Insight**: Use predictions alongside customer feedback
        - **Track Changes**: Monitor how contract changes affect churn probability
        - **Action Items**: Focus on high-risk customers with:
          - Short tenure (< 6 months)
          - Month-to-month contracts
          - High monthly charges
          - Limited support services
        """)


if __name__ == "__main__":
    main()
