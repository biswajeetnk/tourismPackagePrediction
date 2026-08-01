
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

#streamlit page configuration :
st.set_page_config(
    page_title="Tourism Package Prediction",
    layout="wide"
)

st.title("Tourism Package Prediction")

st.markdown(
"""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** based on demographic details, travel preferences, and customer interaction history.

Please provide the customer information below and click **Predict**.
"""
)

st.divider()

# Customer Input Form
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    customer_id = st.number_input(
        "Customer ID",
        min_value=1,
        value=200001
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=1,
        value=15
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=3
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

with col2:

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced",
            "Unmarried"
        ]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        1,
        5,
        3
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        value=25000
    )

predict_button = st.button("Predict")

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({

        "CustomerID": [customer_id],
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]

    })

    # Keep original values for display
    display_df = input_data.copy()

    # Encode categorical columns
    categorical_columns = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation"
    ]

    for column in categorical_columns:
        try:
            input_data[column] = label_encoders[column].transform(input_data[column])
        except ValueError:
            st.error(f"Invalid value entered for {column}")
            st.stop()

    # Arrange columns in training order
    input_data = input_data[feature_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.error("The customer is unlikely to purchase the Wellness Tourism Package.")

    st.metric(
        label="Purchase Probability",
        value=f"{probability:.2%}"
    )

    st.subheader("Input Data")
    st.dataframe(display_df)
