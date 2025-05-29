import streamlit as st
import requests
import pandas as pd
import logging

# Disable Streamlit warnings related to ScriptRunContext
for name, logger in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        logger.disabled = True

# Set app title
st.title("Lymphedema Risk Prediction Application")


def display_prediction_results(result):
    """Display the prediction results with formatting"""
    st.success("Prediction completed successfully!")

    risk_color = {
        "Low": "green",
        "Moderate": "yellow",
        "High": "orange",
        "Very High": "red",
    }

    # Display results in a frame
    st.markdown(
        f"""
    <div style='padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>
        <h3>Prediction Results:</h3>
        <p><b>Probability of Lymphedema:</b> {result['probability']:.2%}</p>
        <p><b>Risk Category:</b> <span style='color: {risk_color[result['risk_category']]}; font-weight: bold;'>{result['risk_category']}</span></p>
        <p><b>Prediction:</b> {'High Risk' if result['prediction'] == 1 else 'Low Risk'}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Recommendations based on risk category
    st.subheader("Recommendations:")
    if result["risk_category"] in ["High", "Very High"]:
        st.markdown(
            """
        * Regular follow-up with oncologist
        * Practice specific lymphatic exercises
        * Wear compression garments when needed
        * Monitor any swelling or changes in the affected area
        """
        )
    else:
        st.markdown(
            """
        * Self-monitoring
        * Routine follow-up with physician
        * Maintain a healthy and active lifestyle
        """
        )


# Create tabs for different input methods
tab1, tab2 = st.tabs(["Form Input", "File Upload"])

with tab1:
    st.write("Please enter patient information to predict the risk of lymphedema")

    # Create form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=50)
            weight = st.number_input(
                "Weight (kg)", min_value=40.0, max_value=200.0, value=70.0, step=0.1
            )
            height = st.number_input(
                "Height (m)", min_value=1.0, max_value=2.5, value=1.65, step=0.01
            )

            # Calculate BMI automatically
            bmi = weight / (height**2)
            st.write(f"Calculated BMI: {bmi:.2f}")

            laterality = st.selectbox("Laterality", ["Right", "Left", "Bilateral"])
            lymph_node = st.selectbox(
                "Lymph Node",
                [
                    "SLNB",
                    "AxillaryDissection(ALND)",
                    "AxillaryLymphNodeSampling",
                    "SLNB&Non-SLNB",
                ],
            )
            menopausal_status = st.selectbox(
                "Menopausal Status", ["Premenopausal", "Postmenopausal"]
            )

            # Previous medical conditions
            pastDM = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
            pastHypertension = st.selectbox("Hypertension", ["Yes", "No"])
            pastCardiac = st.selectbox("Cardiac Disease", ["Yes", "No"])
            pastLiver = st.selectbox("Liver Disease", ["Yes", "No"])
            pastRenalproblems = st.selectbox("Renal Problems", ["Yes", "No"])
            pastScrewsandplatel = st.selectbox(
                "Previous Screws and Plates", ["Yes", "No"]
            )

        with col2:
            # Cancer stage
            t_stage = st.selectbox("T Stage", ["T0", "Tis", "T1", "T2", "T3", "T4"])
            n_stage = st.selectbox("N Stage", ["N0", "N1", "N2", "N3"])
            m_stage = st.selectbox("M Stage", ["M0", "M1"])

            st.write("Specimen Information:")
            specimen = st.selectbox("Specimen", ["Yes", "No"])
            specimen_type = st.selectbox("Specimen Type", ["Conservative", "MRM"])
            peritumoural_lvi = st.selectbox(
                "Peritumoural Lymphovascular Invasion",
                ["Absent", "suspicious", "Present"],
            )

            # Treatments
            chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
            radiotherapy = st.selectbox("Radiotherapy", ["Yes", "No"])
            hormonal = st.selectbox("Hormonal Therapy", ["Yes", "No"])

            # Symptoms
            pain = st.selectbox("Pain", ["Present", "Absent"])
            tenderness = st.selectbox("Tenderness", ["Present", "Absent"])
            stiffness = st.selectbox("Stiffness", ["Present", "Absent"])
            weakness = st.selectbox("Weakness", ["Present", "Absent"])
            referral_pain = st.selectbox("Referral Pain", ["Present", "Absent"])
            swelling = st.selectbox("Swelling", ["Present", "Absent"])
            lymph_node_symptom = st.selectbox(
                "Lymph Node Symptoms (count)",
                options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                format_func=lambda x: str(x),
            )

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Build data for submission
        input_data = {
            "Age": age,
            "Weight": weight,
            "Height.m.": height,
            "BMI": bmi,
            "Laterality": laterality,
            "Lymph_node": lymph_node,
            "Lymph_node.1": lymph_node_symptom,
            "Menopausal_status": menopausal_status,
            "pastDM": pastDM,
            "pastHypertension": pastHypertension,
            "pastCardiac": pastCardiac,
            "pastLiver": pastLiver,
            "pastRenalproblems": pastRenalproblems,
            "pastScrewsandplatel": pastScrewsandplatel,
            "T": t_stage,
            "N": n_stage,
            "M": m_stage,
            "Specimen": specimen,
            "Specimen_type": specimen_type,
            "Peritumoural.lymphovascular.invasion": peritumoural_lvi,
            "chemotherapy": chemotherapy,
            "Radiotherapy": radiotherapy,
            "Hormanal": hormonal,
            "Pain": pain,
            "Tenderness": tenderness,
            "Stiffness": stiffness,
            "Weakness": weakness,
            "Referralpain": referral_pain,
            "Swelling": swelling,
        }

        try:
            # Send request with raw data to API (API will handle preprocessing)
            response = requests.post("http://localhost:5000/predict", json=input_data)
            result = response.json()

            if result["success"]:
                display_prediction_results(result)
            else:
                st.error(f"An error occurred: {result['error']}")
        except Exception as e:
            st.error(f"An error occurred during API connection: {str(e)}")
            st.info("Make sure Flask API server is running on port 5000")

with tab2:
    st.write("Upload a CSV or Excel file with patient data")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Allow user to select a row to predict
            if not df.empty:
                row_index = st.number_input(
                    "Select row to predict (0-based index)",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                )

                if st.button("Predict Selected Row"):
                    # Extract the selected row as a dictionary
                    row_data = df.iloc[row_index].to_dict()

                    try:
                        # Send request to API
                        response = requests.post(
                            "http://localhost:5000/predict", json=row_data
                        )
                        result = response.json()

                        if result["success"]:
                            display_prediction_results(result)
                        else:
                            st.error(f"An error occurred: {result['error']}")
                    except Exception as e:
                        st.error(f"An error occurred during API connection: {str(e)}")
                        st.info("Make sure Flask API server is running on port 5000")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
