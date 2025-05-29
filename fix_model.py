import pandas as pd


def preprocess_input(data):
    """
    Function that processes input data to match the model's expected 19 features
    """
    # Copy of data to avoid changes to the original data
    df = data.copy()

    # Handle missing values in important fields
    # For numeric features, fill with appropriate values
    if "BMI" in df.columns and df["BMI"].isna().any():
        # Calculate BMI if possible
        if "Weight" in df.columns and "Height.m." in df.columns:
            for idx, row in df.iterrows():
                if (
                    pd.isna(row["BMI"])
                    and not pd.isna(row["Weight"])
                    and not pd.isna(row["Height.m."])
                    and row["Height.m."] > 0
                ):
                    df.at[idx, "BMI"] = row["Weight"] / (row["Height.m."] ** 2)
        # If still NA, replace with median or default value
        if df["BMI"].isna().any():
            median_bmi = df["BMI"].median()
            if pd.isna(median_bmi):
                df["BMI"] = df["BMI"].fillna(25.0)  # Default value
            else:
                df["BMI"] = df["BMI"].fillna(median_bmi)

    # Replace NaN in T, N with reasonable defaults
    if "T" in df.columns:
        df["T"] = df["T"].fillna("T1")  # Most common or reasonable default

    if "N" in df.columns:
        df["N"] = df["N"].fillna("N0")  # Most common or reasonable default

    if "Lymph_node" in df.columns:
        df["Lymph_node"] = df["Lymph_node"].fillna(
            "SLNB"
        )  # Most common or reasonable default

    # For Lymph_node.1, ensure it's numeric
    if "Lymph_node.1" in df.columns:
        # Convert to float first to handle any text values
        df["Lymph_node.1"] = pd.to_numeric(df["Lymph_node.1"], errors="coerce")
        df["Lymph_node.1"] = df["Lymph_node.1"].fillna(0)

    # Convert categorical variables to numeric values
    categorical_features = [
        "Laterality",
        "Lymph_node",
        "Menopausal_status",
        "pastDM",
        "pastHypertension",
        "pastCardiac",
        "pastLiver",
        "pastRenalproblems",
        "pastScrewsandplatel",
        "T",
        "N",
        "M",
        "Specimen_type",
        "Peritumoural.lymphovascular.invasion",
        "chemotherapy",
        "Radiotherapy",
        "Hormanal",
        "Pain",
        "Tenderness",
        "Stiffness",
        "Weakness",
        "Referralpain",
        "Swelling",
        "Lymph_node.1",
    ]

    # Binary value mappings
    binary_mappings = {
        "yes": 1,
        "no": 0,
        "Yes": 1,
        "No": 0,
        "TRUE": 1,
        "FALSE": 0,
        "Present": 1,
        "Absent": 0,
        "Right": 0,
        "Left": 0,
        "Bilateral": 1,
    }

    # Special mappings for specific variables
    special_mappings = {
        "Menopausal_status": {"Premenopausal": 0, "Postmenopausal": 1},
        "Specimen_type": {"Conservative": 1, "MRM": 2},
        "T": {"T0": 0, "Tis": 1, "T1": 2, "T2": 3, "T3": 4, "T4": 5},
        "N": {"N0": 0, "N1": 1, "N2": 2, "N3": 3},
        "M": {"M0": 0, "M1": 1},
        "Peritumoural.lymphovascular.invasion": {
            "Absent": 0,
            "suspicious": 1,
            "Present": 2,
        },
        "Lymph_node": {
            "SLNB": 0,
            "AxillaryDissection(ALND)": 1,
            "AxillaryLymphNodeSampling": 2,
            "SLNB&Non-SLNB": 3,
        },
    }

    # Apply transformations to the data
    for col in df.columns:
        if col in special_mappings and df[col].iloc[0] in special_mappings[col]:
            df[col] = df[col].map(special_mappings[col])
            # If there are still any NaN values after mapping, fill with 0
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        elif col in categorical_features:
            # Apply binary transformations
            for old_val, new_val in binary_mappings.items():
                df[col] = df[col].replace(old_val, new_val)
            # If there are still any NaN values after replacement, fill with 0
            if df[col].isna().any():
                df[col] = df[col].fillna(0)

    # Normalize numeric variables
    numeric_features = ["Age", "BMI"]
    for col in numeric_features:
        if col in df.columns:
            # Apply robust scaling
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                # If std is 0 (or very close), just center the data
                df[col] = df[col] - mean

    # Ensure Hormanal exists and correct its name if different
    if "Hormanal" not in df.columns and "hormonal" in df.columns:
        df["Hormanal"] = df["hormonal"]
    # If still missing, create it with default value 0
    if "Hormanal" not in df.columns:
        df["Hormanal"] = 0

    # Create derived variables
    if "Age" in df.columns:
        df["Age_Squared"] = df["Age"] ** 2

    if "BMI" in df.columns and "Age" in df.columns:
        df["BMI_Age_Interaction"] = df["BMI"] * df["Age"]

    # Create required interaction variables
    if "Hormanal" in df.columns and "N" in df.columns:
        df["Hormanal_N_Interaction"] = df["Hormanal"] * df["N"]

    if "Hormanal" in df.columns and "T" in df.columns:
        df["Hormanal_T_Interaction"] = df["Hormanal"] * df["T"]

    if "Radiotherapy" in df.columns and "T" in df.columns:
        df["Radiotherapy_T_Interaction"] = df["Radiotherapy"] * df["T"]

    if (
        "Hormanal" in df.columns
        and "Peritumoural.lymphovascular.invasion" in df.columns
    ):
        df["Hormanal_Peritumoural.lymphovascular.invasion_Interaction"] = (
            df["Hormanal"] * df["Peritumoural.lymphovascular.invasion"]
        )

    # List of required features in the correct order for the model
    required_features = [
        "Swelling",
        "M",
        "Specimen_type",
        "Radiotherapy",
        "pastCardiac",
        "Hormanal_N_Interaction",
        "Laterality",
        "Lymph_node",
        "chemotherapy",
        "N",
        "Age_Squared",
        "Radiotherapy_T_Interaction",
        "pastHypertension",
        "Hormanal_Peritumoural.lymphovascular.invasion_Interaction",
        "BMI_Age_Interaction",
        "Hormanal_T_Interaction",
        "Menopausal_status",
        "T",
        "Hormanal",
    ]

    # Ensure all required features exist
    for feature in required_features:
        if feature not in df.columns:
            # Add missing feature with zero value
            df[feature] = 0

    # Ensure only required features are used and in the correct order
    final_df = df[required_features]

    # Replace any remaining NaN values with 0
    final_df = final_df.fillna(0)

    # Verify feature count
    assert (
        final_df.shape[1] == 19
    ), f"Feature count is {final_df.shape[1]} instead of the expected 19"

    return final_df


# Test the function
if __name__ == "__main__":
    # Create test data similar to Streamlit app input
    test_input = {
        "Age": 45,
        "BMI": 25,
        "Laterality": "Left",
        "Lymph_node": "SLNB",
        "Menopausal_status": "Premenopausal",
        "pastDM": "No",
        "pastHypertension": "No",
        "pastCardiac": "No",
        "pastLiver": "No",
        "pastRenalproblems": "No",
        "pastScrewsandplatel": "No",
        "T": "T2",
        "N": "N0",
        "M": "M0",
        "Specimen_type": "Conservative",
        "Peritumoural.lymphovascular.invasion": "Absent",
        "chemotherapy": "No",
        "Radiotherapy": "No",
        "hormonal": "No",
        "Pain": "Absent",
        "Tenderness": "Absent",
        "Stiffness": "Absent",
        "Weakness": "Absent",
        "Referralpain": "Absent",
        "Swelling": "Absent",
        "Lymph_node.1": "Absent",
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([test_input])

    # Process the data
    processed_data = preprocess_input(df_input)

    # Display results
    print("\nProcessed data:")
    print(processed_data)
    print(f"Feature count: {processed_data.shape[1]}")
    print(f"Feature names: {processed_data.columns.tolist()}")

    # Now test with missing values
    print("\nTesting with missing values:")
    test_input_with_missing = {
        "Age": 69.0,
        "Menopausal_status": "Postmenopausal",
        "Weight": None,
        "Height.m.": 0.00,
        "BMI": None,
        "pastDM": "no",
        "pastHypertension": "yes",
        "pastCardiac": "yes",
        "pastLiver": "no",
        "pastRenalproblems": "no",
        "pastScrewsandplatel": "no",
        "Laterality": "Right",
        "T": "T0",
        "N": "N0",
        "M": "M0",
        "Specimen": "Yes",
        "Specimen_type": "MRM",
        "Lymph_node.1": 0.0,
        "Lymph_node": None,
        "Peritumoural.lymphovascular.invasion": "Absent",
        "chemotherapy": "No",
        "Radiotherapy": "No",
        "Hormanal": "No",
        "Pain": "yes",
        "Tenderness": "no",
        "Stiffness": "no",
        "Weakness": "no",
        "Referralpain": "no",
        "Swelling": "no",
    }

    df_missing = pd.DataFrame([test_input_with_missing])
    processed_data_missing = preprocess_input(df_missing)

    print("\nProcessed data with missing values:")
    print(processed_data_missing)
    print(f"Feature count: {processed_data_missing.shape[1]}")
    print(f"Feature names: {processed_data_missing.columns.tolist()}")
