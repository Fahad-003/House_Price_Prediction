import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="🏠 Ames House Price Prediction",
    layout="wide"
)

st.markdown("""
# 🏠 Ames House Price Prediction

Estimate residential property prices using an XGBoost Regression model
trained on the Ames Housing Dataset.

""")

st.caption("📂 Dataset: ** Ames Housing Dataset ( 2930 Residential Properties )**")

st.image(
    "assets/house_banner.jpg",
    use_container_width=True
)

st.info(
    "💡 Enter the property details from the left sidebar and click **Predict Price** to estimate the market value."
)




# -----------------------------------------------------
# Load Pipeline
# -----------------------------------------------------

pipeline = joblib.load("model/house_price_prediction_pipeline.pkl")

# -----------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------

st.sidebar.header("Property Details")

overall_qual = st.sidebar.slider(
    "Overall Quality",
    1,
    10,
    6
)

overall_cond = st.sidebar.slider(
    "Overall Condition",
    1,
    10,
    5
)

gr_liv_area = st.sidebar.number_input(
    "Ground Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

lot_area = st.sidebar.number_input(
    "Lot Area",
    min_value=1000,
    max_value=100000,
    value=9356
)

total_bsmt_sf = st.sidebar.number_input(
    "Total Basement Area",
    min_value=0,
    max_value=5000,
    value=988
)

first_flr = st.sidebar.number_input(
    "1st Floor Area",
    min_value=300,
    max_value=5000,
    value=1082
)

second_flr = st.sidebar.number_input(
    "2nd Floor Area",
    min_value=0,
    max_value=3000,
    value=0
)

garage_cars = st.sidebar.slider(
    "Garage Capacity",
    0,
    5,
    2
)

garage_area = st.sidebar.number_input(
    "Garage Area",
    min_value=0,
    max_value=1500,
    value=476
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1872,
    max_value=2010,
    value=1972
)

year_remodel = st.sidebar.number_input(
    "Year Remodeled",
    min_value=1950,
    max_value=2010,
    value=1992
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    0,
    5,
    2
)

half_bath = st.sidebar.slider(
    "Half Bathrooms",
    0,
    3,
    0
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

rooms = st.sidebar.slider(
    "Rooms Above Ground",
    2,
    15,
    6
)

fireplaces = st.sidebar.slider(
    "Fireplaces",
    0,
    4,
    1
)

kitchen_qual = st.sidebar.selectbox(
    "Kitchen Quality",
    [
        "Ex",
        "Gd",
        "TA",
        "Fa"
    ]
)

neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    [
        "NAmes",
        "CollgCr",
        "OldTown",
        "Edwards",
        "Somerst",
        "NridgHt",
        "Gilbert",
        "Sawyer",
        "SawyerW",
        "NWAmes",
        "BrkSide",
        "Crawfor",
        "Mitchel",
        "NoRidge",
        "Timber",
        "StoneBr",
        "ClearCr",
        "IDOTRR",
        "SWISU",
        "MeadowV",
        "Blmngtn",
        "BrDale",
        "Veenker",
        "NPkVill",
        "Blueste"
    ]
)

predict = st.sidebar.button(
    "💰 Predict Price",
    use_container_width=True
)

# -----------------------------------------------------
# Default Feature Values
# -----------------------------------------------------

data = {

    'MS SubClass': 50,
    'MS Zoning': 'RL',
    'Lot Frontage': 68,
    'Lot Area': 9356,
    'Street': 'Pave',
    'Alley': 'None',
    'Lot Shape': 'Reg',
    'Land Contour': 'Lvl',
    'Utilities': 'AllPub',
    'Lot Config': 'Inside',
    'Land Slope': 'Gtl',
    'Neighborhood': 'NAmes',
    'Condition 1': 'Norm',
    'Condition 2': 'Norm',
    'Bldg Type': '1Fam',
    'House Style': '1Story',
    'Overall Qual': 6,
    'Overall Cond': 5,
    'Year Built': 1972,
    'Year Remod/Add': 1992,
    'Roof Style': 'Gable',
    'Roof Matl': 'CompShg',
    'Exterior 1st': 'VinylSd',
    'Exterior 2nd': 'VinylSd',
    'Mas Vnr Type': 'BrkFace',
    'Mas Vnr Area': 0,
    'Exter Qual': 'TA',
    'Exter Cond': 'TA',
    'Foundation': 'CBlock',
    'Bsmt Qual': 'TA',
    'Bsmt Cond': 'TA',
    'Bsmt Exposure': 'No',
    'BsmtFin Type 1': 'GLQ',
    'BsmtFin SF 1': 375,
    'BsmtFin Type 2': 'Unf',
    'BsmtFin SF 2': 0,
    'Bsmt Unf SF': 462,
    'Total Bsmt SF': 988,
    'Heating': 'GasA',
    'Heating QC': 'Ex',
    'Central Air': 'Y',
    'Electrical': 'SBrkr',
    '1st Flr SF': 1082,
    '2nd Flr SF': 0,
    'Low Qual Fin SF': 0,
    'Gr Liv Area': 1436,
    'Bsmt Full Bath': 0,
    'Bsmt Half Bath': 0,
    'Full Bath': 2,
    'Half Bath': 0,
    'Bedroom AbvGr': 3,
    'Kitchen AbvGr': 1,
    'Kitchen Qual': 'TA',
    'TotRms AbvGrd': 6,
    'Functional': 'Typ',
    'Fireplaces': 1,
    'Fireplace Qu': 'None',
    'Garage Type': 'Attchd',
    'Garage Yr Blt': 1978,
    'Garage Finish': 'Unf',
    'Garage Cars': 2,
    'Garage Area': 476,
    'Garage Qual': 'TA',
    'Garage Cond': 'TA',
    'Paved Drive': 'Y',
    'Wood Deck SF': 0,
    'Open Porch SF': 26,
    'Enclosed Porch': 0,
    '3Ssn Porch': 0,
    'Screen Porch': 0,
    'Pool Area': 0,
    'Pool QC': 'None',
    'Fence': 'None',
    'Misc Feature': 'None',
    'Misc Val': 0,
    'Mo Sold': 6,
    'Yr Sold': 2008,
    'Sale Type': 'WD ',
    'Sale Condition': 'Normal'
}

# -----------------------------------------------------
# Replace Defaults with User Inputs
# -----------------------------------------------------

data["Overall Qual"] = overall_qual
data["Overall Cond"] = overall_cond
data["Gr Liv Area"] = gr_liv_area
data["Lot Area"] = lot_area
data["Total Bsmt SF"] = total_bsmt_sf
data["1st Flr SF"] = first_flr
data["2nd Flr SF"] = second_flr
data["Garage Cars"] = garage_cars
data["Garage Area"] = garage_area
data["Year Built"] = year_built
data["Year Remod/Add"] = year_remodel
data["Full Bath"] = full_bath
data["Half Bath"] = half_bath
data["Bedroom AbvGr"] = bedrooms
data["TotRms AbvGrd"] = rooms
data["Fireplaces"] = fireplaces
data["Kitchen Qual"] = kitchen_qual
data["Neighborhood"] = neighborhood


# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["HouseAge"] = data["Yr Sold"] - data["Year Built"]

data["RemodelAge"] = (
    data["Yr Sold"] - data["Year Remod/Add"]
)

data["TotalBathrooms"] = (
    data["Full Bath"]
    + 0.5 * data["Half Bath"]
    + data["Bsmt Full Bath"]
    + 0.5 * data["Bsmt Half Bath"]
)

data["TotalPorchSF"] = (
    data["Open Porch SF"]
    + data["Enclosed Porch"]
    + data["3Ssn Porch"]
    + data["Screen Porch"]
)

data["TotalLivingArea"] = (
    data["Gr Liv Area"]
    + data["Total Bsmt SF"]
)

data["TotalHouseArea"] = (
    data["Total Bsmt SF"]
    + data["1st Flr SF"]
    + data["2nd Flr SF"]
)

data["TotalOutdoorArea"] = (
    data["Wood Deck SF"]
    + data["Open Porch SF"]
    + data["Enclosed Porch"]
    + data["3Ssn Porch"]
    + data["Screen Porch"]
)

data["TotalRooms"] = (
    data["TotRms AbvGrd"]
    + data["Bedroom AbvGr"]
)

data["HasGarage"] = int(
    data["Garage Area"] > 0
)

data["HasBasement"] = int(
    data["Total Bsmt SF"] > 0
)

data["HasFireplace"] = int(
    data["Fireplaces"] > 0
)

data["HasPool"] = int(
    data["Pool Area"] > 0
)

data["HasSecondFloor"] = int(
    data["2nd Flr SF"] > 0
)

data["GarageAge"] = (
    data["Yr Sold"]
    - data["Garage Yr Blt"]
)

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_df = pd.DataFrame([data])



# =====================================================
# PREDICTION
# =====================================================

if predict :

    try:

        # -----------------------------
        # Make Prediction
        # -----------------------------

        prediction = pipeline.predict(input_df)[0]

        # Uncomment ONLY if you trained using log1p(SalePrice)
        prediction = np.expm1(prediction)
        
        st.markdown("---")

        st.subheader("💰 Estimated Selling Price")

        # -----------------------------
        # Display Prediction
        # -----------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted House Price",
                value=f"${prediction:,.2f}"
            )

        with col2:

            if prediction < 150000:
                category = "Budget House 🟢"

            elif prediction < 250000:
                category = "Mid-Range House 🟡"

            else:
                category = "Premium House 🔴"

            st.metric(
                label="Category",
                value=category
            )

# =====================================================
# PROPERTY HIGHLIGHTS
# =====================================================

        st.markdown("---")
        st.subheader("🏠 Property Highlights")

        features = []

        if garage_cars > 0:
            features.append("🚗 Garage Available")

        if fireplaces > 0:
            features.append("🔥 Fireplace")

        if total_bsmt_sf > 0:
            features.append("🏗 Basement")

        if second_flr > 0:
            features.append("🏠 Second Floor")

        features.append(f"🍽 Kitchen Quality: {kitchen_qual}")
        features.append(f"📍 Neighborhood: {neighborhood}")
        features.append(f"🛁 {full_bath} Full Bathroom(s)")
        features.append(f"🛏 {bedrooms} Bedroom(s)")

        col1, col2 = st.columns(2)

        half = (len(features) + 1) // 2

        with col1:
            for item in features[:half]:
                st.success(item)

        with col2:
            for item in features[half:]:
                st.success(item)
       



        # -----------------------------
        # Input Summary
        # -----------------------------

        st.markdown("### Selected Property Details")


        st.dataframe(
            input_df[
                [
                    "Overall Qual",
                    "Overall Cond",
                    "Neighborhood",
                    "Gr Liv Area",
                    "Lot Area",
                    "Garage Cars",
                    "Garage Area",
                    "Year Built",
                    "Full Bath",
                    "Kitchen Qual"
                ]
            ],
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction Failed.")

        st.exception(e)

# =====================================================
# FOOTER
# =====================================================


st.markdown("---")

st.caption(
"""
Developed by **Fahad Rizvi**

Machine Learning Model: **XGBoost Regressor**

Built with **Python • Streamlit • Scikit-Learn**
"""
)