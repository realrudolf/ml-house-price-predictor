import streamlit as st
import pandas as pd
import joblib

# Load trained model and dataset
model = joblib.load('model.pkl')
df = pd.read_csv('kc_house_data.csv')

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.caption("Estimate the value of a home based on basic features. Powered by Machine Learning.")

# --- User Inputs ---
bedrooms = st.slider('🛏️ Number of Bedrooms', 1, 10, 3)
bathrooms = st.slider('🛁 Number of Bathrooms', 1, 5, 2)
sqft = st.number_input('📐 Living Area (sqft)', min_value=200, max_value=10000, value=1500, step=50)

# --- Prediction ---
if st.button('🔍 Predict Price'):
    input_data = [[bedrooms, bathrooms, sqft]]
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: **${prediction:,.0f}**")

    st.markdown("---")
    st.markdown(f"""
    ### 📊 Prediction Details
    - **Bedrooms:** {bedrooms}  
    - **Bathrooms:** {bathrooms}  
    - **Living space:** {sqft} sqft
    """)

    # Display similar homes from dataset
    similar = df[
        (df['bedrooms'] == bedrooms) &
        (df['bathrooms'] == bathrooms) &
        (df['sqft_living'].between(sqft - 200, sqft + 200))
    ]

    if not similar.empty:
        st.markdown("### 🏘️ Similar Homes in the Dataset")
        st.dataframe(similar[['bedrooms', 'bathrooms', 'sqft_living', 'price']].head(3))
    else:
        st.info("No similar homes found in dataset.")

# --- Info about model limitations ---
st.markdown("---")
st.info("""
ℹ️ **Note:**  
This model was trained on housing data from the Seattle area (2014–2015) and performs best for standard homes  
(without luxury features like pools, waterfronts, or very unusual sizes).

Prediction is based on: **bedrooms**, **bathrooms**, and **living area (sqft)** only.
""")

# Show data range from dataset
min_sqft = int(df['sqft_living'].min())
max_sqft = int(df['sqft_living'].max())

st.caption(f"(The model was trained on homes between {min_sqft} and {max_sqft} sqft in size.)")
st.caption("Made with ❤️ by Kacper Kubała HYPERCOLOR")