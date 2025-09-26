import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing
model = joblib.load('regression_model.pkl')
pca = joblib.load('pca.pkl')
scaler = joblib.load('scaler.pkl')
mlb = joblib.load('mlb.pkl')

st.title("🎬 Movie Rating Predictor")

# --- Manual Input Form ---
st.subheader("🎯 Predict Single Movie Rating Manually")

with st.form("input_form"):
    runtime = st.number_input("Runtime (in minutes)", min_value=30, max_value=300, value=120)
    year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
    votes = st.number_input("Number of Votes", min_value=0, value=5000)
    genre = st.text_input("Genre (comma-separated)", value="Action, Drama")

    submitted = st.form_submit_button("Predict Rating")

    if submitted:
        try:
            # Preprocess input
            genre_list = [g.strip() for g in genre.split(',')]
            genre_encoded = pd.DataFrame([mlb.transform([genre_list])[0]], columns=mlb.classes_)

            input_data = pd.DataFrame([{
                'RunTime': runtime,
                'YEAR': year,
                'VOTES': votes
            }])

            # Merge with genre features
            for col in mlb.classes_:
                input_data[col] = genre_encoded.get(col, 0)

            # Match training feature order
            input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

            # Scale + PCA
            input_scaled = scaler.transform(input_data)
            input_pca = pca.transform(input_scaled)

            # Predict
            rating_pred = model.predict(input_pca)[0]
            st.success(f"🎯 Predicted IMDb Rating: {rating_pred:.2f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")

# --- CSV Upload ---
st.subheader("📤 Batch Predict from Uploaded CSV File")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Cleanup ---
        df['YEAR'] = df['YEAR'].astype(str).str.extract(r'(\d{4})')
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df['GENRE'] = df['GENRE'].fillna("Unknown").str.strip()
        df['VOTES'] = pd.to_numeric(df['VOTES'].astype(str).str.replace(',', '', regex=False), errors='coerce')
        if 'Gross' in df.columns:
            df['Gross'] = pd.to_numeric(df['Gross'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        if 'RATING' in df.columns:
            df = df[df['RATING'].notna()]

        st.write("✅ Uploaded data (after cleanup):")
        st.dataframe(df.head())

        # --- Predict button ---
        if st.button("🔮 Predict for Uploaded Data"):
            try:
                # Transform genres into multi-hot encoding
                df['GENRE'] = df['GENRE'].apply(lambda x: [g.strip() for g in str(x).split(',')])
                genre_encoded = pd.DataFrame(mlb.transform(df['GENRE']), columns=mlb.classes_)

                input_data = df[['RunTime', 'YEAR', 'VOTES']].copy()
                input_data = pd.concat([input_data, genre_encoded], axis=1)

                # Reindex to match training features
                input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

                # Fill NaNs with medians
                for col in ['RunTime', 'YEAR', 'VOTES']:
                    input_data[col] = input_data[col].fillna(input_data[col].median())

                # Scale + PCA
                input_scaled = scaler.transform(input_data)
                input_pca = pca.transform(input_scaled)

                # Predict all at once
                preds = model.predict(input_pca)
                df['Predicted_Rating'] = preds

                st.success("✅ Predictions complete!")
                st.dataframe(df[['RunTime', 'YEAR', 'VOTES', 'GENRE', 'Predicted_Rating']].head(20))

            except Exception as e:
                st.error(f"❌ Error during batch prediction: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
