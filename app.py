# ===============================================
# 🏠 House Price Prediction App (Final Stable Version)
# ===============================================

import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Predictor + SHAP", layout="wide")
st.title("🏠 House Price Predictor (Final Stable Version)")
st.caption("Works with any CSV — automatic preprocessing via trained pipeline")

# ------------------------------
# Load model (Pipeline handles preprocessing)
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("artifacts/model_xgb.pkl")
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ------------------------------
# Input Options
# ------------------------------
st.header("🧾 Input Data")

mode = st.radio(
    "How do you want to provide inputs?",
    ["Upload CSV row(s)", "Type manually (demo)"],
    horizontal=True
)

df_input = None

# Upload CSV
if mode == "Upload CSV row(s)":
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(df_input.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Manual input
else:
    st.write("🧍 Manual Input Mode (demo)")
    vals = {
        "OverallQual": st.number_input("OverallQual", 1, 10, 6),
        "GrLivArea": st.number_input("GrLivArea (sq ft)", 500, 5000, 1500),
        "GarageCars": st.number_input("GarageCars", 0, 4, 2),
        "TotalBsmtSF": st.number_input("TotalBsmtSF (sq ft)", 0, 3000, 850),
    }
    df_input = pd.DataFrame([vals])

# ------------------------------
# Prediction + Explainability

if st.button("Predict"):
    if model is None:
        st.error("❌ Model not loaded. Please check artifacts folder.")
    elif df_input is None or df_input.empty:
        st.warning("⚠️ Please upload a CSV or enter values first.")
    else:
        try:
            # Add TotalSF automatically if missing
            if "TotalSF" not in df_input.columns:
                df_input["TotalSF"] = (
                    df_input.get("TotalBsmtSF", 0)
                    + df_input.get("1stFlrSF", 0)
                    + df_input.get("2ndFlrSF", 0)
                )

            preds = model.predict(df_input)
            st.success(f"✅ Predicted Sale Price: ${float(np.median(preds)):.2f}")
            st.write("### Prediction Results")
            st.dataframe(pd.DataFrame({"PredictedPrice": preds}))
            st.write("---")
            st.header("🧠 SHAP Explainability")

            try:
                # Use model directly or from pipeline
                inner_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(
                    model.named_steps["pre"].transform(df_input)
                    if hasattr(model, "named_steps")
                    else df_input
                )
                st.set_option("deprecation.showPyplotGlobalUse", False)
                shap.summary_plot(
                    shap_values,
                    model.named_steps["pre"].transform(df_input)
                    if hasattr(model, "named_steps")
                    else df_input,
                    plot_type="bar",
                    show=False,
                )
                st.pyplot(bbox_inches="tight")
            except Exception as shap_error:
                st.warning(f"⚠️ SHAP visualization skipped: {shap_error}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

st.write("---")
st.info("💡 Tip: Works with original Kaggle test.csv or any partial CSV — preprocessing is automatic.")
st.caption("Built by Tara — powered by XGBoost, Optuna, SHAP & Streamlit 🚀")
