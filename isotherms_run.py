import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
import streamlit as st
from io import BytesIO

# Streamlit Title
st.title("Isotherm Data Analysis Interface")

# Define Langmuir and Freundlich Models
def langmuir_isotherm(Ce, q_m, k_L):
    return (q_m * k_L * Ce) / (1 + k_L * Ce)

def freundlich_isotherm(Ce, k_F, n):
    return k_F * Ce**(1 / n)

# Function for Isotherm Fitting
def run_isotherm_fitting(uploaded_file):
    results_list = []
    figure_paths = []  # Store in-memory figures

    # Load Excel file
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names

    # Loop through each sheet
    for sheet in sheet_names:
        # Load data and clean
        data = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=1, usecols=[0, 1])
        data.columns = ['Ce(mg/L)', 'qe(mg/g)']
        Ce_data = data['Ce(mg/L)'].dropna().values
        qe_data = data['qe(mg/g)'].dropna().values

        # Skip sheets with no valid data
        if len(Ce_data) == 0 or len(qe_data) == 0:
            st.warning(f"No valid data in {sheet}. Skipping...")
            continue

        # lmfit setup for Langmuir and Freundlich models
        langmuir_model = Model(langmuir_isotherm)
        langmuir_params = langmuir_model.make_params(q_m=np.max(qe_data), k_L=0.1)

        freundlich_model = Model(freundlich_isotherm)
        freundlich_params = freundlich_model.make_params(k_F=1.0, n=1.0)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Ce_data, qe_data, color='black', label="Data")

        try:
            # Langmuir Fitting
            langmuir_result = langmuir_model.fit(qe_data, Ce=Ce_data, params=langmuir_params)
            ax.plot(Ce_data, langmuir_result.best_fit, 'r-', label="Langmuir Fit")
            r_squared_langmuir = 1 - np.sum(langmuir_result.residual**2) / np.sum((qe_data - np.mean(qe_data))**2)
            results_list.append({
                'Sheet': sheet, 'Model': 'Langmuir', 
                'q_m': langmuir_result.params['q_m'].value, 'k_L': langmuir_result.params['k_L'].value, 
                'R^2': r_squared_langmuir
            })
        except Exception as e:
            st.warning(f"Langmuir fitting failed for {sheet}: {e}")

        try:
            # Freundlich Fitting
            freundlich_result = freundlich_model.fit(qe_data, Ce=Ce_data, params=freundlich_params)
            ax.plot(Ce_data, freundlich_result.best_fit, 'g--', label="Freundlich Fit")
            r_squared_freundlich = 1 - np.sum(freundlich_result.residual**2) / np.sum((qe_data - np.mean(qe_data))**2)
            results_list.append({
                'Sheet': sheet, 'Model': 'Freundlich', 
                'k_F': freundlich_result.params['k_F'].value, 'n': freundlich_result.params['n'].value, 
                'R^2': r_squared_freundlich
            })
        except Exception as e:
            st.warning(f"Freundlich fitting failed for {sheet}: {e}")

        # Customize and Save Figure to Memory
        ax.set_xlabel("Ce (mg/L)")
        ax.set_ylabel("qe (mg/g)")
        ax.legend()
        ax.set_title(f"Isotherm Fits for {sheet}")
        
        fig_io = BytesIO()
        plt.savefig(fig_io, format="png")
        plt.close(fig)
        figure_paths.append(fig_io)

    # Return Results and Figures
    summary_df = pd.DataFrame(results_list)
    return summary_df, figure_paths

# Streamlit File Upload
uploaded_file = st.file_uploader("Upload Excel File (Isotherm Data)", type=["xlsx"])

# Run Analysis
if uploaded_file:
    st.success("File uploaded successfully.")
    summary_df, figure_paths = run_isotherm_fitting(uploaded_file)

    # Display Results Table
    if summary_df is not None and not summary_df.empty:
        st.write("### Fitting Results")
        st.dataframe(summary_df)
    else:
        st.error("No fitting results to display.")

    # Display Figures
    if figure_paths:
        st.write("### Fitting Figures")
        for idx, fig_io in enumerate(figure_paths):
            st.image(fig_io, caption=f"Sheet {idx + 1}", use_container_width=True)

