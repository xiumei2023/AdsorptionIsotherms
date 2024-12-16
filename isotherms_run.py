import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
import streamlit as st
from io import BytesIO

# Streamlit Title
st.title("Isotherm Data Analysis Interface")

# Reminder for File Format
st.info("Reminder: The first row of the uploaded Excel file must contain column headers.\n\n"
        "Required Columns:\n- **First Column**: 'Ce (mg/L)'\n- **Second Column**: 'qe (mg/g)'.\n\n"
        "From the second row onwards, numerical data is required.")

# Define Langmuir and Freundlich Models
def langmuir_isotherm(Ce, q_m, k_L):
    return (q_m * k_L * Ce) / (1 + k_L * Ce)

def freundlich_isotherm(Ce, k_F, n):
    return k_F * Ce**(1 / n)

# Function for Isotherm Fitting
def run_isotherm_fitting(uploaded_file):
    results_list = []
    figure_paths = []
    combined_data = []  # Store original data and fitting results

    # Load Excel file
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names

    # Loop through each sheet
    for sheet in sheet_names:
        data = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=1, usecols=[0, 1])
        data.columns = ['Ce (mg/L)', 'qe (mg/g)']
        Ce_data = data['Ce (mg/L)'].dropna().values
        qe_data = data['qe (mg/g)'].dropna().values

        if len(Ce_data) == 0 or len(qe_data) == 0:
            st.warning(f"No valid data in sheet '{sheet}'. Skipping...")
            continue

        # lmfit setup
        langmuir_model = Model(langmuir_isotherm)
        langmuir_params = langmuir_model.make_params(q_m=np.max(qe_data), k_L=0.1)

        freundlich_model = Model(freundlich_isotherm)
        freundlich_params = freundlich_model.make_params(k_F=1.0, n=1.0)

        # Fit models
        langmuir_result = langmuir_model.fit(qe_data, Ce=Ce_data, params=langmuir_params)
        freundlich_result = freundlich_model.fit(qe_data, Ce=Ce_data, params=freundlich_params)

        # Combine original and fitting data into a DataFrame
        combined_df = pd.DataFrame({
            "Ce (mg/L)": Ce_data,
            "qe (mg/g)": qe_data,
            "Langmuir Fit": langmuir_result.best_fit,
            "Freundlich Fit": freundlich_result.best_fit
        })
        combined_df['Sheet'] = sheet  # Add sheet name
        combined_data.append(combined_df)

        # Store fitting results
        results_list.append({
            'Sheet': sheet, 'Model': 'Langmuir',
            'q_m': langmuir_result.params['q_m'].value,
            'k_L': langmuir_result.params['k_L'].value
        })
        results_list.append({
            'Sheet': sheet, 'Model': 'Freundlich',
            'k_F': freundlich_result.params['k_F'].value,
            'n': freundlich_result.params['n'].value
        })

        # Generate Figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Ce_data, qe_data, color='black', label="Data")
        ax.plot(Ce_data, langmuir_result.best_fit, 'r-', label="Langmuir Fit")
        ax.plot(Ce_data, freundlich_result.best_fit, 'g--', label="Freundlich Fit")
        ax.set_xlabel("Ce (mg/L)")
        ax.set_ylabel("qe (mg/g)")
        ax.legend()
        ax.set_title(f"Isotherm Fits for {sheet}")

        # Save figure to memory
        fig_io = BytesIO()
        plt.savefig(fig_io, format="png")
        plt.close(fig)
        figure_paths.append(fig_io)

    # Return Results
    summary_df = pd.DataFrame(results_list)
    combined_export = pd.concat(combined_data, axis=0)
    return summary_df, figure_paths, combined_export

# Streamlit File Upload
uploaded_file = st.file_uploader("Upload Excel File (Isotherm Data)", type=["xlsx"])

# Run Analysis
if uploaded_file:
    st.success("File uploaded successfully.")
    summary_df, figure_paths, combined_export = run_isotherm_fitting(uploaded_file)

    # Display Results Table
    if summary_df is not None and not summary_df.empty:
        st.write("### Fitting Results")
        st.dataframe(summary_df)

        # Provide Download for Fitting Results
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fitting Results as CSV",
            data=csv_data,
            file_name="isotherm_fitting_results.csv",
            mime="text/csv"
        )

        # Provide Download for Combined Original + Fitting Data
        combined_csv = combined_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Original and Fitting Data",
            data=combined_csv,
            file_name="isotherm_original_and_fitting_data.csv",
            mime="text/csv"
        )

    # Display Figures
    if figure_paths:
        st.write("### Fitting Figures")
        for idx, fig_io in enumerate(figure_paths):
            st.image(fig_io, caption=f"Sheet {idx + 1}", use_container_width=True)
