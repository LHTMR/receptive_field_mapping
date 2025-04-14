import streamlit as st
from src.post_processing.datadlc import DataDLC
from src.post_processing.outlierimputer import OutlierImputer
from src.post_processing.dataneuron import DataNeuron
from src.post_processing.mergeddata import MergedData
from src.post_processing.plotting import Plotting
import pandas as pd
from tempfile import NamedTemporaryFile
import plotly.express as px

st.title("Post Processing")
st.write("This page is for post processing the prediction results together with recorded neuron data.")

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Labeled Data", "Neuron Data", "Merged Data"]
    )

# Tab for file uploaders
with tab1:
    st.header("Upload Files")

    # File uploader for DLC data
    dlc_file = st.file_uploader("Upload DLC Data H5 file that was predicted from the video", type=["h5"])
    # Initialize data_dlc in session state if not already present
    if "data_dlc" not in st.session_state:
        st.session_state.data_dlc = None

    if dlc_file is not None:
        st.success(f"Uploaded DLC file: {dlc_file.name}")

        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(dlc_file.read())
            temp_file_path = temp_file.name

        # Read the HDF5 file using pandas
        try:
            df = pd.read_hdf(temp_file_path)
            # Show df header
            st.write(df.head())
            try:
                # Process the DLC data and store it in session state
                st.session_state.data_dlc = DataDLC(temp_file_path)
                st.success("DLC data processed successfully!")
            except Exception as e:
                st.error(f"Error processing DLC data: {e}")
        except Exception as e:
            st.error(f"Error reading HDF5 file: {e}")


    # Use the stored data_dlc object for further processing
    if st.session_state.data_dlc is not None:
        st.write("DLC data is ready for further processing.")

        # Button to impute outliers
        if st.checkbox("Show Outliers"):
            st.write("""
                     These two plots are to visualize any potential outliers in the labeled
                     data from its derivatives. If the data looks out of the norm, then
                     consider taking the time to impute the data below.
                     (It could take around 5 minutes and 10 minutes for the square and monofilament data respectively)
                     """)
            try:
                # Make derivative interactive plots of data_dlc.df_square and data_dlc.df_monofil
                df_square_derivative = OutlierImputer.transform_to_derivative(st.session_state.data_dlc.df_square)
                df_monofil_derivative = OutlierImputer.transform_to_derivative(st.session_state.data_dlc.df_monofil)

                fig_square = px.line(df_square_derivative, title="Square Derivative")
                fig_monofil = px.line(df_monofil_derivative, title="Monofilament Derivative")
                st.plotly_chart(fig_square, use_container_width=True)
                st.plotly_chart(fig_monofil, use_container_width=True)
            except Exception as e:
                st.error(f"Error imputing outliers: {e}")
        
        # Impute outliers for the square points
        if st.checkbox("Impute Square Outliers (Optional)"):
            try:
                # Get the standard deviation threshold from the user
                std_threshold = st.number_input(
                    "Enter the standard deviation threshold for outlier imputation:",
                    min_value=0.1, value=4.0, step=0.1
                )

                # Add a button to confirm the input and start processing
                if st.button("Impute Square Outliers"):
                    st.session_state.data_dlc.impute_outliers(
                        std_threshold=std_threshold,
                        square=True,
                        filament=False
                    )
                    st.success("Outliers imputed successfully for the square points!")

                    #! SOMETHING IS WRONG WITH THIS PART, the shown plot is not imputed
                    # Make derivative interactive plot of imputed data_dlc.df_square
                    df_square_derivative_after = OutlierImputer.transform_to_derivative(
                        st.session_state.data_dlc.df_square
                    )
                    fig_square = px.line(df_square_derivative_after, title="Square Derivative After Imputation")
                    st.plotly_chart(fig_square, use_container_width=True, key="square_derivative_chart")
            except Exception as e:
                st.error(f"Error imputing outliers: {e}")

        # Impute outliers for the filament points
        if st.checkbox("Impute Filament Outliers (Optional)"):
            try:
                # Get the standard deviation threshold from the user
                std_threshold = st.number_input(
                    "Enter the standard deviation threshold for outlier imputation:",
                    min_value=0.1, value=4.0, step=0.1,
                    key="filament_threshold_input"
                )

                # Add a button to confirm the input and start processing
                if st.button("Impute Filament Outliers"):
                    st.session_state.data_dlc.impute_outliers(
                        std_threshold=std_threshold,
                        square=False,
                        filament=True
                    )
                    st.success("Outliers imputed successfully for the filament points!")

                    # Make derivative interactive plot of imputed data_dlc.df_monofil
                    df_monofil_derivative_after = OutlierImputer.transform_to_derivative(
                        st.session_state.data_dlc.df_monofil
                    )
                    fig_monofil = px.line(df_monofil_derivative_after, title="Monofilament Derivative")
                    st.plotly_chart(fig_monofil, use_container_width=True, key="filament_derivative_chart")
            except Exception as e:
                st.error(f"Error imputing outliers: {e}")

        if st.checkbox("Calculate Bending Coefficients"):
            try:
                st.session_state.data_dlc.get_bending_coefficients()
                st.success("Bending coefficients calculated successfully!")
                
                # Make interactive plot of bending coefficients
                fig_bending = px.line(st.session_state.data_dlc.df_bending_coefficients, title="Bending Coefficients")
                st.plotly_chart(fig_bending, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating bending coefficients: {e}")

        if st.checkbox("Apply Homography"):
            try:
                st.session_state.data_dlc.apply_homography()
                st.success("Homography applied successfully!")
                
                # Show header of transformed data
                st.write("Transformed Monofilament Data:")
                st.write(st.session_state.data_dlc.df_transformed_monofil.head())
            except Exception as e:
                st.error(f"Error applying homography: {e}")

# Placeholder for other features in the second tab
with tab2:
    st.header("Upload Files & Inputs")

    # File uploader for Neuron data
    neuron_file = st.file_uploader("Upload Neuron Data File that was collected during the video", type=["xlsx"])
    if neuron_file is not None:
        st.success(f"Uploaded Neuron file: {neuron_file.name}")

        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            temp_file.write(neuron_file.read())
            temp_file_path = temp_file.name

        # Read the Excel file using pandas
        try:
            df_neuron = pd.read_excel(temp_file_path)
            # Show df_neuron header
            st.write(df_neuron.head())
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    # Fetch inputs for original frequency and then target frequency of video fps
    original_fps = st.number_input("Enter the original frequency (hz) of the neuron data:", min_value=1, step=1)
    target_fps = st.number_input("Enter the target frequency (fps) for processing:", min_value=1, step=1)

    # Button to begin processing shows after file uploaded and inputs given
    if neuron_file is not None and original_fps and target_fps:
        if st.button("Process Neuron Data"):
            neuron_data = DataNeuron(temp_file_path, original_fps)
            st.success("Neuron data processed successfully!")

# Placeholder for other features in the second tab
with tab3:
    st.header("Other Features")
    st.write("Additional functionality can go here.")