import plotly.express as px
from tempfile import NamedTemporaryFile
import pandas as pd
from src.post_processing.plotting import Plotting
from src.post_processing.plotting_plotly import PlottingPlotly
from src.post_processing.mergeddata import MergedData
from src.post_processing.dataneuron import DataNeuron
from src.post_processing.outlierimputer import OutlierImputer
from src.post_processing.datadlc import DataDLC
import streamlit as st

#! schedule a presentation practice? ask Magnus if needed


def get_plot_inputs(default_title="",
                    default_x="index",
                    default_y="value",
                    default_color="#00f900",  # Green
                    key_prefix=""):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        title = st.text_input("Title (empty is valid)", value=default_title,
                              label_visibility="visible", key=f"{key_prefix}_title")
    with col2:
        x_label = st.text_input("X Axis", value=default_x,
                                label_visibility="visible", key=f"{key_prefix}_x")
    with col3:
        y_label = st.text_input("Y Axis", value=default_y,
                                label_visibility="visible", key=f"{key_prefix}_y")
    with col4:
        color = st.color_picker("Color", default_color,
                                label_visibility="visible", key=f"{key_prefix}_color")

    return title, x_label, y_label, color


def get_dual_y_axis_plot_inputs(default_title="",
                                default_x="index",
                                default_y1="value 1",
                                default_y2="value 2",
                                default_color1="#1f77b4",  # Blue
                                default_color2="#d62728",  # Red
                                key_prefix=""):
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        title = st.text_input("Title", value=default_title,
                              label_visibility="visible", key=f"{key_prefix}_title")
    with col2:
        x_label = st.text_input("X Axis", value=default_x,
                                label_visibility="visible", key=f"{key_prefix}_x")
    with col3:
        y_label_1 = st.text_input("Y Axis 1", value=default_y1,
                                  label_visibility="visible", key=f"{key_prefix}_y1")
    with col4:
        y_label_2 = st.text_input("Y Axis 2", value=default_y2,
                                  label_visibility="visible", key=f"{key_prefix}_y2")
    with col5:
        color_1 = st.color_picker("Color 1", default_color1,
                                  label_visibility="visible", key=f"{key_prefix}_color1")
    with col6:
        color_2 = st.color_picker("Color 2", default_color2,
                                  label_visibility="visible", key=f"{key_prefix}_color2")
    with col7:
        # button to invert the secondary y axis for clearer visibility
        invert_y = st.checkbox("Invert Y Axis 2", value=False,
                               label_visibility="visible", key=f"{key_prefix}_invert_y")

    return title, x_label, y_label_1, y_label_2, color_1, color_2, invert_y


def get_all_plotly_cmaps():
    cmap_dict = {}
    for cmap_group in [px.colors.sequential, px.colors.diverging, px.colors.cyclical, px.colors.qualitative]:
        for name in dir(cmap_group):
            if not name.startswith('_'):
                cmap_dict[name] = getattr(cmap_group, name)
    return cmap_dict


plotly_cmaps = get_all_plotly_cmaps()


def get_all_matplotlib_cmaps():
    import matplotlib.cm as cm
    cmap_dict = {}
    for name in dir(cm):
        if not name.startswith('_'):
            cmap_dict[name] = getattr(cm, name)
    return cmap_dict


matplotlib_cmaps = get_all_matplotlib_cmaps()


def get_temp_video_path(video_file, session_key="labeled_video_path"):
    if session_key in st.session_state:
        return st.session_state[session_key]

    # Create a temporary file and store its path in session_state
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())
        st.session_state[session_key] = temp_video_file.name

    return st.session_state[session_key]


def assign_video_path(key="get_video_key"):
    # Optional input for video path, just mark as success if it's already in session state
    if "labeled_video_path" in st.session_state:
        st.success("Labeled video path already in session state!")
    else:
        video_file = st.file_uploader(
            "Upload Labeled Video File", type=["mp4", "avi"], key=key)
        if video_file is not None:
            st.session_state["labeled_video_path"] = get_temp_video_path(
                video_file, session_key="labeled_video_path")
            st.success("Labeled video path assigned successfully!")


st.title("Post Processing")
st.write("This page is for post processing the prediction results together with recorded neuron data.")

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Labeled Data", "Neuron Data", "Merged Data"]
)

# Tab for processing labeled DLC data
with tab1:
    st.header("Manage File")

    if "h5_path" in st.session_state:
        st.session_state.data_dlc = DataDLC(st.session_state.h5_path)
        st.success("DLC data processed successfully!")

    else:
        # File uploader for DLC data
        dlc_file = st.file_uploader(
            "Upload DLC Data H5 file that was predicted from the video", type=["h5"])
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
                st.write(df)
                try:
                    # Process the DLC data and store it in session state
                    st.session_state.data_dlc = DataDLC(temp_file_path)
                    st.success("DLC data processed successfully!")
                except Exception as e:
                    st.error(f"Error processing DLC data: {e}")
            except Exception as e:
                st.error(f"Error reading HDF5 file: {e}")

    st.header("Processing")
    # Use the stored data_dlc object for further processing
    if st.session_state.data_dlc is not None:

        st.markdown("""
            #### Outliers
            The DLC data was processed to extract the square and monofilament
            points. The first two plots below are meant to visualize any potential
            outliers in the labeled data (square & monofilament) from its
            derivatives. If the data looks out of the norm, then
            consider taking the time to impute/replace the data below.
            (It could take around 5 minutes and 10 minutes for the square and
            monofilament data respectively)
            """)
        with st.expander("Plotting & Imputing", expanded=False):
            # Button to impute outliers
            if st.checkbox("Plot Derivatives for Outlier Detection"):
                try:
                    # Make derivative interactive plots of data_dlc.df_square and data_dlc.df_monofil
                    df_square_derivative = OutlierImputer.transform_to_derivative(
                        st.session_state.data_dlc.df_square)
                    df_monofil_derivative = OutlierImputer.transform_to_derivative(
                        st.session_state.data_dlc.df_monofil)

                    fig_square = px.line(
                        df_square_derivative, title="Square Derivative")
                    fig_square.update_layout(title_x=0.5)
                    fig_monofil = px.line(
                        df_monofil_derivative, title="Monofilament Derivative")
                    fig_monofil.update_layout(title_x=0.5)
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
                        st.success(
                            "Outliers imputed successfully for the square points!")

                        # Make derivative interactive plot of imputed data_dlc.df_square
                        df_square_derivative_after = OutlierImputer.transform_to_derivative(
                            st.session_state.data_dlc.df_square
                        )
                        fig_square = px.line(
                            df_square_derivative_after, title="Square Derivative After Imputation")
                        fig_square.update_layout(title_x=0.5)
                        st.plotly_chart(
                            fig_square, use_container_width=True, key="square_derivative_chart")
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
                        st.success(
                            "Outliers imputed successfully for the filament points!")

                        # Make derivative interactive plot of imputed data_dlc.df_monofil
                        df_monofil_derivative_after = OutlierImputer.transform_to_derivative(
                            st.session_state.data_dlc.df_monofil
                        )
                        fig_monofil = px.line(
                            df_monofil_derivative_after, title="Monofilament Derivative After Imputation")
                        fig_monofil.update_layout(title_x=0.5)
                        st.plotly_chart(
                            fig_monofil, use_container_width=True, key="filament_derivative_chart")
                except Exception as e:
                    st.error(f"Error imputing outliers: {e}")

        st.markdown("""
            #### Bending Coefficients
            The bending coefficients are the coefficients of the quadratic polynomial
            fitted to the x and y coordinates of the six monofilament points per frame.
            """)
        st.session_state.data_dlc.get_bending_coefficients()
        st.success("Bending coefficients calculated successfully!")
        with st.expander("Plotting", expanded=False):
            try:
                # Get customization inputs for the plot
                title, x_label, y_label, color = get_plot_inputs(
                    default_title="",
                    default_x="Frame",
                    default_y="Bending Value",
                    default_color="#00f900",
                    key_prefix="bending"
                )

                if st.checkbox("Plot Bending Coefficients"):
                    # Make interactive plot of bending coefficients
                    fig_bending = px.line(st.session_state.data_dlc.df_bending_coefficients,
                                          title="Bending Coefficients",
                                          color_discrete_sequence=[color])
                    fig_bending.update_layout(
                        title_x=0.5,
                        title=title,
                        xaxis_title=x_label,
                        yaxis_title=y_label
                    )
                    st.plotly_chart(fig_bending, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating bending coefficients: {e}")

        st.write("""
            #### Homography
            The homography points are the four corners of the square in the video.
            The order of the points are transformed thusly based on the inputs below:
            """)

        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input("Homography minimum", value=0, min_value=0,
                                    step=1, key="homo_min")
        with col2:
            end = st.number_input("Homography maximum", value=20, min_value=0,
                                  step=1, key="homo_max")
        st.write(f"""
                 - Top Left -> ({start}, {end})
                 - Top Right -> ({end}, {end})
                 - Bottom Left -> ({end}, {start})
                 - Bottom Right -> ({start}, {start}) \n
                 With this, the monofilament data will be also transformed to the
                 new square space for clearer visualization.
                 """)
        st.session_state.data_dlc.assign_homography_points(start, end)
        st.session_state.data_dlc.apply_homography()
        st.success("Homography applied successfully!")
        # Show header of transformed data
        st.write("Transformed Monofilament Data:")
        st.write(st.session_state.data_dlc.df_transformed_monofil)

        with st.expander("Plotting animations", expanded=False):
            st.write("(Below inputs for both interactive and video)")
            # Get customization inputs for the plot
            title, x_label, y_label, color = get_plot_inputs(
                default_title="Homography Plot",
                default_x="x (mm)",
                default_y="y (mm)",
                default_color="#00f900",
                key_prefix="homography"
            )

            if st.checkbox("Interactive Homography Plot"):
                try:
                    # Make interactive plot of homography points
                    fig = PlottingPlotly.plot_homography_interactive(
                        homography_points=st.session_state.data_dlc.homography_points,
                        df_transformed_monofil=st.session_state.data_dlc.df_transformed_monofil,
                        title=title,
                        x_label=x_label,
                        y_label=y_label,
                        color=color
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting homography interactive: {e}")

            st.write("(Below inputs for video only)")
            col1, col2 = st.columns(2)
            with col1:
                # Take a tuple input of two integers for figsize
                width = st.number_input("Figure Width", value=12, min_value=1,
                                        step=1, key="fig_width")
            with col2:
                height = st.number_input("Figure Height", value=12, min_value=1,
                                         step=1, key="fig_height")
            figsize = (width, height)

            if st.button("Generate and Download a Homography Video"):
                try:
                    video_bytes = PlottingPlotly.generate_homography_video(
                        st.session_state.data_dlc.homography_points,
                        st.session_state.data_dlc.df_transformed_monofil,
                        fps=30,
                        title=title,
                        x_label=x_label,
                        y_label=y_label,
                        color=color,
                        figsize=figsize
                    )
                    st.success("Video generated!")
                    st.video(video_bytes)

                    st.download_button(
                        label="Download Homography Video",
                        data=video_bytes,
                        file_name="homography_animation.mp4",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Error creating homography video: {e}")


# Tab for processing neuron data
with tab2:
    st.header("Upload Files & Inputs")

    # File uploader for Neuron data
    neuron_file = st.file_uploader(
        "Upload Neuron Data File that was collected during the video", type=["xlsx"])
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
            st.write(df_neuron)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    # Fetch inputs for original frequency and then target frequency of video fps
    col1, col2 = st.columns(2)
    with col1:
        original_fps = st.number_input("Enter the original sample rate of the neuron data:",
                                       value=None, min_value=1, step=1)
    with col2:
        target_fps = st.number_input("Enter the target sample rate of labeled data:",
                                     value=30, min_value=1, step=1)

    # Button to begin processing shows after file uploaded and inputs given
    st.session_state.neuron_data = None
    if neuron_file is not None and original_fps and target_fps:
        st.session_state.neuron_data = DataNeuron(temp_file_path, original_fps)
        st.write(st.session_state.neuron_data.df)
        st.success("Neuron data processed successfully!")

    st.header("Processing")
    if st.session_state.neuron_data is not None:
        st.write("""
            #### Neuron Data
            This will plot the neuron data and the IFF (Instantaneous Frequency Firing) data.
            The IFF was calculated as the reciprocal of the time difference between spikes.
            However, there is a possibility that this will contain too many rows for streamlit 
            depending on the original frequency of the neuron data. If this is the case, then
            consider skipping this steps visualization.
            """)
        with st.expander("Plotting", expanded=False):
            # Get customization inputs for the plot
            title, x_label, y_label_1, y_label_2, color_1, color_2, invert_y_2 = \
                get_dual_y_axis_plot_inputs(
                    default_title="Neuron Data, Original Sample Rate",
                    default_x="index",
                    default_y1="neuron spikes",
                    default_y2="IFF",
                    default_color1="#1f77b4",  # Blue
                    default_color2="#d62728",  # Red
                    key_prefix="neuron_plot"
                )
            if st.checkbox("Plot Neuron Data"):
                try:
                    # Make interactive dual-axis plot of neuron data
                    fig = PlottingPlotly.plot_dual_y_axis(
                        df=st.session_state.neuron_data.df,
                        columns=["Spikes", "IFF"],
                        title=title,
                        xlabel=x_label,
                        ylabel_1=y_label_1,
                        ylabel_2=y_label_2,
                        color_1=color_1,
                        color_2=color_2,
                        invert_y_2=invert_y_2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error showing neuron data: {e}")

        st.markdown("""
            #### Downsampling
            The neuron data is downsampled to the target frequency (fps) for
            merging with the labeled data in the next tab.
            For the downsampling, the max value of the IFF for the window is taken,
            whereas the sum of the Neuron spikes is taken for the window.
            """)
        # Downsample the neuron data to the target frequency
        st.session_state.neuron_data.downsample(target_fps)
        st.success("Neuron data downsampled successfully!")

        with st.expander("Plotting", expanded=False):
            # Get customization inputs for the plot
            title, x_label, y_label_1, y_label_2, color_1, color_2, invert_y_2 = \
                get_dual_y_axis_plot_inputs(
                    default_title="Neuron Data, Downsampled",
                    default_x="index",
                    default_y1="Neuron Spikes",
                    default_y2="IFF",
                    default_color1="#1f77b4",  # Blue
                    default_color2="#d62728",  # Red
                    key_prefix="neuron_plot_downsampled"
                )
            if st.checkbox("Downsample the Neuron Data"):
                try:
                    # Make interactive dual-axis plot of downsampled neuron data
                    fig = PlottingPlotly.plot_dual_y_axis(
                        df=st.session_state.neuron_data.downsampled_df,
                        columns=["Spikes", "IFF"],
                        title=title,
                        xlabel=x_label,
                        ylabel_1=y_label_1,
                        ylabel_2=y_label_2,
                        color_1=color_1,
                        color_2=color_2,
                        invert_y_2=invert_y_2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error downsampling neuron data: {e}")

# Tab for merging data and then plotting
with tab3:
    if "merged_data" not in st.session_state:
        st.session_state.merged_data = None

    st.header("Final Processing")
    st.markdown("""
        #### Merging
        The DLC data and the neuron data are merged together based on the time column.
        The merged data will be used for further analysis and visualization.
        The bending coefficient threshold input is for just that, to filter a
        new Bending_Binary column that will be used for the merging process,
        as well as cleaning of later plots for visualization.
        """)
    if st.session_state.data_dlc is not None and st.session_state.neuron_data is not None:
        # get threshold for bending coefficient
        threshold = st.number_input("Threshold for Bending Coefficient",
                                    value=0.8, min_value=0.0, step=0.01)

        # Button to process merged data
        if st.button("Process Merged Data"):
            # Merge the two datasets on the 'Time' column
            st.session_state.merged_data = MergedData(
                st.session_state.data_dlc,
                st.session_state.neuron_data,
                threshold=threshold
            )

        if st.session_state.merged_data is not None:
            # Show header of merged data
            st.success("Merged data processed successfully!")
            st.markdown("""
            Showing the csv of the data after merging with state thresholds applied.
            """)
            col1, col2 = st.columns(2)
            with col1:
                # get boolean input for thresholding
                bending = st.checkbox(
                    "Threshold on Bending", value=False, key="merged_bending")
            with col2:
                spikes = st.checkbox(
                    "Threshold on Spikes", value=False, key="merged_spikes")
            cleaned_df = st.session_state.merged_data.threshold_data(
                bending=bending, spikes=spikes)
            st.write(cleaned_df)
            st.write(
                f"Original length: {len(st.session_state.merged_data.df_merged)}")
            st.write(f"Cleaned length: {len(cleaned_df)}")
    else:
        st.warning(
            "Please upload and process both DLC and neuron data files before merging.")

    if st.session_state.merged_data is not None:

        # Button to plot merged data
        st.markdown("""
            #### Plotting Merging
            The plot below is meant to visualize the merging of
            neuron data and the labeled data together. Zoom in closer
            if necessary.
            """)
        with st.expander("Plotting", expanded=False):
            # Get customization inputs for the plot
            title, x_label, y_label_1, y_label_2, color_1, color_2, invert_y_2 = \
                get_dual_y_axis_plot_inputs(
                    default_title="Merged Data",
                    default_x="index",
                    default_y1="Neuron Spikes (filled)",
                    default_y2="Bending Binary",
                    default_color1="#1f77b4",  # Blue
                    default_color2="#d62728",  # Red
                    key_prefix="merged_plot"
                )

            if st.checkbox("Plot Merged Data"):
                try:
                    # Make interactive dual-axis plot of merged data
                    fig = PlottingPlotly.plot_dual_y_axis(
                        df=st.session_state.merged_data.df_merged,
                        columns=["Spikes_Filled", "Bending_Binary"],
                        title=title,
                        xlabel=x_label,
                        ylabel_1=y_label_1,
                        ylabel_2=y_label_2,
                        color_1=color_1,
                        color_2=color_2,
                        invert_y_2=invert_y_2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting merged data: {e}")

        # Animated RF Mapping
        st.markdown("""
            #### RF Mapping Animation
            The RF Mapping animation is generated by plotting the
            monofilament points in the transformed space and modified with
            the input column for size and color. \n
            The rows can also be thresholded to plot only the points
            that are above the threshold for the bending coefficient and/or
            non-zero for the neuron spikes.
            """)
        with st.expander("Generate RF Mapping Animation", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                bending = st.checkbox("Threshold on Bending",
                                      value=True,
                                      key="rf_mapping_bending")
                spikes = st.checkbox("Threshold on Spikes",
                                     value=True,
                                     key="rf_mapping_spikes")
                cmap = st.selectbox("Colormap", sorted(
                    matplotlib_cmaps.keys()), index=0, key="rf_mapping_cmap")
            with col2:
                title = st.text_input(
                    "Title (optional)", value="RF Mapping Animation", key="rf_mapping_title")
                x_label = st.text_input(
                    "X Axis", value="x (mm)", key="rf_mapping_x")
                y_label = st.text_input(
                    "Y Axis", value="y (mm)", key="rf_mapping_y")
            with col3:
                fps = st.number_input(
                    "FPS", value=30, min_value=1, step=1, key="rf_mapping_fps")
                columns = st.session_state.merged_data.df_merged.columns.tolist()
                size_col = st.selectbox(
                    "Size Column", columns, index=32, key="rf_mapping_size")
                color_col = st.selectbox(
                    "Color Column", columns, index=35, key="rf_mapping_color")

            # Button to generate RF Mapping Animation
            if st.button("Generate RF Mapping Animation"):
                try:
                    st.write("Generating RF Mapping Animation...")
                    video_bytes = PlottingPlotly.plot_rf_mapping_animated(
                        merged_data=st.session_state.merged_data,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=st.session_state.data_dlc.homography_points,
                        size_col=size_col,
                        color_col=color_col,
                        title=title,
                        cmap=cmap,
                        bending=bending,
                        spikes=spikes,
                        fps=fps
                    )
                    st.success("RF Mapping Animation generated successfully!")

                    # Display the video in Streamlit
                    st.video(video_bytes)

                    # Provide a download button for the video
                    st.download_button(
                        label="Download RF Mapping Animation",
                        data=video_bytes,
                        file_name="rf_mapping_animation.mp4",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Error generating RF Mapping Animation: {e}")

        st.markdown("""
            #### KDE / Scatter Plot (interactive)
            The KDE / Scatter plots are generated by plotting the 
            monofilament points in the transformed space and modified with
            the input column for size and color. \n
            The rows can also be thresholded to plot only the points
            that are above the threshold for the bending coefficient and/or
            non-zero for the neuron spikes.
            """)
        with st.expander("Plot KDE / Scatter", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                bending = st.checkbox("Bending threshold",
                                      value=True,
                                      key="kde_scatter_bending")
                spikes = st.checkbox("Spikes threshold",
                                     value=True,
                                     key="kde_scatter_spikes")
                selected_cmap = st.selectbox("Colormap", sorted(
                    plotly_cmaps.keys()), index=0, key="kde_scatter_cmap")

            with col2:
                title = st.text_input(
                    "Title (optional)", value="KDE / Scatter Plot", key="kde_scatter_title")
                x_label = st.text_input(
                    "X Axis", value="x (mm)", key="kde_scatter_x")
                y_label = st.text_input(
                    "Y Axis", value="y (mm)", key="kde_scatter_y")

            with col3:
                columns = st.session_state.merged_data.df_merged.columns.tolist()
                st.write("Inputs for Scatter Plot only:")
                size_col = st.selectbox(
                    "Size Column", columns, index=32, key="kde_scatter_size")
                color_col = st.selectbox(
                    "Color Column", columns, index=35, key="kde_scatter_color")

            with col4:
                st.write("Inputs for interactive KDE Plot only:")
                bw_bending = st.number_input(
                    "Bandwidth for Bending", value=0.1, min_value=0.01,
                    step=0.01, key="kde_scatter_bw_bending")
                bw_spikes = st.number_input(
                    "Bandwidth for Spikes", value=0.1, min_value=0.01,
                    step=0.01, key="kde_scatter_bw_spikes")
                cmap_spikes = st.selectbox(
                    "Colormap for Spikes",
                    sorted(plotly_cmaps.keys()),
                    index=1, key="kde_scatter_cmap_spikes")
                bw_threshold = st.number_input(
                    "Percentage threshold for Bandwidth", value=0.05, min_value=0.01,
                    step=0.05, key="kde_scatter_bw_threshold")

            if st.checkbox("Show KDE Density (interactive)"):
                try:
                    fig_kde = PlottingPlotly.plot_kde_density(
                        st.session_state.merged_data,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=st.session_state.data_dlc.homography_points,
                        bending=bending, spikes=spikes,
                        title=title,
                        xlabel=x_label, ylabel=y_label,
                        cmap_bending=selected_cmap, cmap_spikes=cmap_spikes,
                        bw_bending=bw_bending, bw_spikes=bw_spikes,
                        threshold_percentage=bw_threshold,
                    )
                    st.plotly_chart(fig_kde, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting KDE density: {e}")

            if st.checkbox("Show Scatter Plot (interactive)"):
                try:
                    fig_scatter = PlottingPlotly.plot_scatter(
                        st.session_state.merged_data,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=st.session_state.data_dlc.homography_points,
                        size_col=size_col,
                        color_col=color_col,
                        bending=bending, spikes=spikes,
                        title=title,
                        xlabel=x_label, ylabel=y_label,
                        cmap=selected_cmap
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting scatter plot: {e}")

            st.write("""
                Below inputs are only for a plot with background framing from
                the original video, they still however use some of the above
                inputs if they're not restated.
                """)
            # get video file to get frame for background
            assign_video_path(key="kde_scatter_video_key")

            if "labeled_video_path" in st.session_state:

                col1, col2 = st.columns(2)
                with col1:
                    matplotlib_cmap = st.selectbox(
                        "Matplotlib Colormap",
                        sorted(matplotlib_cmaps.keys()),
                        index=0, key="kde_scatter_matplotlib_cmap"
                    )
                    # select integer for indexed rame & data
                    index_frame = st.number_input(
                        "Index Frame",
                        value=101, min_value=0, step=1,
                        key="kde_scatter_index_frame"
                    )
                with col2:
                    width = st.number_input("Figure Width", value=12,
                                            min_value=1, step=1, key="fig_width_2")
                    height = st.number_input("Figure Height", value=12,
                                             min_value=1, step=1, key="fig_height_2")
                    figsize = (width, height)

                if st.checkbox("Show KDE Plot (background frame)"):
                    try:
                        fig, ax = Plotting.plot_kde_density(
                            merged_data=st.session_state.merged_data,
                            x_col="tf_FB2_x",
                            y_col="tf_FB2_y",
                            homography_points=st.session_state.data_dlc.homography_points,
                            bending=bending, spikes=spikes,
                            title=title,
                            xlabel=x_label, ylabel=y_label,
                            cmap=matplotlib_cmap,
                            figsize=figsize,
                            frame=True,
                            video_path=st.session_state.labeled_video_path,
                            index=index_frame
                        )
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting KDE density: {e}")

                if st.checkbox("Show Scatter Plot (background frame)"):
                    try:
                        fig, ax = Plotting.plot_scatter(
                            merged_data=st.session_state.merged_data,
                            x_col="tf_FB2_x",
                            y_col="tf_FB2_y",
                            homography_points=st.session_state.data_dlc.homography_points,
                            size_col=size_col,
                            color_col=color_col,
                            bending=bending, spikes=spikes,
                            title=title,
                            xlabel=x_label, ylabel=y_label,
                            cmap=matplotlib_cmap,
                            figsize=figsize,
                            frame=True,
                            video_path=st.session_state.labeled_video_path,
                            index=index_frame
                        )
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting scatter plot: {e}")

        st.markdown("""
            #### Scrolling Video Overlay
            The scrolling video overlay is generated by plotting the line plots
            of two target columns +- 50 frames from the current frame in the
            original video. \n
            The threshold for the bending coefficient is also shown visually
            if it is one of the target columns.
            """)
        with st.expander("Scrolling Video Inputs", expanded=False):
            assign_video_path(key="scrolling_video_key")

            if "labeled_video_path" in st.session_state:

                # Choose columns
                available_columns = st.session_state.merged_data.df_merged.columns.tolist()
                scroll_columns = st.multiselect(
                    "Select two columns for the scrolling plot",
                    available_columns,
                    default=["Bending_ZScore", "Spikes"]
                )

                if len(scroll_columns) != 2:
                    st.warning("Please select exactly two columns.")
                else:
                    # Color pickers
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        title = st.text_input("Title (optional)",
                                              value="Scrolling Overlay Video",
                                              key="scroll_title")
                    with col2:
                        color_1 = st.color_picker(f"Line color for {scroll_columns[0]}",
                                                  value="#1f77b4", key="color1_scroll")
                    with col3:
                        color_2 = st.color_picker(f"Line color for {scroll_columns[1]}",
                                                  value="#d62728", key="color2_scroll")

                    # Button to trigger generation
                    if st.button("Generate Scrolling Overlay Video"):
                        try:
                            st.write("Generating Scrolling Video...")

                            video_bytes = PlottingPlotly.generate_scroll_over_video(
                                merged_data=st.session_state.merged_data,
                                columns=scroll_columns,
                                video_path=st.session_state.labeled_video_path,
                                color_1=color_1,
                                color_2=color_2,
                                title=title
                            )

                            st.success(
                                "Scrolling Overlay Video generated successfully!")
                            st.video(video_bytes)

                            st.download_button(
                                label="Download Scrolling Overlay Video",
                                data=video_bytes,
                                file_name="scrolling_overlay_video.mp4",
                                mime="video/mp4"
                            )
                        except Exception as e:
                            st.error(
                                f"Error generating Scrolling Overlay Video: {e}")
