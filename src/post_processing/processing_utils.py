import plotly.express as px
import streamlit as st
import matplotlib.cm as cm

from tempfile import NamedTemporaryFile

def get_temp_video_path(video_file, session_key="labeled_video_path"):
    """
    Retrieve the temporary video path from Streamlit's session state.

    This function checks if a specific session key exists in the Streamlit session
    state and returns the associated value, typically the path to a labeled video.

    Args:
        video_file (str): The name of the video file (currently unused in this function).
        session_key (str, optional): The session state key to look for. 
            Defaults to "labeled_video_path".

    Returns:
        str or None: The path stored in session state under the given key, 
        or None if the key does not exist.
    """
    if session_key in st.session_state:
        return st.session_state[session_key]

    # Create a temporary file and store its path in session_state
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_file.read())
        st.session_state[session_key] = temp_video_file.name

    return st.session_state[session_key]

def assign_video_path(key="get_video_key"):
    """
    Assign the labeled video path to Streamlit's session state.

    This function checks if a labeled video path is already stored in session state.
    If not, it prompts the user to upload a labeled video file through Streamlit's file uploader.
    Once uploaded, the path is stored in the session state under the key "labeled_video_path".

    Args:
        key (str, optional): A unique key for the Streamlit file uploader widget.
            Defaults to "get_video_key".

    Side Effects:
        Updates `st.session_state["labeled_video_path"]` if a video is uploaded.
        Displays success messages in the Streamlit UI.
    """
    # Optional input for video path
    # just mark as success if it's already in session state
    if "labeled_video_path" in st.session_state:
        st.success("Labeled video path already in session state!")
    else:
        video_file = st.file_uploader(
            "Upload Labeled Video File", type=["mp4", "avi"], key=key)
        if video_file is not None:
            st.session_state["labeled_video_path"] = get_temp_video_path(
                video_file, session_key="labeled_video_path")
            st.success("Labeled video path assigned successfully!")

def get_all_matplotlib_cmaps():
    """
    Retrieve all available colormaps from the `matplotlib.cm` module.

    This function dynamically gathers all public colormaps defined in `matplotlib.cm`
    and stores them in a dictionary with their names as keys.

    Returns:
        dict: A dictionary where keys are colormap names (str) and values are the corresponding colormap objects.
    """
    cmap_dict = {}
    for name in dir(cm):
        if not name.startswith('_'):
            cmap_dict[name] = getattr(cm, name)
    return cmap_dict

def get_all_plotly_cmaps():
    """
    Retrieve all available Plotly colormaps from different color categories.

    This function extracts colormaps from the following Plotly color categories:
        - Sequential
        - Diverging
        - Cyclical
        - Qualitative

    It returns a dictionary mapping colormap names to their corresponding color scale lists.

    Returns:
        dict: A dictionary where keys are colormap names (str) and values are lists of color hex codes.
    """
    cmap_dict = {}
    for cmap_group in [px.colors.sequential,
                       px.colors.diverging,
                       px.colors.cyclical,
                       px.colors.qualitative]:
        for name in dir(cmap_group):
            if not name.startswith('_'):
                cmap_dict[name] = getattr(cmap_group, name)
    return cmap_dict

def get_plot_inputs(default_title="",
                    default_x="index",
                    default_y="value",
                    default_color="#00f900",  # Green
                    key_prefix=""):
    """
    Display Streamlit input widgets for plot customization and return user-defined values.

    This function renders four input fields in a single row using Streamlit columns:
    - A text input for the plot title
    - A text input for the x-axis label
    - A text input for the y-axis label
    - A color picker for the plot line color

    It allows default values and a key prefix to avoid widget ID collisions across reruns.

    Args:
        default_title (str): Default value for the plot title input field.
        default_x (str): Default label for the x-axis.
        default_y (str): Default label for the y-axis.
        default_color (str): Default color for the plot line, as a hex string.
        key_prefix (str): Optional prefix for Streamlit widget keys to ensure uniqueness.

    Returns:
        tuple: A tuple containing four values:
            - title (str): User-defined plot title.
            - x_label (str): User-defined x-axis label.
            - y_label (str): User-defined y-axis label.
            - color (str): User-selected color in hex format.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        title = st.text_input("Title (empty is valid)",
                              value=default_title,
                              label_visibility="visible",
                              key=f"{key_prefix}_title")
    with col2:
        x_label = st.text_input("X Axis",
                                value=default_x,
                                label_visibility="visible",
                                key=f"{key_prefix}_x")
    with col3:
        y_label = st.text_input("Y Axis",
                                value=default_y,
                                label_visibility="visible",
                                key=f"{key_prefix}_y")
    with col4:
        color = st.color_picker("Color",
                                default_color,
                                label_visibility="visible",
                                key=f"{key_prefix}_color")

    return title, x_label, y_label, color

def get_dual_y_axis_plot_inputs(default_title="",
                                default_x="index",
                                default_y1="value 1",
                                default_y2="value 2",
                                default_color1="#1f77b4",  # Blue
                                default_color2="#d62728",  # Red
                                key_prefix=""):
    """
    Display Streamlit input widgets for customizing a dual y-axis plot and return user-defined settings.

    This function renders input widgets across 7 columns to let users specify:
    - Plot title
    - X-axis label
    - Y-axis labels for both primary and secondary axes
    - Line colors for both data series
    - Option to invert the secondary y-axis

    Args:
        default_title (str): Default plot title.
        default_x (str): Default x-axis label.
        default_y1 (str): Default label for the primary y-axis.
        default_y2 (str): Default label for the secondary y-axis.
        default_color1 (str): Default color for the first line, as a hex string.
        default_color2 (str): Default color for the second line, as a hex string.
        key_prefix (str): Prefix for Streamlit widget keys to ensure uniqueness.

    Returns:
        tuple: A tuple containing:
            - title (str): User-defined plot title.
            - x_label (str): X-axis label.
            - y_label_1 (str): Primary y-axis label.
            - y_label_2 (str): Secondary y-axis label.
            - color_1 (str): Hex color for the primary data line.
            - color_2 (str): Hex color for the secondary data line.
            - invert_y (bool): Whether to invert the secondary y-axis.
    """
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        title = st.text_input("Title",
                              value=default_title,
                              label_visibility="visible",
                              key=f"{key_prefix}_title")
    with col2:
        x_label = st.text_input("X Axis",
                                value=default_x,
                                label_visibility="visible",
                                key=f"{key_prefix}_x")
    with col3:
        y_label_1 = st.text_input("Y Axis 1",
                                  value=default_y1,
                                  label_visibility="visible",
                                  key=f"{key_prefix}_y1")
    with col4:
        y_label_2 = st.text_input("Y Axis 2",
                                  value=default_y2,
                                  label_visibility="visible",
                                  key=f"{key_prefix}_y2")
    with col5:
        color_1 = st.color_picker("Color 1",
                                  default_color1,
                                  label_visibility="visible",
                                  key=f"{key_prefix}_color1")
    with col6:
        color_2 = st.color_picker("Color 2",
                                  default_color2,
                                  label_visibility="visible",
                                  key=f"{key_prefix}_color2")
    with col7:
        # button to invert the secondary y axis for clearer visibility
        invert_y = st.checkbox("Invert Y Axis 2",
                               value=False,
                               label_visibility="visible",
                               key=f"{key_prefix}_invert_y")

    return title, x_label, y_label_1, y_label_2, color_1, color_2, invert_y

