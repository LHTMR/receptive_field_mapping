from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import MutableMapping

import ipywidgets as widgets
import pandas as pd
import plotly.express as px
from IPython.display import Markdown, Video, clear_output, display

from src.post_processing.datadlc import DataDLC
from src.post_processing.dataneuron import DataNeuron
from src.post_processing.mergeddata import MergedData
from src.post_processing.outlierimputer import OutlierImputer
from src.post_processing.plotting_plotly import PlottingPlotly
from src.post_processing import processing_utils


class PostProcessingUI:
    def __init__(self, state: MutableMapping[str, object], shim) -> None:
        self.state = state
        self.shim = shim
        self.plotly_cmaps = processing_utils.get_all_plotly_cmaps()
        self.matplotlib_cmaps = processing_utils.get_all_matplotlib_cmaps()

    def build(self) -> widgets.Tab:
        labeled_tab = self._build_labeled_data_tab()
        neuron_tab = self._build_neuron_tab()
        merged_tab = self._build_merged_data_tab()

        tabs = widgets.Tab(children=[labeled_tab, neuron_tab, merged_tab])
        tabs.set_title(0, "Labeled Data")
        tabs.set_title(1, "Neuron Data")
        tabs.set_title(2, "Merged Data")
        return tabs

    # Labeled data tab -------------------------------------------------
    def _build_labeled_data_tab(self) -> widgets.Accordion:
        style = {"description_width": "160px"}
        full_width = widgets.Layout(width="100%")

        dlc_path_text = widgets.Text(
            description="Existing path:",
            placeholder="Optional: /path/to/predictions.h5",
            style=style,
            layout=full_width,
        )
        dlc_upload = widgets.FileUpload(accept=".h5", multiple=False, description="Upload .h5")
        dlc_load_button = widgets.Button(description="Load DLC data", icon="upload")
        dlc_load_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_load_dlc(_: widgets.Button) -> None:
            with dlc_load_output:
                clear_output()
                path_value = dlc_path_text.value.strip()
                temp_path = None

                if path_value:
                    candidate = Path(path_value.strip("\"").strip("'"))
                    if not candidate.exists():
                        print(f"File not found: {candidate}")
                        return
                    temp_path = candidate
                elif dlc_upload.value:
                    upload = next(iter(dlc_upload.value.values()))
                    suffix = Path(upload["metadata"]["name"]).suffix or ".h5"
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_file.write(upload["content"])
                    temp_file.close()
                    temp_path = Path(temp_file.name)
                    self.state["dlc_upload_path"] = temp_path
                    print(f"Uploaded file stored at {temp_path}")
                else:
                    print("Provide a path or upload a DLC prediction file (.h5).")
                    return

                try:
                    data_dlc = DataDLC(str(temp_path))
                    self.state["data_dlc"] = data_dlc
                    self.state["h5_path"] = temp_path
                    self.state["df_square_derivative_original"] = OutlierImputer.transform_to_derivative(
                        data_dlc.df_square.copy()
                    )
                    self.state["df_monofil_derivative_original"] = OutlierImputer.transform_to_derivative(
                        data_dlc.df_monofil.copy()
                    )
                    self.state["df_square_derivative_after"] = self.state["df_square_derivative_original"].copy()
                    self.state["df_monofil_derivative_after"] = self.state["df_monofil_derivative_original"].copy()

                    display(Markdown("**DLC square points preview (first rows):**"))
                    display(data_dlc.df_square.head())
                    display(Markdown("**Average likelihoods:**"))
                    likelihood_text = data_dlc.get_avg_likelihoods().replace("\n", "  \n")
                    display(Markdown(likelihood_text))
                except Exception as exc:
                    print(f"Failed to load DLC data: {exc}")

        dlc_load_button.on_click(on_load_dlc)

        imputation_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        std_square_input = widgets.FloatText(value=5.0, description="Square std", step=0.1, style=style)
        model_square_dropdown = widgets.Dropdown(
            options=["All Models"] + list(OutlierImputer.models.keys()),
            value="BR",
            description="Square model",
            style=style,
        )
        square_impute_button = widgets.Button(description="Impute square outliers", icon="magic")

        std_filament_input = widgets.FloatText(value=5.0, description="Filament std", step=0.1, style=style)
        model_filament_dropdown = widgets.Dropdown(
            options=["All Models"] + list(OutlierImputer.models.keys()),
            value="BR",
            description="Filament model",
            style=style,
        )
        filament_impute_button = widgets.Button(description="Impute filament outliers", icon="magic")

        def on_impute_square(_: widgets.Button) -> None:
            with imputation_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None:
                    print("Load DLC data first.")
                    return
                model = model_square_dropdown.value
                model_name = None if model == "All Models" else model
                try:
                    data_dlc.impute_outliers(
                        std_threshold=float(std_square_input.value),
                        square=True,
                        filament=False,
                        model_name=model_name,
                    )
                    self.state["df_square_derivative_after"] = OutlierImputer.transform_to_derivative(
                        data_dlc.df_square.copy()
                    )
                    print("Square points imputed successfully.")
                    json_path = Path("latest_square.json")
                    if json_path.exists():
                        with open(json_path, "r") as fh:
                            display(json.load(fh))
                except Exception as exc:
                    print(f"Square imputation failed: {exc}")

        def on_impute_filament(_: widgets.Button) -> None:
            with imputation_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None:
                    print("Load DLC data first.")
                    return
                model = model_filament_dropdown.value
                model_name = None if model == "All Models" else model
                try:
                    data_dlc.impute_outliers(
                        std_threshold=float(std_filament_input.value),
                        square=False,
                        filament=True,
                        model_name=model_name,
                    )
                    self.state["df_monofil_derivative_after"] = OutlierImputer.transform_to_derivative(
                        data_dlc.df_monofil.copy()
                    )
                    print("Filament points imputed successfully.")
                    json_path = Path("latest_filament.json")
                    if json_path.exists():
                        with open(json_path, "r") as fh:
                            display(json.load(fh))
                except Exception as exc:
                    print(f"Filament imputation failed: {exc}")

        square_impute_button.on_click(on_impute_square)
        filament_impute_button.on_click(on_impute_filament)

        derivative_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        plot_square_before_button = widgets.Button(description="Square derivative (before)", icon="line-chart")
        plot_square_after_button = widgets.Button(description="Square derivative (after)", icon="line-chart")
        plot_filament_before_button = widgets.Button(description="Filament derivative (before)", icon="line-chart")
        plot_filament_after_button = widgets.Button(description="Filament derivative (after)", icon="line-chart")

        def _plot_derivative(df: pd.DataFrame, title: str) -> None:
            fig = px.line(df, title=title)
            fig.update_layout(title_x=0.5, xaxis_title="Frame", yaxis_title="Derivative value")
            fig.show()

        def on_plot_square_before(_: widgets.Button) -> None:
            with derivative_output:
                clear_output()
                df = self.state.get("df_square_derivative_original")
                if df is None:
                    print("Load DLC data first.")
                    return
                _plot_derivative(df, "Square derivative (before imputation)")

        def on_plot_square_after(_: widgets.Button) -> None:
            with derivative_output:
                clear_output()
                df = self.state.get("df_square_derivative_after")
                if df is None:
                    print("Impute square outliers first.")
                    return
                _plot_derivative(df, "Square derivative (after imputation)")

        def on_plot_filament_before(_: widgets.Button) -> None:
            with derivative_output:
                clear_output()
                df = self.state.get("df_monofil_derivative_original")
                if df is None:
                    print("Load DLC data first.")
                    return
                _plot_derivative(df, "Filament derivative (before imputation)")

        def on_plot_filament_after(_: widgets.Button) -> None:
            with derivative_output:
                clear_output()
                df = self.state.get("df_monofil_derivative_after")
                if df is None:
                    print("Impute filament outliers first.")
                    return
                _plot_derivative(df, "Filament derivative (after imputation)")

        plot_square_before_button.on_click(on_plot_square_before)
        plot_square_after_button.on_click(on_plot_square_after)
        plot_filament_before_button.on_click(on_plot_filament_before)
        plot_filament_after_button.on_click(on_plot_filament_after)

        labeled_video_path_input = widgets.Text(
            value=str(self.state.get("processed_video_path", "")),
            description="Video path:",
            style=style,
            layout=full_width,
        )
        labeled_video_button = widgets.Button(description="Generate labeled video", icon="video")
        labeled_video_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_generate_labeled_video(_: widgets.Button) -> None:
            with labeled_video_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None:
                    print("Load DLC data first.")
                    return
                video_path_value = labeled_video_path_input.value.strip()
                if not video_path_value:
                    print("Provide the path to the source video (preprocessed).")
                    return
                video_path = Path(video_path_value.strip("\"").strip("'"))
                if not video_path.exists():
                    print(f"Video not found: {video_path}")
                    return
                try:
                    video_bytes = PlottingPlotly.generate_labeled_video(data_dlc, str(video_path))
                    display(Video(data=video_bytes, embed=True))
                    self.state["labeled_video_path"] = video_path
                except Exception as exc:
                    print(f"Failed to generate labeled video: {exc}")

        labeled_video_button.on_click(on_generate_labeled_video)

        bending_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        compute_bending_button = widgets.Button(description="Compute bending coefficients", icon="chart-line")
        bending_plot_button = widgets.Button(description="Plot bending coefficients", icon="line-chart")
        bending_title = widgets.Text(value="Bending Coefficients", description="Title", style=style)
        bending_xlabel = widgets.Text(value="Frame", description="X axis", style=style)
        bending_ylabel = widgets.Text(value="Bending value", description="Y axis", style=style)
        bending_color_picker = widgets.ColorPicker(value="#00f900", description="Line colour")

        def on_compute_bending(_: widgets.Button) -> None:
            with bending_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None:
                    print("Load DLC data first.")
                    return
                try:
                    data_dlc.get_bending_coefficients()
                    self.state["df_bending_coefficients"] = data_dlc.df_bending_coefficients
                    print("Bending coefficients calculated. Preview:")
                    display(data_dlc.df_bending_coefficients.head())
                except Exception as exc:
                    print(f"Failed to compute bending coefficients: {exc}")

        def on_plot_bending(_: widgets.Button) -> None:
            with bending_output:
                clear_output()
                data = self.state.get("df_bending_coefficients")
                if data is None:
                    print("Compute bending coefficients first.")
                    return
                df = pd.DataFrame({"Frame": range(len(data)), "Bending": data})
                fig = px.line(
                    df,
                    x="Frame",
                    y="Bending",
                    title=bending_title.value,
                    color_discrete_sequence=[bending_color_picker.value],
                )
                fig.update_layout(title_x=0.5, xaxis_title=bending_xlabel.value, yaxis_title=bending_ylabel.value)
                fig.show()

        compute_bending_button.on_click(on_compute_bending)
        bending_plot_button.on_click(on_plot_bending)

        homography_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        homography_video_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        homo_min_input = widgets.IntText(value=0, description="Minimum", style=style)
        homo_max_input = widgets.IntText(value=20, description="Maximum", style=style)
        apply_homography_button = widgets.Button(description="Apply homography", icon="project-diagram")
        homography_plot_button = widgets.Button(description="Interactive plot", icon="map")
        homography_video_button = widgets.Button(description="Homography animation", icon="film")
        homography_fig_width = widgets.IntText(value=12, description="Figure width", style=style)
        homography_fig_height = widgets.IntText(value=12, description="Figure height", style=style)
        homography_video_fps = widgets.IntText(value=30, description="FPS", style=style)

        def on_apply_homography(_: widgets.Button) -> None:
            with homography_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None:
                    print("Load DLC data first.")
                    return
                try:
                    data_dlc.assign_homography_points(int(homo_min_input.value), int(homo_max_input.value))
                    data_dlc.apply_homography()
                    self.state["homography_applied"] = True
                    print("Homography applied. Transformed monofilament preview:")
                    display(data_dlc.df_transformed_monofil.head())
                except Exception as exc:
                    print(f"Failed to apply homography: {exc}")

        def on_plot_homography(_: widgets.Button) -> None:
            with homography_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None or data_dlc.homography_points is None:
                    print("Apply homography first.")
                    return
                try:
                    fig = PlottingPlotly.plot_homography_interactive(
                        homography_points=data_dlc.homography_points,
                        df_transformed_monofil=data_dlc.df_transformed_monofil,
                        title="Homography Plot",
                        x_label="x (mm)",
                        y_label="y (mm)",
                        color="#00f900",
                    )
                    fig.show()
                except Exception as exc:
                    print(f"Failed to plot homography: {exc}")

        def on_homography_video(_: widgets.Button) -> None:
            with homography_video_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                if data_dlc is None or data_dlc.homography_points is None:
                    print("Apply homography first.")
                    return
                try:
                    figsize = (int(homography_fig_width.value), int(homography_fig_height.value))
                    video_bytes = PlottingPlotly.generate_homography_video(
                        data_dlc.homography_points,
                        data_dlc.df_transformed_monofil,
                        fps=int(homography_video_fps.value),
                        title="Homography Animation",
                        x_label="x (mm)",
                        y_label="y (mm)",
                        color="#00f900",
                        figsize=figsize,
                    )
                    display(Video(data=video_bytes, embed=True))
                except Exception as exc:
                    print(f"Failed to generate homography video: {exc}")

        apply_homography_button.on_click(on_apply_homography)
        homography_plot_button.on_click(on_plot_homography)
        homography_video_button.on_click(on_homography_video)

        upload_box = widgets.VBox(
            [widgets.HTML("<h4>Load DLC data</h4>"), dlc_path_text, dlc_upload, dlc_load_button, dlc_load_output],
            layout=widgets.Layout(gap="0.4em"),
        )

        imputation_box = widgets.VBox(
            [
                widgets.HTML("<h4>Outlier imputation</h4>"),
                widgets.HBox([std_square_input, model_square_dropdown, square_impute_button]),
                widgets.HBox([std_filament_input, model_filament_dropdown, filament_impute_button]),
                imputation_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        derivative_box = widgets.VBox(
            [
                widgets.HTML("<h4>Derivative plots</h4>"),
                widgets.HBox([plot_square_before_button, plot_square_after_button]),
                widgets.HBox([plot_filament_before_button, plot_filament_after_button]),
                derivative_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        labeled_video_box = widgets.VBox(
            [
                widgets.HTML("<h4>Regenerate labeled video</h4>"),
                labeled_video_path_input,
                labeled_video_button,
                labeled_video_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        bending_box = widgets.VBox(
            [
                widgets.HTML("<h4>Bending coefficients</h4>"),
                compute_bending_button,
                widgets.HBox([bending_plot_button, bending_color_picker]),
                widgets.HBox([bending_title, bending_xlabel, bending_ylabel]),
                bending_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        homography_box = widgets.VBox(
            [
                widgets.HTML("<h4>Homography & animation</h4>"),
                widgets.HBox([homo_min_input, homo_max_input, apply_homography_button]),
                widgets.HBox([homography_plot_button, homography_video_button]),
                widgets.HBox([homography_fig_width, homography_fig_height, homography_video_fps]),
                homography_output,
                homography_video_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        accordion = widgets.Accordion(
            [upload_box, imputation_box, derivative_box, labeled_video_box, bending_box, homography_box]
        )
        titles = [
            "Load DLC data",
            "Outlier imputation",
            "Derivative plots",
            "Regenerate labeled video",
            "Bending coefficients",
            "Homography",
        ]
        for idx, title in enumerate(titles):
            accordion.set_title(idx, title)
        return accordion

    # Neuron data tab --------------------------------------------------
    def _build_neuron_tab(self) -> widgets.Accordion:
        style = {"description_width": "160px"}
        full_width = widgets.Layout(width="100%")

        neuron_path_text = widgets.Text(
            description="Existing path:",
            placeholder="Optional: /path/to/neuron.csv or .xlsx",
            style=style,
            layout=full_width,
        )
        neuron_upload = widgets.FileUpload(accept=".csv,.xlsx", multiple=False, description="Upload data")
        original_fps_input = widgets.IntText(value=0, description="Original sample rate", style=style)
        target_fps_input = widgets.IntText(value=30, description="Target sample rate", style=style)

        neuron_load_button = widgets.Button(description="Load neuron data", icon="upload")
        neuron_load_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_load_neuron(_: widgets.Button) -> None:
            with neuron_load_output:
                clear_output()
                path_value = neuron_path_text.value.strip()
                temp_path = None
                if path_value:
                    candidate = Path(path_value.strip("\"").strip("'"))
                    if not candidate.exists():
                        print(f"File not found: {candidate}")
                        return
                    temp_path = candidate
                elif neuron_upload.value:
                    upload = next(iter(neuron_upload.value.values()))
                    suffix = Path(upload["metadata"]["name"]).suffix or ".csv"
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_file.write(upload["content"])
                    temp_file.close()
                    temp_path = Path(temp_file.name)
                    self.state["neuron_upload_path"] = temp_path
                    print(f"Uploaded file stored at {temp_path}")
                else:
                    print("Provide a path or upload a neuron data file (.csv or .xlsx).")
                    return

                if not original_fps_input.value or original_fps_input.value <= 0:
                    print("Set the original sample rate (Hz) before loading.")
                    return

                try:
                    data_neuron = DataNeuron(str(temp_path), int(original_fps_input.value))
                    self.state["neuron_data"] = data_neuron
                    self.state["neuron_path"] = temp_path
                    display(Markdown("**Neuron data preview:**"))
                    display(data_neuron.df.head())
                except Exception as exc:
                    print(f"Failed to load neuron data: {exc}")

        neuron_load_button.on_click(on_load_neuron)

        neuron_plot_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        neuron_plot_button = widgets.Button(description="Plot neuron data", icon="area-chart")
        neuron_plot_title = widgets.Text(value="Neuron data, original sample rate", description="Title", style=style)
        neuron_plot_xlabel = widgets.Text(value="Index", description="X axis", style=style)
        neuron_plot_ylabel1 = widgets.Text(value="Neuron spikes", description="Y1 axis", style=style)
        neuron_plot_ylabel2 = widgets.Text(value="IFF", description="Y2 axis", style=style)
        neuron_color1 = widgets.ColorPicker(value="#1f77b4", description="Spike colour")
        neuron_color2 = widgets.ColorPicker(value="#d62728", description="IFF colour")
        neuron_invert_checkbox = widgets.Checkbox(value=False, description="Invert IFF axis")

        def on_plot_neuron(_: widgets.Button) -> None:
            with neuron_plot_output:
                clear_output()
                data_neuron: DataNeuron | None = self.state.get("neuron_data")
                if data_neuron is None:
                    print("Load neuron data first.")
                    return
                try:
                    fig = PlottingPlotly.plot_dual_y_axis(
                        df=data_neuron.df,
                        columns=["Spike", "IFF"],
                        xlabel=neuron_plot_xlabel.value,
                        ylabel_1=neuron_plot_ylabel1.value,
                        ylabel_2=neuron_plot_ylabel2.value,
                        title=neuron_plot_title.value,
                        color_1=neuron_color1.value,
                        color_2=neuron_color2.value,
                        invert_y_2=neuron_invert_checkbox.value,
                    )
                    fig.show()
                except Exception as exc:
                    print(f"Failed to plot neuron data: {exc}")

        neuron_plot_button.on_click(on_plot_neuron)

        neuron_downsample_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        neuron_downsample_button = widgets.Button(description="Downsample neuron data", icon="compress")
        neuron_downsample_plot_button = widgets.Button(description="Plot downsampled data", icon="area-chart")

        def on_downsample_neuron(_: widgets.Button) -> None:
            with neuron_downsample_output:
                clear_output()
                data_neuron: DataNeuron | None = self.state.get("neuron_data")
                if data_neuron is None:
                    print("Load neuron data first.")
                    return
                if not target_fps_input.value or target_fps_input.value <= 0:
                    print("Set the target sample rate (Hz) before downsampling.")
                    return
                try:
                    data_neuron.downsample(int(target_fps_input.value))
                    self.state["neuron_downsampled_df"] = data_neuron.downsampled_df
                    print("Downsampled data preview:")
                    display(data_neuron.downsampled_df.head())
                except Exception as exc:
                    print(f"Downsampling failed: {exc}")

        def on_plot_downsampled(_: widgets.Button) -> None:
            with neuron_downsample_output:
                clear_output()
                data_neuron: DataNeuron | None = self.state.get("neuron_data")
                if data_neuron is None or data_neuron.downsampled_df is None:
                    print("Downsample the neuron data first.")
                    return
                try:
                    fig = PlottingPlotly.plot_dual_y_axis(
                        df=data_neuron.downsampled_df,
                        columns=["Spike", "IFF"],
                        xlabel="Frame",
                        ylabel_1="Sum of spikes",
                        ylabel_2="Max IFF",
                        title="Neuron data (downsampled)",
                        color_1=neuron_color1.value,
                        color_2=neuron_color2.value,
                        invert_y_2=neuron_invert_checkbox.value,
                    )
                    fig.show()
                except Exception as exc:
                    print(f"Failed to plot downsampled data: {exc}")

        neuron_downsample_button.on_click(on_downsample_neuron)
        neuron_downsample_plot_button.on_click(on_plot_downsampled)

        load_box = widgets.VBox(
            [
                widgets.HTML("<h4>Load neuron data</h4>"),
                neuron_path_text,
                neuron_upload,
                widgets.HBox([original_fps_input, target_fps_input, neuron_load_button]),
                neuron_load_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        plot_box = widgets.VBox(
            [
                widgets.HTML("<h4>Visualise original data</h4>"),
                widgets.HBox([neuron_plot_button, neuron_invert_checkbox]),
                widgets.HBox([neuron_plot_title, neuron_plot_xlabel]),
                widgets.HBox([neuron_plot_ylabel1, neuron_plot_ylabel2]),
                widgets.HBox([neuron_color1, neuron_color2]),
                neuron_plot_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        downsample_box = widgets.VBox(
            [
                widgets.HTML("<h4>Downsample & inspect</h4>"),
                widgets.HBox([neuron_downsample_button, neuron_downsample_plot_button]),
                neuron_downsample_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        accordion = widgets.Accordion([load_box, plot_box, downsample_box])
        accordion.set_title(0, "Load neuron data")
        accordion.set_title(1, "Visualise original data")
        accordion.set_title(2, "Downsample & inspect")
        return accordion

    # Merged data tab --------------------------------------------------
    def _build_merged_data_tab(self) -> widgets.Accordion:
        style = {"description_width": "160px"}

        max_gap_fill_slider = widgets.IntSlider(value=10, min=1, max=50, step=1, description="Max gap fill", style=style)
        threshold_slider = widgets.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.05, description="Z-score threshold", style=style)
        merge_button = widgets.Button(description="Merge DLC & neuron data", icon="link")
        merge_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        size_dropdown = widgets.Dropdown(description="Size column")
        color_dropdown = widgets.Dropdown(description="Colour column")

        def refresh_column_options() -> None:
            merged = self.state.get("merged_data")
            if merged is None:
                return
            columns = merged.df_merged.columns.tolist()
            size_dropdown.options = columns
            color_dropdown.options = columns
            if "Bending_ZScore" in columns:
                size_dropdown.value = "Bending_ZScore"
            if "Spike" in columns:
                color_dropdown.value = "Spike"

        def on_merge_clicked(_: widgets.Button) -> None:
            with merge_output:
                clear_output()
                data_dlc: DataDLC | None = self.state.get("data_dlc")
                neuron_data: DataNeuron | None = self.state.get("neuron_data")
                if data_dlc is None or neuron_data is None:
                    print("Load DLC and neuron data before merging.")
                    return
                try:
                    merged = MergedData(
                        data_dlc,
                        neuron_data,
                        max_gap_fill=int(max_gap_fill_slider.value),
                        threshold=float(threshold_slider.value),
                    )
                    self.state["merged_data"] = merged
                    print("Merged data preview:")
                    display(merged.df_merged.head())
                    refresh_column_options()
                except Exception as exc:
                    print(f"Merging failed: {exc}")

        merge_button.on_click(on_merge_clicked)

        bending_toggle = widgets.Checkbox(value=True, description="Apply bending threshold")
        spikes_toggle = widgets.Checkbox(value=True, description="Require spikes")
        filter_button = widgets.Button(description="Preview filtered rows", icon="table")
        filter_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_filter_clicked(_: widgets.Button) -> None:
            with filter_output:
                clear_output()
                merged: MergedData | None = self.state.get("merged_data")
                if merged is None:
                    print("Merge the data first.")
                    return
                df = merged.threshold_data(bending=bending_toggle.value, spikes=spikes_toggle.value)
                print(f"Rows after filtering: {len(df)}")
                display(df.head())

        filter_button.on_click(on_filter_clicked)

        plot_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        kde_button = widgets.Button(description="Interactive KDE", icon="satellite")
        scatter_button = widgets.Button(description="Interactive scatter", icon="braille")
        plot_title_input = widgets.Text(value="KDE / Scatter Plot", description="Title", style=style)
        plot_xlabel_input = widgets.Text(value="x (mm)", description="X axis", style=style)
        plot_ylabel_input = widgets.Text(value="y (mm)", description="Y axis", style=style)
        plotly_cmap_dropdown = widgets.Dropdown(options=sorted(self.plotly_cmaps.keys()), value="Viridis", description="Plotly cmap")
        plotly_spikes_cmap_dropdown = widgets.Dropdown(
            options=sorted(self.plotly_cmaps.keys()), value="Inferno", description="Spikes cmap"
        )
        bw_bending_input = widgets.FloatText(value=0.1, description="Bandwidth (bending)", style=style)
        bw_spikes_input = widgets.FloatText(value=0.1, description="Bandwidth (spikes)", style=style)
        bw_threshold_input = widgets.FloatText(value=0.05, description="% threshold", style=style)

        def _require_merge_and_homography() -> tuple[bool, DataDLC, MergedData] | tuple[bool, None, None]:
            merged: MergedData | None = self.state.get("merged_data")
            data_dlc: DataDLC | None = self.state.get("data_dlc")
            if merged is None or data_dlc is None or data_dlc.homography_points is None:
                print("Merge data and apply homography before plotting.")
                return False, None, None
            return True, data_dlc, merged

        def on_kde_clicked(_: widgets.Button) -> None:
            with plot_output:
                clear_output()
                ok, data_dlc, merged = _require_merge_and_homography()
                if not ok:
                    return
                try:
                    fig = PlottingPlotly.plot_kde_density_interactive(
                        merged,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=data_dlc.homography_points,
                        bending=bending_toggle.value,
                        spikes=spikes_toggle.value,
                        title=plot_title_input.value,
                        xlabel=plot_xlabel_input.value,
                        ylabel=plot_ylabel_input.value,
                        cmap_bending=plotly_cmap_dropdown.value,
                        cmap_spikes=plotly_spikes_cmap_dropdown.value,
                        bw_bending=float(bw_bending_input.value),
                        bw_spikes=float(bw_spikes_input.value),
                        threshold_percentage=float(bw_threshold_input.value),
                    )
                    fig.show()
                except Exception as exc:
                    print(f"Failed to plot KDE: {exc}")

        def on_scatter_clicked(_: widgets.Button) -> None:
            with plot_output:
                clear_output()
                ok, data_dlc, merged = _require_merge_and_homography()
                if not ok:
                    return
                try:
                    fig = PlottingPlotly.plot_scatter_interactive(
                        merged,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=data_dlc.homography_points,
                        size_col=size_dropdown.value,
                        color_col=color_dropdown.value,
                        bending=bending_toggle.value,
                        spikes=spikes_toggle.value,
                        title=plot_title_input.value,
                        xlabel=plot_xlabel_input.value,
                        ylabel=plot_ylabel_input.value,
                        cmap=plotly_cmap_dropdown.value,
                    )
                    fig.show()
                except Exception as exc:
                    print(f"Failed to plot scatter: {exc}")

        kde_button.on_click(on_kde_clicked)
        scatter_button.on_click(on_scatter_clicked)

        animation_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})
        rf_video_button = widgets.Button(description="RF map animation (video)", icon="play")
        scatter_video_button = widgets.Button(description="Scatter animation (video)", icon="play-circle")
        scroll_video_button = widgets.Button(description="Scrolling overlay video", icon="video-camera")
        animation_fps_input = widgets.IntText(value=30, description="Video FPS", style=style)
        animation_fig_width = widgets.IntText(value=12, description="Figure width", style=style)
        animation_fig_height = widgets.IntText(value=12, description="Figure height", style=style)

        def on_rf_video(_: widgets.Button) -> None:
            with animation_output:
                clear_output()
                ok, data_dlc, merged = _require_merge_and_homography()
                if not ok:
                    return
                try:
                    figsize = (int(animation_fig_width.value), int(animation_fig_height.value))
                    video_bytes = PlottingPlotly.plot_rf_mapping_animated(
                        merged,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=data_dlc.homography_points,
                        size_col=size_dropdown.value,
                        color_col=color_dropdown.value,
                        title=plot_title_input.value,
                        bending=bending_toggle.value,
                        spikes=spikes_toggle.value,
                        xlabel=plot_xlabel_input.value,
                        ylabel=plot_ylabel_input.value,
                        fps=int(animation_fps_input.value),
                        figsize=figsize,
                        cmap=plotly_cmap_dropdown.value,
                    )
                    display(Video(data=video_bytes, embed=True))
                except Exception as exc:
                    print(f"RF map animation failed: {exc}")

        def on_scatter_video(_: widgets.Button) -> None:
            with animation_output:
                clear_output()
                ok, data_dlc, merged = _require_merge_and_homography()
                if not ok:
                    return
                if not hasattr(PlottingPlotly, "generate_scatter_plot_animation"):
                    print("Scatter animation helper is not available in this repository version.")
                    return
                try:
                    figsize = (int(animation_fig_width.value), int(animation_fig_height.value))
                    video_bytes = PlottingPlotly.generate_scatter_plot_animation(
                        merged,
                        x_col="tf_FB2_x",
                        y_col="tf_FB2_y",
                        homography_points=data_dlc.homography_points,
                        size_col=size_dropdown.value,
                        color_col=color_dropdown.value,
                        bending=bending_toggle.value,
                        spikes=spikes_toggle.value,
                        title=plot_title_input.value,
                        xlabel=plot_xlabel_input.value,
                        ylabel=plot_ylabel_input.value,
                        cmap=plotly_cmap_dropdown.value,
                        fps=int(animation_fps_input.value),
                        figsize=figsize,
                    )
                    display(Video(data=video_bytes, embed=True))
                except Exception as exc:
                    print(f"Scatter animation failed: {exc}")

        def on_scroll_video(_: widgets.Button) -> None:
            with animation_output:
                clear_output()
                merged: MergedData | None = self.state.get("merged_data")
                video_path = self.state.get("labeled_video_path")
                if merged is None:
                    print("Merge data first.")
                    return
                if video_path is None or not Path(video_path).exists():
                    print("Provide a labeled video path in the DLC tab before generating the scrolling overlay video.")
                    return
                try:
                    video_bytes = PlottingPlotly.generate_scroll_over_video(
                        merged_data=merged,
                        columns=["Bending_ZScore", "Spike"],
                        video_path=str(video_path),
                        color_1="#1f77b4",
                        color_2="#d62728",
                        title="Scrolling Overlay Video",
                    )
                    display(Video(data=video_bytes, embed=True))
                except Exception as exc:
                    print(f"Scrolling overlay failed: {exc}")

        rf_video_button.on_click(on_rf_video)
        scatter_video_button.on_click(on_scatter_video)
        scroll_video_button.on_click(on_scroll_video)

        merge_box = widgets.VBox(
            [
                widgets.HTML("<h4>Merge DLC & neuron data</h4>"),
                widgets.HBox([max_gap_fill_slider, threshold_slider, merge_button]),
                merge_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        filter_box = widgets.VBox(
            [
                widgets.HTML("<h4>Filter preview</h4>"),
                widgets.HBox([bending_toggle, spikes_toggle, filter_button]),
                filter_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        plot_box = widgets.VBox(
            [
                widgets.HTML("<h4>Interactive plots</h4>"),
                widgets.HBox([kde_button, scatter_button]),
                widgets.HBox([plot_title_input, plot_xlabel_input, plot_ylabel_input]),
                size_dropdown,
                color_dropdown,
                plotly_cmap_dropdown,
                plotly_spikes_cmap_dropdown,
                widgets.HBox([bw_bending_input, bw_spikes_input, bw_threshold_input]),
                plot_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        animation_box = widgets.VBox(
            [
                widgets.HTML("<h4>Animations & videos</h4>"),
                widgets.HBox([rf_video_button, scatter_video_button, scroll_video_button]),
                widgets.HBox([animation_fps_input, animation_fig_width, animation_fig_height]),
                animation_output,
            ],
            layout=widgets.Layout(gap="0.4em"),
        )

        accordion = widgets.Accordion([merge_box, filter_box, plot_box, animation_box])
        accordion.set_title(0, "Merge data")
        accordion.set_title(1, "Filter preview")
        accordion.set_title(2, "Interactive plots")
        accordion.set_title(3, "Animations & videos")
        return accordion


def build_post_processing_tabs(state: MutableMapping[str, object], shim) -> widgets.Tab:
    return PostProcessingUI(state, shim).build()
