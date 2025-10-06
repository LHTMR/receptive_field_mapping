from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

import ipywidgets as widgets
from IPython.display import Markdown, clear_output, display

from src.train_predict import dlc_utils


class PredictionUI:
    def __init__(self, state: MutableMapping[str, object], shim) -> None:
        self.state = state
        self.shim = shim

    def build(self) -> widgets.Tab:
        style = {"description_width": "160px"}
        full_width = widgets.Layout(width="100%")

        project_input = widgets.Text(
            description="Project path:",
            placeholder="Full path to your DeepLabCut project folder",
            style=style,
            layout=full_width,
        )
        load_button = widgets.Button(description="Load project", icon="folder-open")
        load_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        video_upload = widgets.FileUpload(accept=".mp4,.avi,.mov", multiple=False, description="Upload video")
        preprocess_button = widgets.Button(description="Preprocess video", icon="cogs")
        preprocess_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        prediction_button = widgets.Button(description="Run prediction & create labeled video", icon="film")
        prediction_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_load_clicked(_: widgets.Button) -> None:
            with load_output:
                clear_output()
                path_text = project_input.value.strip()
                if not path_text:
                    print("Enter the DeepLabCut project path to continue.")
                    return
                project_path = Path(path_text.strip("\"").strip("'"))
                if not project_path.exists():
                    print(f"Path not found: {project_path}")
                    return

                config_path = project_path / "config.yaml"
                if not config_path.exists():
                    print(f"config.yaml not found in {project_path}")
                    return

                videos_dir = project_path / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                train_folders = sorted(project_path.glob("dlc-models-pytorch/iteration-0/*/train"))
                train_folder = train_folders[0] if train_folders else None

                self.state["project_path"] = project_path
                self.state["config_path"] = config_path
                self.state["videos_dir"] = videos_dir
                self.state["training_folder"] = train_folder

                self.shim.set_output(load_output)
                try:
                    dlc_utils.init_project(str(config_path), str(project_path))
                    if train_folder:
                        dlc_utils.clean_snapshots(str(train_folder))
                finally:
                    self.shim.set_output(None)

                summary = [
                    f"Project loaded: {project_path}",
                    f"Videos directory: {videos_dir}",
                    f"Training folder: {train_folder if train_folder else 'not found (will be created after training)'}",
                ]
                for line in summary:
                    print(line)

        load_button.on_click(on_load_clicked)

        def on_preprocess_clicked(_: widgets.Button) -> None:
            with preprocess_output:
                clear_output()
                project_path = self.state.get("project_path")
                videos_dir = self.state.get("videos_dir")
                if project_path is None or videos_dir is None:
                    print("Load a DeepLabCut project before preprocessing a video.")
                    return
                if not video_upload.value:
                    print("Upload a video file to preprocess.")
                    return

                upload = next(iter(video_upload.value.values()))
                original_name = upload["metadata"]["name"]
                temp_input_path = Path(videos_dir) / original_name
                with open(temp_input_path, "wb") as fh:
                    fh.write(upload["content"])

                processed_video_name = f"processed_{Path(original_name).stem}.mp4"
                processed_video_path = Path(videos_dir) / processed_video_name

                print("Preprocessing video... this may take a while.")
                dlc_utils.preprocess_video(str(temp_input_path), str(processed_video_path))
                try:
                    temp_input_path.unlink()
                except FileNotFoundError:
                    pass

                self.state["processed_video_path"] = processed_video_path
                self.state["processed_video_name"] = processed_video_name
                print(f"Video preprocessed and saved to {processed_video_path}")

        preprocess_button.on_click(on_preprocess_clicked)

        def on_prediction_clicked(_: widgets.Button) -> None:
            with prediction_output:
                clear_output()
                config_path = self.state.get("config_path")
                processed_video_path = self.state.get("processed_video_path")
                videos_dir = self.state.get("videos_dir")

                if None in (config_path, processed_video_path, videos_dir):
                    print("Load the project and preprocess a video before running predictions.")
                    return

                self.shim.set_output(prediction_output)
                try:
                    dlc_utils.predict_and_show_labeled_video(
                        str(config_path),
                        str(processed_video_path),
                        str(videos_dir),
                    )
                    if "h5_path" in self.state:
                        print(f"Latest DLC data stored at {self.state['h5_path']}")
                except Exception as exc:
                    print(f"Prediction failed: {exc}")
                finally:
                    self.shim.set_output(None)

        prediction_button.on_click(on_prediction_clicked)

        tab1_box = widgets.VBox(
            [
                widgets.HTML(
                    "<h3>Create Labeled Video</h3><p>Load a project, upload a video, preprocess it and generate predictions.</p>"
                ),
                project_input,
                load_button,
                load_output,
                widgets.HTML("<hr>"),
                video_upload,
                preprocess_button,
                preprocess_output,
                widgets.HTML("<hr>"),
                prediction_button,
                prediction_output,
            ],
            layout=widgets.Layout(gap="0.6em"),
        )

        num_frames_slider = widgets.IntSlider(value=10, min=5, max=50, step=5, description="Frames to extract", style=style)
        extract_button = widgets.Button(description="Extract frames & launch Napari", icon="image")
        extract_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        epochs_slider = widgets.IntSlider(value=25, min=5, max=50, step=5, description="Pose epochs", style=style)
        detector_epochs_slider = widgets.IntSlider(value=50, min=5, max=100, step=5, description="Detector epochs", style=style)
        retrain_button = widgets.Button(description="Retrain model", icon="redo")
        retrain_output = widgets.Output(layout={"border": "1px solid #ddd", "padding": "0.2em"})

        def on_extract_clicked(_: widgets.Button) -> None:
            with extract_output:
                clear_output()
                config_path = self.state.get("config_path")
                processed_video_path = self.state.get("processed_video_path")
                if None in (config_path, processed_video_path):
                    print("Load a project and preprocess a video before extracting frames.")
                    return
                self.shim.set_output(extract_output)
                try:
                    dlc_utils.update_num_frames2pick(str(config_path), int(num_frames_slider.value))
                    dlc_utils.run_labeling(str(config_path), str(processed_video_path))
                except Exception as exc:
                    print(f"Frame extraction / labeling failed: {exc}")
                finally:
                    self.shim.set_output(None)

        extract_button.on_click(on_extract_clicked)

        def on_retrain_clicked(_: widgets.Button) -> None:
            with retrain_output:
                clear_output()
                project_path = self.state.get("project_path")
                config_path = self.state.get("config_path")
                processed_video_path = self.state.get("processed_video_path")
                videos_dir = self.state.get("videos_dir")
                train_folder = self.state.get("training_folder")

                if None in (project_path, config_path, processed_video_path, videos_dir):
                    print("Load project, preprocess a video, and extract frames before retraining.")
                    return

                if train_folder is None or not Path(train_folder).exists():
                    train_folders = sorted(Path(project_path).glob("dlc-models-pytorch/iteration-0/*/train"))
                    train_folder = train_folders[0] if train_folders else None
                    self.state["training_folder"] = train_folder

                if train_folder is None:
                    print("Could not find a training folder. Run DeepLabCut training at least once to create it.")
                    return

                if not dlc_utils.is_labeling_done(str(project_path)):
                    print("No labeled data found. Label and save frames in Napari before retraining.")
                    return

                self.shim.set_output(retrain_output)
                try:
                    dlc_utils.add_video_to_config(str(config_path), str(processed_video_path))
                    dlc_utils.clean_snapshots(str(train_folder))
                    dlc_utils.delete_prev_pred(str(videos_dir))
                    dlc_utils.clear_training_datasets(str(project_path))

                    dlc_utils.run_retraining(
                        str(config_path),
                        str(train_folder),
                        int(epochs_slider.value),
                        int(detector_epochs_slider.value),
                    )

                    pose_fig = dlc_utils.show_pose_training_loss(str(train_folder))
                    if pose_fig:
                        display(pose_fig)
                    detector_fig = dlc_utils.show_detector_training_loss(str(train_folder))
                    if detector_fig:
                        display(detector_fig)

                    dlc_utils.predict_and_show_labeled_video(
                        str(config_path),
                        str(processed_video_path),
                        str(videos_dir),
                    )
                except Exception as exc:
                    print(f"Retraining failed: {exc}")
                finally:
                    self.shim.set_output(None)

        retrain_button.on_click(on_retrain_clicked)

        tab2_box = widgets.VBox(
            [
                widgets.HTML(
                    "<h3>Labeling / Retraining</h3><p>Extract frames for Napari labeling, then retrain the model once labels are saved.</p>"
                ),
                num_frames_slider,
                extract_button,
                extract_output,
                widgets.HTML("<hr>"),
                epochs_slider,
                detector_epochs_slider,
                retrain_button,
                retrain_output,
            ],
            layout=widgets.Layout(gap="0.6em"),
        )

        tabs = widgets.Tab(children=[tab1_box, tab2_box])
        tabs.set_title(0, "Create Labeled Video")
        tabs.set_title(1, "Labeling / Retraining")
        return tabs


def build_prediction_tabs(state: MutableMapping[str, object], shim) -> widgets.Tab:
    return PredictionUI(state, shim).build()
