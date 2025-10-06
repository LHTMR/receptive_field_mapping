from __future__ import annotations

from typing import Any, MutableMapping, Optional

import ipywidgets as widgets
from pathlib import Path

from IPython.display import JSON, Markdown, Image, Video, display


def attach_streamlit_shim(state: Optional[MutableMapping[str, Any]] = None) -> tuple[MutableMapping[str, Any], "StreamlitNotebookShim"]:
    from src.train_predict import dlc_utils

    notebook_state: MutableMapping[str, Any] = state or {}
    shim = StreamlitNotebookShim(notebook_state)
    dlc_utils.st = shim
    return notebook_state, shim


class StreamlitNotebookShim:
    def __init__(self, state: MutableMapping[str, Any]):
        self.session_state = state
        self._output: Optional[widgets.Output] = None

    def set_output(self, output: Optional[widgets.Output]) -> None:
        self._output = output

    def _display(self, obj: Any) -> None:
        if self._output is not None:
            with self._output:
                display(obj)
        else:
            display(obj)

    def _display_markdown(self, text: str) -> None:
        self._display(Markdown(text))

    def success(self, text: str) -> None:
        self._display_markdown(f"✅ {text}")

    def info(self, text: str) -> None:
        self._display_markdown(f"ℹ️ {text}")

    def warning(self, text: str) -> None:
        self._display_markdown(f"⚠️ {text}")

    def error(self, text: str) -> None:
        self._display_markdown(f"❌ {text}")

    def write(self, obj: Any) -> None:
        self._display(obj)

    def markdown(self, text: str) -> None:
        self._display_markdown(text)

    def title(self, text: str) -> None:
        self._display_markdown(f"# {text}")

    def header(self, text: str) -> None:
        self._display_markdown(f"## {text}")

    def subheader(self, text: str) -> None:
        self._display_markdown(f"### {text}")

    def image(self, image: Any, caption: Optional[str] = None, width: Optional[int] = None) -> None:
        if isinstance(image, (str, Path)):
            img = Image(filename=str(image), width=width)
        else:
            img = Image(data=image, width=width)
        self._display(img)
        if caption:
            self._display_markdown(f"*{caption}*")

    def video(self, data: Any, format: str = "mp4") -> None:
        if isinstance(data, (bytes, bytearray)):
            vid = Video(data=data, embed=True)
        else:
            vid = Video(filename=str(data))
        self._display(vid)

    def json(self, obj: Any) -> None:
        self._display(JSON(obj))

    def plotly_chart(self, fig: Any, use_container_width: bool = True) -> None:
        fig.show()

    def pyplot(self, fig: Any, use_container_width: bool = True) -> None:
        self._display(fig)

    def stop(self) -> None:
        raise RuntimeError("Execution stopped by StreamlitNotebookShim.stop().")
