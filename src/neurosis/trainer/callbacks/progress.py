from typing import Any, Optional

from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.style import Style


class NeurosisProgressTheme(RichProgressBarTheme):
    """
    Styles to associate to different progress bar components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html

    """

    # 8e53e0
    description: str | Style = "white"
    progress_bar: str | Style = "#711EE3"
    progress_bar_finished: str | Style = "#6206E0"
    progress_bar_pulse: str | Style = "#711EE3"
    batch_progress: str | Style = "white"
    time: str | Style = "grey70"
    processing_speed: str | Style = "sky_blue1"
    metrics: str | Style = "bright_white"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"


class NeurosisProgressBar(RichProgressBar):
    """Subclass of lightning.pytorch.callbacks.progress.RichProgressBar with custom default theme."""

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = NeurosisProgressTheme(),
        console_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(refresh_rate, leave, theme, console_kwargs)
