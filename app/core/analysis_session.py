from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisSession:
    """Single source of truth for the current dashboard analysis.

    This object contains inference/XAI/report state only. It never
    trains, edits, or overwrites a model checkpoint.
    """

    study_id: Optional[str] = None
    series_id: Optional[str] = None
    selected_files: List[str] = field(default_factory=list)
    middle_slice: Optional[str] = None
    tensor_shape: Optional[str] = None
    device: Optional[str] = None
    model_name: str = "Swin Transformer"
    model_checkpoint: Optional[str] = None
    predicted_class: Optional[str] = None
    predicted_class_id: Optional[int] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    series_metadata: Dict[str, Any] = field(default_factory=dict)
    selected_metadata: Dict[str, Any] = field(default_factory=dict)
    processed_images: List[Any] = field(default_factory=list)
    gradcam_generated: bool = False
    xai_paths: Dict[str, str] = field(default_factory=dict)
    xai_result: Optional[Dict[str, Any]] = None
    report_generated: bool = False
    report_path: Optional[str] = None
    report_summary: Optional[Dict[str, Any]] = None

    @property
    def has_prediction(self) -> bool:
        return bool(self.predicted_class)

    @property
    def has_xai(self) -> bool:
        return bool(self.gradcam_generated and self.xai_paths)

    @property
    def has_report(self) -> bool:
        return bool(self.report_generated and self.report_path)

    def reset(self) -> None:
        """Clear analysis-dependent state while preserving the session object."""
        self.study_id = None
        self.series_id = None
        self.selected_files = []
        self.middle_slice = None
        self.tensor_shape = None
        self.device = None
        self.predicted_class = None
        self.predicted_class_id = None
        self.confidence = None
        self.probabilities = {}
        self.series_metadata = {}
        self.selected_metadata = {}
        self.processed_images = []
        self.gradcam_generated = False
        self.xai_paths = {}
        self.xai_result = None
        self.report_generated = False
        self.report_path = None
        self.report_summary = None
