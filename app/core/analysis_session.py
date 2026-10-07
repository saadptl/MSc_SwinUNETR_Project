from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisSession:
    """Single source of truth for the current dashboard analysis.

    The session stores inference, segmentation, XAI, and report state only.
    It never trains, edits, or overwrites a model checkpoint.
    """

    # ------------------------------------------------------------------
    # Existing classification state
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Existing classification XAI state
    # ------------------------------------------------------------------
    gradcam_generated: bool = False
    xai_paths: Dict[str, str] = field(default_factory=dict)
    xai_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Research XAI state
    # ------------------------------------------------------------------
    research_xai_available: bool = False

    # ------------------------------------------------------------------
    # Existing report state
    # ------------------------------------------------------------------
    report_generated: bool = False
    report_path: Optional[str] = None
    report_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 3D segmentation state
    # ------------------------------------------------------------------
    segmentation_result: Optional[Dict[str, Any]] = None

    segmentation_volume: Any = None
    segmentation_input_volume: Any = None

    segmentation_paths: Dict[str, Any] = field(default_factory=dict)
    segmentation_slice_indices: List[int] = field(default_factory=list)
    segmentation_slice_index: Optional[int] = None

    segmentation_ready: bool = False

    # ------------------------------------------------------------------
    # Clinical Level Verification & Anatomical Tracing state
    # ------------------------------------------------------------------
    clinical_level_verification: bool = False
    anatomical_tracing_confirmed: bool = False
    c2_scout_status: str = "Local Lumbar FOV (C2 Scout External)"
    level_verification_notes: str = ""

    # ------------------------------------------------------------------
    # Clinical Spine Measurements & Morphometrics state
    # ------------------------------------------------------------------
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    level_morphometrics: Dict[str, Any] = field(default_factory=dict)
    detected_lesion_metrics: Dict[str, Any] = field(default_factory=dict)
    multi_level_scorecard: List[Dict[str, Any]] = field(default_factory=list)
    segmentation_summary: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def has_prediction(self) -> bool:
        return bool(self.predicted_class)

    @property
    def has_xai(self) -> bool:
        """Return True when current-session or research XAI exists."""

        current_session_xai = bool(
            self.gradcam_generated and self.xai_paths
        )

        project_root = Path(__file__).resolve().parents[2]
        xai_root = project_root / "outputs" / "segmentation"

        research_dirs = [
            xai_root / "rsna_part45_3d_xai",
            xai_root / "rsna_part46_point_targeted_3d_xai",
            xai_root / "rsna_part46b_multi_point_3d_xai",
            xai_root / "rsna_part46c_xai_summary",
            xai_root / "rsna_part46d_xai_visual_inspection",
        ]

        research_xai = any(path.exists() for path in research_dirs)

        return bool(
            current_session_xai
            or self.research_xai_available
            or research_xai
        )

    @property
    def has_report(self) -> bool:
        return bool(self.report_generated and self.report_path)

    @property
    def has_segmentation(self) -> bool:
        """Return True when a completed 3D segmentation is available."""

        return bool(
            self.segmentation_ready
            and self.segmentation_result
            and self.segmentation_volume is not None
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all analysis-dependent state while preserving the session."""

        # Existing classification state
        self.study_id = None
        self.series_id = None
        self.selected_files = []
        self.middle_slice = None
        self.tensor_shape = None
        self.device = None
        self.model_name = "Swin Transformer"
        self.model_checkpoint = None

        self.predicted_class = None
        self.predicted_class_id = None
        self.confidence = None
        self.probabilities = {}

        self.series_metadata = {}
        self.selected_metadata = {}
        self.processed_images = []

        # Existing XAI state
        self.gradcam_generated = False
        self.xai_paths = {}
        self.xai_result = None

        # Research XAI state
        self.research_xai_available = False

        # Existing report state
        self.report_generated = False
        self.report_path = None
        self.report_summary = None

        # Segmentation state
        self.segmentation_result = None
        self.segmentation_volume = None
        self.segmentation_input_volume = None
        self.segmentation_paths = {}
        self.segmentation_slice_indices = []
        self.segmentation_slice_index = None
        self.segmentation_ready = False

        # Measurements
        self.measurements = []
        self.level_morphometrics = {}
        self.detected_lesion_metrics = {}