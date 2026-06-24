"""Shared constants for tracker selection and MS-DC-ELT reporting."""

TRACKER_CHOICES = (
    "bytetrack",
    "ocsort",
    "botsort",
    "dist_tracker",
    "official_ocsort",
    "official_botsort",
    "msdc_elt",
)
OPTIONAL_TRACKER_CHOICES = ("", *TRACKER_CHOICES)
BASELINE_TRACKER_CHOICES = TRACKER_CHOICES[:-1]
OPTIONAL_BASELINE_TRACKER_CHOICES = ("", *BASELINE_TRACKER_CHOICES)
PAPER_TRACKER_CHOICES = ("bytetrack", "ocsort", "botsort", "msdc_elt")
DATASET_EXPORT_TRACKER_CHOICES = ("bytetrack", "botsort", "ocsort", "msdc_elt")

FORMAL_FRAME_LIMIT = 5400
FORMAL_MSDC_VARIANT = "msdc_v3"
MAIN_MSDC_TRACKER = FORMAL_MSDC_VARIANT
SUMMARY_MAIN_TRACKERS = {"bytetrack", "ocsort", "botsort", "msdc_elt", MAIN_MSDC_TRACKER}

METRIC_FIELDS = (
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDSW",
    "FP",
    "FN",
    "IDTP",
    "IDFP",
    "IDFN",
)

SPEED_FIELDS = (
    "run_id",
    "commit_hash",
    "video_path",
    "seq_name",
    "tracker",
    "method",
    "resolution",
    "requested_frames",
    "processed_frames",
    "total_time_s",
    "mean_fps",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_memory_mb",
    "detector_calls_total",
    "detector_calls_high_det",
    "detector_calls_low_det",
    "detector_calls_tracker_update",
    "mean_read_decode_ms",
    "mean_low_detection_ms",
    "mean_high_split_ms",
    "mean_low_filter_budget_ms",
    "mean_observation_build_ms",
    "mean_evidence_update_ms",
    "mean_output_nms_ms",
    "mean_render_write_ms",
    "mean_read_ms",
    "mean_high_det_ms",
    "mean_low_det_ms",
    "mean_tracker_ms",
    "mean_msdc_low_filter_ms",
    "mean_msdc_observation_build_ms",
    "mean_msdc_evidence_update_ms",
    "mean_msdc_output_ms",
    "mean_msdc_debug_ms",
    "mean_msdc_total_update_ms",
    "mean_render_ms",
    "mean_write_ms",
)
SPEED_OUTPUT_FIELDS = (*SPEED_FIELDS, "status", "failure")

METHOD_LABELS = {
    "bytetrack": "FFCA-YOLO + ByteTrack",
    "ocsort": "FFCA-YOLO + OC-SORT",
    "botsort": "FFCA-YOLO + BoT-SORT",
    "msdc_elt": "FFCA-YOLO + MS-DC-ELT",
    FORMAL_MSDC_VARIANT: "FFCA-YOLO + MS-DC-ELT v3",
    "v3_candidate_topk_no_roi_no_motion": "FFCA-YOLO + MS-DC-ELT v3",
}
