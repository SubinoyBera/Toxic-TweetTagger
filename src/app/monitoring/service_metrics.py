from prometheus_client import Counter, Histogram

# Requests successfully served
PREDICTION_REQUEST_SUCCESS = Counter(
    "predict_requests_success_total",
    "Total successful prediction responses"
)

# Requests failed
PREDICTION_REQUEST_FAILED = Counter(
    "predict_requests_failed_total",
    "Total failed prediction requests"
)

# Prediction class distribution (hate / non-hate)
PREDICTION_CLASS = Counter(
    "prediction_class_total",
    "Count of predicted classes",
    ["class_label"]         # label dimension
)

# Prediction label confidence
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence distribution by predicted class",
    ["class_label"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Inference responce time
INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds",
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
)

EXPLAINER_REQUEST_SUCCESS = Counter(
    "explainer_requests_success_total",
    "Total successful explain requests"
)

EXPLAINER_REQUEST_FAILED = Counter(
    "explainer_requests_failed_total",
    "Total failed explain requests"
)

# Feedback counter
FEEDBACK_REQUEST_SUCCESS = Counter(
    "feedback_subissions_success_total",
    "Total successful feedback submissions"
)

FEEDBACK_VALIDATION = Counter(
    "feedback_validation_total",
    "Feedback validation result",
    ["result"]
)

FEEDBACK_REQUEST_FAILED = Counter(
    "feedback_submissions_failed_total",
    "Total failed feedback submissions"
)