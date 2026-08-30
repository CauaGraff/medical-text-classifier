from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "API request latency", ["endpoint"]
)
ERROR_COUNT = Counter(
    "api_errors_total", "Total API errors", ["endpoint"]
)
