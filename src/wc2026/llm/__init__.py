"""Local-LLM explanation layer (no LLM-as-predictor).

This package wraps Ollama for SHAP-to-prose match previews. The
inference layer is local + CPU-friendly (Mistral 7B Instruct, q4_K_M
quantized) so it works on the same Fly machine that runs the API,
with graceful degradation when the sidecar is unreachable.
"""
