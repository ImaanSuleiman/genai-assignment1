# AI-use log (copy into the report appendix; keep it truthful)

| Tool | Used for | How the output was tested or corrected |
|---|---|---|
| Claude | Scaffolding of the data pipeline, models, Optuna/MLflow scripts, ONNX export, FastAPI/React/Docker setup, LaTeX template | CPU smoke tests on synthetic data, 13 pytest checks (corruption specs, manifests, balanced batches, bottleneck, MoE maths, style conditioning), ONNX vs PyTorch parity, API tests, browser test |
| Google Stitch | UI design | Screenshots kept as evidence |

Add: what you changed by hand, what bugs you found, which design decisions you made yourself, and any other tool you used (ChatGPT, Copilot, ...).
You can be asked to explain or modify any component in the evaluation, so read every file in src/genai at least once.
