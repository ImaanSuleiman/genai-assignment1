# Demo video script (5 to 7 minutes, upload to YouTube, link only in the report)

Record the screen with OBS or the Windows Game Bar. Do NOT upload the video to Google Classroom.

| Time | What to show |
|---|---|
| 0:00 | One sentence: the four tasks and the stack. Show the repo and README. |
| 0:30 | Startup from scratch in a terminal: `git clone`, `python scripts/download_models.py`, `docker compose up --build`. Open http://localhost:8080 in the browser (no VS Code). |
| 1:15 | Task 1 Universal Restoration: upload an image, apply salt-and-pepper (high), blur, occlusion; show input, restored output, error map, settings, inference time. Upload an already corrupted image. Download the result. |
| 2:30 | Task 2 Hard-Routed: show the four classifier probabilities, predicted corruption, selected expert, identity bypass on a clean image. Show one misrouted case. |
| 3:30 | Task 3 Soft MoE: show the four routing weights, the strongest expert, then add a second corruption to show the weights spreading. |
| 4:30 | Task 4 Face-to-Sketch: upload a photo, then capture one with the webcam; generate Style 1, 2, 3; download a sketch. |
| 5:30 | Experiment tracking: `mlflow ui`, open the Optuna study runs, a loss curve, the logged sample images. |
| 6:30 | Close: where the report, repo and models are. |

Required items to be visible: image uploading, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, experiment-tracking records.
