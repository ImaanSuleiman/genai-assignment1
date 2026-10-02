# Google Stitch design brief (paste into https://stitch.withgoogle.com)

Create a responsive web app design called "Restoration & Sketch Studio". Light theme, indigo accent (#4F46E5), white rounded cards on a very light slate background, soft shadows, clean sans-serif type.

Layout: a left sidebar (collapses to a horizontal tab bar on mobile) with the app name, the subtitle "Generative AI, Assignment 1, four models, one app", four navigation items (Task 1 Universal Restoration, Task 2 Hard-Routed Restoration, Task 3 Soft Mixture-of-Experts, Task 4 Face-to-Sketch Generator) and a small status indicator at the bottom ("backend ready, all models found").

Restoration screens (Tasks 1 to 3): page title + one-sentence description. Two columns. Left card "Input": drag-and-drop upload area, a row of 8 clean sample thumbnails, a segmented control for corruption (Already corrupted / Salt-and-pepper / Gaussian blur / Occlusion / None), a segmented control for severity (Low / Medium / High), a seed field, a primary button "Restore image". Right: a "Result" card with four image panels (Clean reference, Input corrupted, Restored output with a Download PNG link, Absolute error map), then stat tiles (inference time, PSNR input to output, image source). Task 2 adds a "Classifier probabilities" card with four horizontal bars and the selected expert. Task 3 adds a "Routing weights" card with four horizontal bars (identity, salt-and-pepper expert, blur expert, occlusion expert) with the strongest highlighted, and an optional "second corruption" control.

Face-to-sketch screen (Task 4): left card "Photo" with upload area, a "Use webcam" button with capture, a segmented control Style 1 / Style 2 / Style 3, primary button "Generate sketch". Right card "Result": original photograph and generated sketch side by side with a download link, tiles for inference time and model name.

Export the design (screenshots of each screen, plus the Stitch project link) and keep the screenshots: the assignment requires evidence of the original Stitch design in the report (report/figures/stitch_design.png).
