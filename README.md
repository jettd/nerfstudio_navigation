# nerfstudio_navigation

This is a custom fork of [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (base: v1.1.5), modified to work with the harvest drone imagery pipeline. **It is not interchangeable with vanilla nerfstudio with the rails web app** — the harvest viewer and job_gateway SLURM scripts depend on additions made in this fork.

The original upstream README is preserved as `README_upstream.md`.

---

## What was added

- **Flask telemetry server** — embedded in the viewer process, exposes:
  - `GET /telemetry` — current camera position/orientation
  - `POST /teleport` — move viewer to a specified position
  - `GET /scene_bounds` — bounding box of the loaded scene
  - `GET /nearest_view` — closest training camera to current position
  - `GET /nearest_views` — N closest training cameras
- **Model swapping** — swap between loaded models without restarting the viewer
- **Minimap** — spatial overview of training camera positions

These additions are what allow the harvest Rails UI to provide interactive session controls (teleportation, nearest-view download, scene bounds display).

---

## Installation

Requires CUDA 13.2 and PyTorch 2.1.2 (built against CUDA 11.8). Verify GPU drivers before proceeding.

This repo must be installed as an **editable pip install** into the `harvest_nerfstudio_editable` conda environment. That name is hardcoded in job_gateway SLURM scripts — do not change it.

```bash
conda create -n harvest_nerfstudio_editable python=3.10
conda activate harvest_nerfstudio_editable
cd /path/to/nerfstudio_navigation
pip install -e .
```

`pip install -e .` triggers CUDA extension compilation (gsplat). This will fail if CUDA drivers are not correctly installed.

Do **not** `pip install nerfstudio` into this environment — the vanilla package will overwrite the fork's entry points and break the Flask telemetry server.

---

## Usage

This repo is not invoked directly. job_gateway generates SLURM batch scripts that activate the `harvest_nerfstudio_editable` conda env and call `ns-viewer`, `ns-train`, `ns-render`, and `ns-export`. The Flask server starts automatically when the viewer launches.

See [DEPLOYMENT.md](../DEPLOYMENT.md) for full setup instructions.
