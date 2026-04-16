### Structure
thermal_tracker/           # Core library
├── __init__.py
├── heat_diffusion.py      # 2D FTCS heat equation solver (Neumann BC, CFL-safe)
├── pattern_projection.py  # Grid / random / adaptive dot pattern strategies
├── scene.py               # Pinhole camera model, 6-DoF pose, thermal renderer
├── dot_tracker.py         # Blob detection + nearest-neighbour frame-to-frame tracking
├── pose_estimation.py     # PnP-based 6-DoF pose from 2D–3D correspondences
└── visualization.py       # Annotated frames, motion trails, video writer

demos/                     # Runnable demonstrations
├── full_pipeline_demo.py  # End-to-end: project → diffuse → track → estimate pose → video
├── heat_diffusion_demo.py # Visualises dot spreading over time
└── motion_tracking_demo.py# Camera sweep with tracked dots

tests/                     # 77 tests, all passing
├── test_heat_diffusion.py # Energy conservation, CFL, edge cases
├── test_pattern_projection.py  # Grid/random/adaptive, min separation, reproducibility
├── test_scene.py          # Camera model, rendering, diffusion integration
├── test_dot_tracker.py    # Detection, multi-dot tracking, pruning
├── test_pose_estimation.py# PnP accuracy (<0.1 m translation, <2° rotation)
├── test_visualization.py  # Overlay correctness
└── test_integration.py    # Full pipeline end-to-end



# Prompt
your code doesnt accomplish what i want. create another directory, ONLY FOR images and video generation

FOR EACH 6-DoF, do the following:
    generate 3 dots (use a global constant), then move them to the left slightly (using global rate that i can change), then erode them (to mimic dissipation), then generate three new dots. do this 15 times a second to generate a 5 second video (i also want all the images stored). by constantly shilfting old image to left of frame, eroding it, and adding new points, it gives the impression of movement to the right.)

    Now, what i explained in 1 DoF applyfor all 6DoF, and ensure proper folder strucctures. i want 75 images per DoF, which would make a 5 second video (also use global constants i can edit). 


# Output

Completed: *Run script and verify output* (2/2)

Created generate.py — a self-contained script with no imports from thermal_tracker.

**Output:** 450 PNGs + 6 MP4 videos across 6 DoF directories:

```
generation/output/
    translate_x/   (75 frames + video)  — dots slide left
    translate_y/   (75 frames + video)  — dots slide down
    translate_z/   (75 frames + video)  — dots expand from centre
    rotate_yaw/    (75 frames + video)  — dots shift left (edge-weighted)
    rotate_pitch/  (75 frames + video)  — dots shift down (edge-weighted)
    rotate_roll/   (75 frames + video)  — dots rotate around centre
```

**Global constants** you can edit at the top of the file:

| Constant | Default | Purpose |
|---|---|---|
| `N_DOTS_PER_FRAME` | 3 | Fresh dots per frame |
| `SHIFT_RATE` | 5.0 | Translation pixels/frame |
| `ROTATION_RATE` | 0.025 | Roll radians/frame |
| `ZOOM_RATE` | 0.035 | Expansion factor/frame (translate-Z) |
| `FADE_FACTOR` | 0.85 | Intensity decay/frame |
| `ERODE_FACTOR` | 0.90 | Radius shrink/frame |
| `FPS` | 15 | Frames per second |
| `DURATION_S` | 5 | Video length (seconds) |
| `N_FRAMES` | 75 | `FPS × DURATION_S` |

Per frame: shift existing dots → erode them → deposit 3 new dots → render.

Made changes.

# PDR


## PDR pipeline created — 4 files in pdr

| File | Purpose |
|------|---------|
| model.py | U-Net (7.8M params): 2-channel grayscale in, 1-channel corrected out |
| dataset.py | Synthetic training data — generates (diffused_pair, clean_target) on-the-fly using same fade/erode params as generation |
| train.py | Training script with `--smoke` flag for quick test |
| infer.py | Loads checkpoint, processes real frames, outputs corrected PNGs + MP4 video |

## Smoke test results
- **Training**: 2 epochs, 20 samples, ~33s/epoch on CPU. Loss dropped 0.44 → 0.34.
- **Inference**: 75 corrected frames + 1 MP4 video produced for `translate_x`
- **Checkpoint**: pdr_model.pt (29.7 MB)

## For Google Colab (real training)

Upload these 4 files to Colab, then run:
```bash
python train.py --epochs 50 --samples 10000 --batch 8
```
The training uses **synthetic data generated on the fly** (no need to upload frames). Once done, download pdr_model.pt and place it locally.

Then run inference — frames are already at `generation/output/{dof}/frames/`:
```bash
python pdr/infer.py                    # all 6 DoFs
python pdr/infer.py translate_x        # single DoF
```

Output goes to:
- Corrected frames: `pdr/output/{dof}/frames/frame_XXXX.png`
- Video: `pdr/output/{dof}/{dof}_corrected.mp4`

Note: with only 2 smoke-test epochs the correction is minimal. Real training (50+ epochs, 10k+ samples on Colab GPU) will produce visibly sharper dots.

Made changes.