# Thermal 6-DoF Pose Estimation Pipeline

A complete end-to-end pipeline for generating synthetic 6-degree-of-freedom thermal dot sequences, tracking them across frames, and estimating camera pose using SLAM-based methods with persistent point tracking.

## Overview

This project implements a full thermal tracking and pose estimation system with four interconnected modules:

1. **Generation** (`generation/`) - Synthetic thermal dot video generation
2. **PDR** (`pdr/`) - Thermal image diffusion reversal using PDRNet
3. **Tracking** (`track/`) - Persistent centroid-based blob tracking with CSV export
4. **Pose Estimation** (`pose/`) - SLAM-based 6-DoF pose recovery from tracked points

The pipeline produces **75-frame sequences** for 6 different degrees of freedom (3 translations + 3 rotations), with ground truth poses available for validation.

## Architecture

```
Synthetic Thermal Frames (generation/)
    ↓
Thermal Diffusion Reversal (pdr/)
    ↓
Persistent Blob Tracking (track/)
    ↓ (CSV: frame, point_id, x, y)
    ↓
SLAM Pose Estimation (pose/)
    ↓
Benchmark & Error Metrics
```

### Key Features

- **Persistent Point Tracking**: Each thermal dot is assigned a unique ID that persists across frames, enabling triangulation and pose constraints
- **Two-Stage Bootstrap**: Essential matrix as primary method, homography decomposition as fallback for sparse/planar geometry
- **Multi-Frame Triangulation**: Reconstructs 3D world points from multiple frame observations with reprojection error validation
- **Per-Frame PnP Solving**: Estimates camera pose for each frame after bootstrap and triangulation
- **Ground Truth Validation**: All 6 sequences have known ground truth rotation/translation for error analysis

---

## Module Breakdown

### 1. Generation Module (`generation/generate.py`)

**Purpose**: Generates synthetic 6-DoF thermal dot sequences for controlled testing

**Key Classes**:
- `Dot`: Represents a thermal dot with position and fade properties
- `Shift` functions: Define transformations for each degree of freedom

**Parameters**:
- `SHIFT_RATE = 5.0`: Pixels per frame for translation
- `ZOOM_RATE = 1.02`: Zoom multiplier for translate_z
- `ROTATION_RATE = 0.05`: Radians per frame for rotations
- `IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480`
- `N_FRAMES = 75`: Frames per sequence
- `FPS = 15`: Video frame rate

**Output**:
- `generation/output/{dof}/frame_XXXXX.png` - Individual frame files
- `generation/output/{dof}/{dof}.mp4` - MP4 video

**Usage**:
```bash
cd generation
uv run ./generate.py
```

**Output Example**:
```
translate_x: 75 frames + video -> generation/output/translate_x/
translate_y: 75 frames + video -> generation/output/translate_y/
translate_z: 75 frames + video -> generation/output/translate_z/
rotate_yaw: 75 frames + video -> generation/output/rotate_yaw/
rotate_pitch: 75 frames + video -> generation/output/rotate_pitch/
rotate_roll: 75 frames + video -> generation/output/rotate_roll/
```

---

### 2. PDR Module (`pdr/`)

**Purpose**: Reverses thermal diffusion effects using a U-Net based PDRNet model

**Key Components**:
- `PDRNet`: U-Net architecture with encoder-decoder structure for thermal image enhancement
- `_ConvBlock`: Convolutional blocks with batch normalization and ReLU
- `model.py`: Model definition and architecture
- `train.py`: Training script for PDRNet
- `infer.py`: Inference/enhancement script

**Model Architecture**:
- Input: Diffused thermal images [B, 2, 640, 480]
- Output: Enhanced thermal images [B, 1, 640, 480]
- 4 encoder layers + 4 decoder layers with skip connections
- Sigmoid activation for output range [0, 1]

**Key Parameters** (in `model.py`):
- `BASE_CHANNELS = 16`: Initial channel count
- `NUM_LAYERS = 4`: Depth of U-Net

**Testing**: `tests/test_pdr.py` - 26 test cases covering model creation, forward passes, gradients, and odd dimensions

---

### 3. Tracking Module (`track/track_blob.py`)

**Approach**: Various tracking modules were evaluated, including optical flow, frame differencing, and feature-based methods, and persistent blob centroid tracking was chosen for robustness on synthetic thermal dot sequences.

**Purpose**: Persistent centroid-based tracking of thermal blobs across frames

**Key Classes**:
- `GroundTruth`: Ground truth motion parameters for each DoF
- `FrameResult`: Per-frame tracking results
- `TrackingResult`: Complete sequence tracking data
- `TrackedPoint`: Individual tracked point with persistent ID

**Key Functions**:
- `thermal_mask()`: Binary mask of bright pixels above threshold
- `to_gray()`: Convert BGR frames to grayscale
- `find_blobs()`: Identify blob centroids in thermal mask
- `match_blobs()`: Assign persistent IDs to blobs across frames

**Parameters** (in `common.py`):
- `BRIGHTNESS_THRESHOLD = 200`: Pixel intensity threshold for blob detection
- `MIN_BLOB_AREA = 10`: Minimum pixels for valid blob
- `N_FRAMES = 75`: Frames per sequence
- `IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480`

**Output**:
- `track/output/blob/{dof}/{dof}_blob_tracks.csv` - CSV with columns: `frame, point_id, x, y`
- One row per tracked point per frame
- Persistent point IDs enable cross-frame matching

**CSV Format Example**:
```
frame,point_id,x,y
0,0,320.5,240.2
0,1,350.1,260.8
0,2,280.3,220.5
1,0,325.2,240.5
1,1,355.0,261.2
1,2,285.1,220.8
```

**Usage**:
```bash
cd track
uv run ./track_blob.py translate_x
```

**Testing**: `tests/test_track.py` - 22 test cases covering thermal masking, grayscale conversion, and dataclass operations

---

### 4. Pose Estimation Module (`pose/pose_slam.py`)

**Purpose**: SLAM-based 6-DoF pose estimation from persistent tracked points

**Key Functions**:
- `bootstrap_pose()`: Estimate initial camera pose from first two frames
  - Primary: Essential matrix + cv2.recoverPose
  - Fallback: Homography decomposition for sparse/planar geometry
  
- `triangulate_point()`: Recover 3D world position of tracked point
  - Uses cv2.triangulatePoints
  - Validates with reprojection error (threshold: 2.5 pixels)
  
- `solve_pose_from_map()`: Estimate camera pose for new frame
  - Uses cv2.solvePnP with EPNP method
  - Requires 4+ point-to-3D correspondences

- `run_dof()`: Main pipeline orchestrator
  1. Load tracking CSV
  2. Bootstrap pose from frames 0-1
  3. Triangulate all observed points into 3D
  4. For each frame: solve PnP for camera pose
  5. Return pose results and error metrics

**Parameters** (in `common.py`):
- `BOOTSTRAP_MIN_TRACKS = 4`: Minimum common points for bootstrap
- `MIN_POINTS_FOR_PNP = 4`: Minimum 3D-2D correspondences
- `MAX_REPROJ_ERROR = 2.5`: Max reprojection error for valid triangulation

**Camera Intrinsics** (in `common.py`):
```python
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float32)
# Focal length: 500 pixels
# Principal point: (320, 240) = image center
```

**Data Structures**:
- `PoseGT`: Ground truth rotation vector and translation vector
- `FramePoseResult`: Per-frame pose with validity flag
- `PoseResult`: Sequence-level statistics

**Output**:
```
FramePoseResult(
    rvec=array([...]),    # Rotation vector (3,)
    tvec=array([...]),    # Translation vector (3,)
    valid=True/False
)
```

**Usage**:
```bash
cd pose
uv run ./benchmark.py translate_x  # Estimates pose for single DoF
# or
python pose_slam.py                # Runs all 6 DoF
```

**Testing**: `tests/test_pose.py` - 15 test cases covering error metrics, CSV parsing, and data structures

---

## Data Flow Pipeline

### Step 1: Generate Synthetic Frames
```bash
cd generation
uv run ./generate.py
```
Output: 75 PNG frames per DoF in `generation/output/{dof}/frame_*.png`

### Step 2: Track Thermal Blobs
```bash
cd track
uv run ./track_blob.py translate_x
```
Output: CSV file at `track/output/blob/translate_x/translate_x_blob_tracks.csv`

### Step 3: Estimate Pose (Benchmark)
```bash
cd pose
uv run ./benchmark.py translate_x
```
Output: Console table with rotation/translation error metrics

**Complete Example Output**:
```
════════════════════════════════════════════════════════════════
           6-DoF SLAM Pose Estimation Results
════════════════════════════════════════════════════════════════
DoF              Valid Frames   Mean Rot Err (°)   Mean Trans Err (°)
────────────────────────────────────────────────────────────────
translate_x          75/75              5.2                3.1
translate_y          75/75              4.8                2.9
translate_z          75/75              6.1                4.2
rotate_yaw           75/75              3.5                2.1
rotate_pitch         75/75              4.0                2.5
rotate_roll          75/75              3.7                2.3
════════════════════════════════════════════════════════════════
```

---

## Configuration & Tuning

### Generation Parameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `SHIFT_RATE` | 5.0 | 1-20 | Translation speed (px/frame) |
| `ZOOM_RATE` | 1.02 | 1.01-1.10 | Z-axis zoom factor |
| `ROTATION_RATE` | 0.05 | 0.01-0.2 | Rotation speed (rad/frame) |
| `FADE_FACTOR` | 0.85 | 0-1 | Dot brightness decay |
| `ERODE_FACTOR` | 0.9 | 0-1 | Morphological erosion amount |

### Tracking Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `BRIGHTNESS_THRESHOLD` | 200 | Pixel intensity cutoff for blob detection |
| `MIN_BLOB_AREA` | 10 | Minimum connected pixels for valid blob |
| `TRACKING_DISTANCE` | 100 | Max euclidean distance for blob matching |

### Pose Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `BOOTSTRAP_MIN_TRACKS` | 4 | Minimum common points for initial pose |
| `MIN_POINTS_FOR_PNP` | 4 | Minimum 3D-2D correspondences per frame |
| `MAX_REPROJ_ERROR` | 2.5 | Max reprojection error (pixels) for valid 3D point |

---

## Error Metrics

### Rotation Error (degrees)
- **Metric**: Geodesic distance between estimated and ground truth rotation matrices
- **Formula**: `arccos((trace(R_est^T * R_gt) - 1) / 2)` in radians, converted to degrees
- **Interpretation**:
  - 0°: Perfect rotation estimate
  - <5°: Excellent alignment
  - 5-10°: Good alignment
  - >10°: Poor alignment

### Translation Direction Error (degrees)
- **Metric**: Angular error between estimated and ground truth translation directions
- **Formula**: `arccos((t_est · t_gt) / (||t_est|| ||t_gt||))` in radians, converted to degrees
- **Interpretation**:
  - 0°: Translation in correct direction
  - <5°: Direction well-estimated
  - 5-10°: Significant direction error
  - >90°: Opposite direction (failure)

### Valid Frames Percentage
- **Metric**: Frames with successful PnP pose estimation
- **Target**: 100% (75/75 frames)
- **Failure Causes**:
  - Insufficient 3D-2D point correspondences
  - Degenerate geometry (collinear points)
  - Tracking failures

---

## Installation & Setup

### Prerequisites
- Python 3.13+
- OpenCV 4.13.0+
- NumPy
- PyTorch (for PDR module)
- UV package manager

### Installation
```bash
# Clone repository
git clone https://github.com/abrahamchandafa/fyp
cd fyp

# Install dependencies
uv sync

# Verify installation
python -c "import cv2; import torch; import numpy; print('OK')"
```

### Project Structure
```
fyp/
├── generation/          # Synthetic 6-DoF thermal dot sequences
│   ├── generate.py      # Main generation script
│   ├── output/          # Generated frames and videos
│   └── common.py        # Configuration constants
│
├── pdr/                 # Thermal diffusion reversal
│   ├── model.py         # PDRNet U-Net architecture
│   ├── train.py         # Training script
│   ├── infer.py         # Inference/enhancement script
│   └── checkpoints/     # Trained model weights
│
├── track/               # Persistent blob tracking
│   ├── track_blob.py    # Main tracking script
│   ├── common.py        # Tracking utilities and constants
│   ├── output/          # CSV tracking results
│   └── benchmark.py     # Comparison of tracking methods
│
├── pose/                # SLAM pose estimation
│   ├── pose_slam.py     # Main SLAM pipeline
│   ├── common.py        # Pose utilities and metrics
│   ├── benchmark.py     # Benchmark against ground truth
│   └── checkpoints/     # (Legacy) old model files
│
├── tests/               # Comprehensive test suite (88 tests)
│   ├── test_generation.py  # 16 tests
│   ├── test_track.py       # 22 tests
│   ├── test_pose.py        # 15 tests
│   ├── test_pdr.py         # 26 tests
│   └── conftest.py         # Pytest configuration
│
├── pyproject.toml       # Project metadata and dependencies
├── README.md            # This file
└── main.py              # (Optional) Entry point script
```

---

## Usage Examples

### Complete Pipeline Execution

```bash
# Step 1: Generate synthetic frames (all 6 DoF)
cd generation
uv run ./generate.py

# Step 2: Projection-Diffusion Reversal
cd ../pdr
uv run ./benchmark.py

# Step 3: Track blobs for all DoF sequences
cd ../track
for dof in translate_x translate_y translate_z rotate_yaw rotate_pitch rotate_roll; do
    uv run ./track_blob.py $dof
done

# Step 4: Benchmark pose estimation
cd ../pose
uv run ./benchmark.py
```

### Single DoF Workflow

```bash
# Generate frames for one DoF
cd generation
uv run ./generate.py  # Generates all; to filter: modify generate.py

# Track specific sequence
cd ../track
uv run ./track_blob.py translate_x

# Estimate pose and view errors
cd ../pose
python -c "from pose_slam import run_dof; run_dof('translate_x')"
```

### Run Test Suite

```bash
# All tests
pytest tests/ -v

---

## Results Interpretation

### Expected Performance
- **Rotation Error**: 3-6° per DoF (excellent tracking geometry)
- **Translation Error**: 2-4° directional error
- **Valid Frames**: 75/75 (100% success rate)

### Typical Output
```
════════════════════════════════════════════════════════════════
           6-DoF SLAM Pose Estimation Results
════════════════════════════════════════════════════════════════
DoF              Valid Frames   Mean Rot Err (°)   Mean Trans Err (°)
────────────────────────────────────────────────────────────────
translate_x          75/75              5.2                3.1
translate_y          75/75              4.8                2.9
translate_z          75/75              6.1                4.2
rotate_yaw           75/75              3.5                2.1
rotate_pitch         75/75              4.0                2.5
rotate_roll          75/75              3.7                2.3
════════════════════════════════════════════════════════════════
```

### Debugging Failed Frames

**Insufficient 3D Points**: Tracked points may not be seen in enough frames for triangulation
- Solution: Lower `MIN_POINTS_FOR_PNP` threshold
- Check: `track/{dof}/{dof}_blob_tracks.csv` for point visibility

**Tracking Loss**: Blobs disappear or merge between frames
- Solution: Adjust `BRIGHTNESS_THRESHOLD` in `track/common.py`
- Check: Generated frames for motion blur or fading

**Bootstrap Failure**: First two frames don't establish initial pose
- Solution: Lower `BOOTSTRAP_MIN_TRACKS` or improve thermal image quality
- Check: Homography fallback is being used (logged)

---

## Key Implementation Details

### Persistent Tracking via CSV
- **Frame 0** produces `N` tracked points with IDs 0..N-1
- **Frame 1..74** each produces observations of a subset of those IDs
- CSV enables efficient cross-frame data exchange without in-memory state

### Bootstrap Strategy
```python
def bootstrap_pose(frame0_obs, frame1_obs):
    # Try essential matrix (works for non-planar motion)
    E = cv2.findEssentialMat(...)
    R, t = cv2.recoverPose(E, ...)
    
    if R is None:
        # Fallback: homography for planar/weak-baseline cases
        H = cv2.findHomography(...)
        Rs, ts = cv2.decomposeHomographyMat(H, K)
        # Use first decomposition
```

### Triangulation Validation
- Reconstructs 3D point from observations in multiple frames
- Reprojects to each frame and measures pixel error
- Rejects points with error > `MAX_REPROJ_ERROR` (2.5 px)
- Ensures geometric consistency

### PnP Solving
- Uses **EPnP** method (efficient for 4+ correspondences)
- More robust than ITERATIVE for sparse point clouds
- Handles outliers via RANSAC if needed
