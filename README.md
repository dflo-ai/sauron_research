# HAT-ReID Research

History-Aware Transformation for Re-Identification features.
R&D repository for testing HAT-LDA before sauron-services integration.

https://github.com/dflo-ai/sauron_research/raw/master/assets/demo_tracked.mp4

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python scripts/benchmark_latency.py

# Run evaluation (requires video file)
python scripts/run_evaluation.py data/videos/test.mp4 --visualize
```

## Architecture

```
src/
├── hat_reid/     # Core HAT-LDA algorithm
│   ├── lda.py    # Linear Discriminant Analysis
│   └── queues.py # History queues (FIFO, Score-based)
├── tracker/      # ReID-enhanced tracker
│   └── hat_tracker.py
└── embeddings/   # Feature extraction
    └── extractor.py
```

## Key Components

### HAT-LDA (`src/hat_reid/lda.py`)
Learns discriminative subspace from track history. Maximizes between-track separation using shrinkage covariance estimation.

### HATTracker (`src/tracker/hat_tracker.py`)
Multi-object tracker with HAT integration. Uses appearance-only matching (pure ReID similarity).

### EmbeddingExtractor (`src/embeddings/extractor.py`)
Extracts ReID features from person crops using pretrained CNN (ResNet50/18, EfficientNet).

## Configuration

Configuration is defined in `configs/default.yaml`. Key parameters:

- `hat.transfer_factor_threshold`: History/tracks ratio to activate HAT (default: 3.0)
- `hat.history_max_len`: Max samples per track history (default: 100)
- `tracker.match_score_threshold`: Minimum similarity for association (default: 0.4)
- `hat.history_queue_type`: Queue type - "fifo" or "score"

See [`docs/configuration-parameters.md`](./docs/configuration-parameters.md) for complete parameter reference, tuning guide, and scenario-based configurations.

## Usage

```python
from src.hat_reid import LDA, FIFOQueue, get_device
from src.tracker import HATTracker
from src.embeddings import EmbeddingExtractor

# Initialize
device = get_device()  # Requires CUDA
tracker = HATTracker(device=str(device))
extractor = EmbeddingExtractor(device=str(device))

# Process frame
embeddings = extractor.extract_from_frame(frame, boxes)
track_ids = tracker.update(boxes, scores, embeddings, frame_id)
```

## Running Tests

```bash
pytest tests/ -v
```

## Integration with Sauron

See `docs/sauron-integration-guide.md` for migration instructions.

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- CUDA (GPU required per project spec)
