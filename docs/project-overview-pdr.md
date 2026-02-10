# Project Overview & Product Development Requirements (PDR)

## 1. Project Summary

**Name:** HAT-ReID (History-Aware Transformation for Person Re-Identification)
**Version:** 0.2.0 (Edge Case Hardened)
**Status:** Phase 3 In Progress
**Domain:** Computer Vision, Multi-Object Tracking, Person Re-Identification

HAT-ReID is a research module implementing temporal-aware person re-identification for video surveillance. It combines JointBDOE detection, deep feature extraction, and Hungarian matching with advanced re-ranking and motion validation to achieve stable identity tracking across occlusions and crowded scenes. Recently hardened with 25 edge case fixes (IoU guards, NaN prevention, GPU resource pooling, stale entry cleanup, deterministic tie-breaking).

**Target Problem:** Reduce ID fragmentation (1990→12 IDs in demo) and prevent ID theft during person crossings.

**Recent Milestone:** Edge case hardening complete (88/88 tests passing, 25 fixes across pipeline, gallery, matching)

## 2. Product Goals

### Primary Goals (Phase 1-2)
- **G1:** Reduce spurious track IDs via dual-track confirmation system
- **G2:** Implement robust feature matching with rank-list voting
- **G3:** Validate assignments using velocity-based motion history
- **G4:** Prevent ID theft during person crossing events

### Secondary Goals (Phase 3+)
- **G5:** Dynamic threshold adaptation for varying scenes
- **G6:** Support alternative feature extractors (FastReID, custom backbones)
- **G7:** Real-time performance on consumer GPUs (30 FPS @ 720p)

## 3. Functional Requirements

| ID | Requirement | Status | Notes |
|----|-------------|--------|-------|
| FR1 | Detect persons using JointBDOE (body + orientation) | Implemented | Primary detector, fallback YOLO11 available |
| FR2 | Extract 512-dim features using OSNet or FastReID | Implemented | Configurable backbone selection |
| FR3 | Match features using Hungarian algorithm + rank-list voting | Implemented | Triplet-loss weighted, top-k=20 |
| FR4 | Maintain permanent IDs after 5 consecutive frames | Implemented | Tentative→permanent transition |
| FR5 | Validate assignments against motion history | Implemented | Rejects velocity/direction violations |
| FR6 | Apply CVPR2017 k-reciprocal re-ranking | Implemented | Optional, +5-15% accuracy |
| FR7 | Detect and handle crossing tracks | Implemented | Hungarian + stricter thresholds |
| FR8 | Adapt similarity threshold dynamically | Implemented | Percentile-based adjustment |
| FR9 | Render visualization with gallery + HUD + analytics | Implemented | Extended frame layout |
| FR10 | Export track data (JSON format) | Implemented | Per-frame detections + IDs |

## 4. Non-Functional Requirements

| ID | Requirement | Target | Status |
|----|-------------|--------|--------|
| NFR1 | Real-time inference | 30 FPS @ 720p GPU | Implemented |
| NFR2 | Scalability | 100+ concurrent tracks | Implemented (dict-based gallery) |
| NFR3 | Memory efficiency | <2GB VRAM (GPU model) | Optimized (batch processing) |
| NFR4 | Configuration flexibility | 50+ parameters | Implemented (Pydantic YAML) |
| NFR5 | Code maintainability | Modular <200 LOC per file | Achieved |
| NFR6 | Documentation coverage | 100% public API | In progress |

## 5. Success Metrics

### Tracking Quality
- **ID Consistency:** <5 ID switches per person per minute
- **ID Completeness:** >85% of frames with assigned IDs for tracked persons
- **False Positives:** <10% spurious tracks (tentative rejection)
- **Track Fragmentation:** <3 fragments per person (target: 1-2)

### Computational
- **Latency:** <34ms per frame (30 FPS) on NVIDIA A100
- **Memory:** <1.5GB VRAM during inference
- **Throughput:** 100+ persons/frame supported

### Benchmark Results (Demo Video)
- Input: Multi-person surveillance, variable lighting, occlusions
- Unique IDs before dedup: 1990
- Unique IDs after HAT-ReID: 12
- Actual persons: ~16
- **Reduction:** 98.6% ID fragmentation elimination

## 6. Acceptance Criteria

### Phase 1: Core Implementation ✓
- [x] JointBDOE detector integrated
- [x] Feature extraction pipeline working
- [x] Hungarian assignment functional
- [x] Dual-track system (tentative→permanent) operational
- [x] Basic visualization rendering

### Phase 2: Advanced Matching ✓
- [x] Rank-list majority voting implemented
- [x] k-Reciprocal re-ranking (CVPR2017) integrated
- [x] Motion validation with velocity tracking
- [x] Crossing detection and ID theft prevention
- [x] Extended frame visualization (analytics layout)

### Phase 3: Edge Case Hardening & Optimization
- [x] Adaptive threshold adjustment
- [x] Quality-weighted feature fusion
- [x] Full configuration system (50+ params)
- [x] Edge case hardening: 25 fixes (IoU guards, NaN prevention, GPU pooling, stale cleanup)
- [x] Gallery robustness: schema validation, backups, EMA re-normalization, O(1) lookups
- [x] Deterministic tie-breaking and degenerate bbox handling
- [ ] Parameter tuning guide (in progress)

### Phase 4: Production Readiness
- [x] Unit test coverage: 88/88 tests passing (baseline set)
- [x] Core documentation updated (codebase-summary, system-architecture, code-standards)
- [ ] Comprehensive documentation completion
- [ ] Performance benchmarking suite
- [ ] Deployment guidelines

## 7. Architecture Overview

### High-Level Data Flow
```
Video Input
    ↓
[Detector] JointBDOE → Detections(box, conf, orientation)
    ↓
[Extractor] OSNet/FastReID → Features(512-dim, normalized)
    ↓
[Matcher] Hungarian + Rank-List → Assignments(det_idx→track_id)
    ↓
[Validator] Motion Check → Confirmed assignments
    ↓
[Gallery] Update features, increment counters
    ↓
[Visualization] Render boxes, IDs, stats, thumbnails
    ↓
Output (video, JSON tracks)
```

### Component Responsibilities

| Component | Role | Key Class |
|-----------|------|-----------|
| Detection | Person + orientation localization | `JointBDOEDetector` |
| Feature Extraction | 512-dim embedding computation | `ReIDFeatureExtractor`, `FastReIDExtractor` |
| Gallery | Track storage + assignment | `PersonGallery`, `GalleryEntry` |
| Matching | Hungarian + rank-list voting | `majority_vote_reidentify()`, `apply_torchreid_reranking()` |
| Pipeline | Orchestration + state management | `VideoReIDPipeline` |
| Visualization | Rendering + layout | `ExtendedFrameRenderer`, `HUDRenderer`, `GalleryPanelRenderer` |

## 8. Technical Constraints & Dependencies

### External Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Tensor operations, model inference |
| torchvision | >=0.15.0 | Vision utilities |
| torchreid | Latest | OSNet backbone, re-ranking algorithm |
| opencv-python | >=4.8.0 | Image processing, video I/O |
| lap | >=0.4.0 | Hungarian algorithm (linear assignment) |
| pydantic | >=2.0.0 | Config validation |
| numpy | >=1.24.0 | Numerical operations |
| scikit-learn | >=1.2.0 | Cosine similarity, metrics |

### Hardware Requirements
- **GPU:** NVIDIA GPU with CUDA capability (A100, RTX 4090, RTX 3090, T4)
- **RAM:** Minimum 8GB, recommended 16GB+
- **VRAM:** 2-4GB for model + batch processing

### Configuration Constraints
- Similarity threshold: 0.4-0.95 (cosine similarity)
- Rank list size: 1-50 (memory/accuracy tradeoff)
- Min frames for ID: 1-10 (noise tolerance)
- Velocity max speed: 50-200 pixels/frame (scene dependent)

## 9. Roadmap & Phases

### Phase 1: Core System (COMPLETED)
**Timeline:** Q4 2024
**Deliverables:** Basic pipeline, detection, extraction, matching

### Phase 2: Advanced Matching (COMPLETED)
**Timeline:** Q1 2025
**Deliverables:** Rank-list voting, re-ranking, motion validation, crossing detection

### Phase 3: Optimization & Tuning (IN PROGRESS)
**Timeline:** Q1-Q2 2025
**Deliverables:** Parameter guide, performance benchmarks, ablation studies

### Phase 4: Production & Documentation (PLANNED)
**Timeline:** Q2 2025
**Deliverables:** Full test coverage, deployment guide, API documentation

## 10. Known Limitations

1. **Single-GPU Inference:** No distributed processing support yet
2. **Fixed Frame Rate:** Assumes consistent FPS (no variable rate support)
3. **Scene Dependent:** Thresholds require tuning for different environments
4. **Occlusion Handling:** Limited by detector; extended occlusions cause track gaps
5. **Feature Dimension:** Fixed 512-dim from OSNet; custom extractors need retraining

## 11. Success Criteria Tracking

- [x] Reduces ID fragmentation from 1990→12 (98.6% reduction)
- [x] <5 ID switches per person (motion validation)
- [x] Real-time performance (30 FPS @ 720p)
- [x] Configurable via YAML (50+ parameters)
- [x] Prevents ID theft in crossing events
- [ ] Unit test coverage >80%
- [ ] Complete documentation suite
- [ ] Production deployment guide

## 12. Next Steps

1. **Immediate:** Complete parameter tuning guide (6+ scenarios, deadline 2025-02-15)
2. **Short-term:** Performance benchmarking suite (latency, memory, throughput)
3. **Medium-term:** Ablation studies on algorithm contributions
4. **Long-term:** Distributed processing, alternative backbones, graph-based tracking

---

**Document Version:** 1.1
**Last Updated:** 2026-02-10
**Maintained By:** Research Team
**Key Changes:** Added Phase 3 edge case hardening completion, 25 robustness fixes documented, version bumped to 0.2.0
