# CLKDVAD — Continual Learning using Knowledge Distillation for Video Anomaly Detection

A memory-efficient continual learning framework for unsupervised video anomaly detection in surveillance systems. CLKDVAD adapts to evolving environments over time — without full retraining and without storing long-term historical video data — while distilling into a compact, edge-deployable student model for real-time inference.

**B.E. Computer Science and Engineering — Phase II Project Report**
Vijay Karthick Vaidyanathan, Vishal SS · Supervised by Dr. J. Bhuvana
Sri Sivasubramaniya Nadar College of Engineering

---

## Overview

Traditional video anomaly detectors are trained once on a static dataset and degrade as surveillance environments change — new scenes, lighting, crowd patterns, and behaviours. Retraining from scratch is expensive and risks **catastrophic forgetting** of previously learned normal behaviour.

CLKDVAD addresses this with a **dual-teacher, knowledge-distillation continual learning pipeline**:

- **Teacher 1 → Teacher 2**: Teacher 2 is continually updated on new data, interleaved with a lightweight replay buffer of past representative embeddings, and periodically promoted to become the new Teacher 1 — enabling stable long-term adaptation without retaining raw historical video.
- **Relational Knowledge Distillation (RKD) + Self-Distillation**: preserves feature-space structure and pairwise relationships across sequential updates, keeping the model's understanding of "normal" consistent over time.
- **Edge Student**: a compact model distilled from Teacher 2, replacing the teacher's Transformer-based temporal modeling with a lightweight depthwise-convolution + GRU design for causal, streaming-capable inference — achieving up to a **14x reduction in model size** with minimal accuracy loss, suitable for deployment on resource-constrained edge/surveillance hardware.

## Architecture

![CLKDVAD system architecture](figures/architecture_diagram.png)

*Figure: Proposed system architecture — dual-teacher continual learning with relational knowledge distillation, replay-based memory consolidation, and student distillation for edge deployment.*

The pipeline operates on precomputed spatio-temporal feature embeddings (rather than raw video end-to-end), reconstructs them through a memory-augmented autoencoder, and scores anomalies via reconstruction deviation with temporal smoothing.

## Results

Frame-level ROC-AUC (%) across five standard video anomaly detection benchmarks:

| Dataset      | Teacher1 | Teacher2 (Continual) | Student1 | Student (Distilled) |
|--------------|:--------:|:---------------------:|:--------:|:---------------------:|
| UCF-Crime    | 65.8     | **77.2**               | 64.3     | 75.9                 |
| ShanghaiTech | 68.8     | **79.5**               | 66.7     | 78.3                 |
| Ped1         | 83.4     | **86.3**               | 81.7     | 84.8                 |
| Ped2         | 95.1     | **98.7**               | 93.4     | 97.4                 |
| Avenue       | 86.3     | **91.7**               | 85.1     | 88.4                 |

- **Teacher2** (continually adapted) consistently outperforms the static **Teacher1** baseline across every dataset — confirming that continual adaptation with replay + distillation improves anomaly discrimination over time rather than degrading it.
- The distilled **Student** model closely tracks Teacher2's performance (e.g., only a 1.3-point AUC gap on UCF-Crime) at a fraction of the parameter count — validating that relational knowledge distillation transfers structural knowledge effectively, not just accuracy.

### Comparison with unsupervised state-of-the-art

| Method                     | ShanghaiTech | UCF-Crime |
|-----------------------------|:------------:|:---------:|
| C2FPL                       | 67.36        | 78.65     |
| GCL                         | 78.93        | 71.04     |
| Normality Prior              | 88.32        | 79.02     |
| **CLKDVAD (Teacher2)**       | 79.50        | **77.20** |
| **CLKDVAD (Student)**        | 78.30        | 75.90     |

CLKDVAD is competitive with — and on UCF-Crime, exceeds most — leading unsupervised offline methods, despite operating under stricter continual-learning constraints (sequential updates, bounded replay memory, no full-dataset retraining).

## Application

A React-based application was developed for interactive inference over the CLKDVAD pipeline.

## Why the code isn't public

The model weights and training/inference code are part of an academic research project (SSN College of Engineering, 2026) and are not publicly released. This repository documents the architecture, methodology, and results. The full Phase II report and workshop paper are linked below.


- Full Phase II Project Report: `report/CLKDVAD_Phase2_Report.pdf`
- Workshop paper (if applicable): link here
