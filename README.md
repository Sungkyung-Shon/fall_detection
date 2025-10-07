# **Fall Detection System with ST-GCN for Patrol Cars**
A real-time fall detection system for patrol cars that combines BoxMOT tracking with ST-GCN (Spatial Temporal Graph Convolutional Networks) to improve fallen person recognition accuracy.

## 🎯 Overview
This project implemets and enhanced fall detection system designed for patrol car applications.By integrating ST-GCN with the BoxMOT framwork, the system achieves more conservative and reliable fall detection with reduced false alarms.

### Key Improvements with ST-GCN
- **Reduced event duration**: 1.25s → 0.70s (44% improvement)
- **More conservative scoring**: Median score 0.790 → 0.413
- **Maintained overall performance** (AP=0.012) while improving reliability
- **Lower false positive rate**

### Core Components
- **[BoxMOT](https://github.com/mikel-brostrom/boxmot)**: Multi-object tracking framework
- **ST-GCN**: Spatial Temporal Graph Convolutional Network for skeleton-based action recognition

## 📊 Datasets Used

- **[Fall Detection Dataset (Roboflow)](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8)**: Primary fall detection training data
- **[GMDCSA24 Fall Detection Dataset](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)**: Additional validation dataset

  ## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Sungkyung-Shon/fall_detection.git
cd fall_detection

# Install BoxMOT
pip install boxmot

