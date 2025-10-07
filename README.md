# **Fall Detection System with ST-GCN for Patrol Cars**
A real-time fall detection system for patrol cars that combines BoxMOT tracking with ST-GCN (Spatial Temporal Graph Convolutional Networks) to improve fallen person recognition accuracy.

## 🎯 Overview
This project implemets and enhanced fall detection system designed for patrol car applications.By integrating ST-GCN with the BoxMOT framwork, the system achieves more conservative and reliable fall detection with reduced false alarms.

### Key Improvements with ST-GCN
- **Reduced event duration**: 1.25s → 0.70s (44% improvement)
- **More conservative scoring**: Median score 0.790 → 0.413
- **Maintained overall performance** (AP=0.012) while improving reliability
- **Lower false positive rate**
  
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/cbe692d7-533c-4434-9927-da42086784ba" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4219df43-f1ca-48c1-9ea5-37bda3a6e041" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/1d78fa6a-b110-4b2f-b4d3-2d99b045f6ad" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4c30ea4c-44e6-4082-9d87-8ea060a76f11" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/66909088-7602-4e2f-bd91-ad0d759b36e4" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4db4349f-9493-4e86-951a-de3979ca709c" />







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

