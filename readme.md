# Edge-AI Defect Classification for Semiconductor SEM Images

## Problem
Real-time defect detection in semiconductor fabs suffers from latency and bandwidth issues.

## Solution
Lightweight CNN model deployed to Edge (ONNX format) for real-time defect classification.

## Dataset
- 10 SEM defect classes
- Grayscale images
- 70/15/15 split

## Model
- CNN (3 Conv layers)
- Input: 128x128 grayscale
- Accuracy: 96.25%
- Framework: TensorFlow
- Exported to ONNX

## Edge Deployment
- Compatible with NXP eIQ pipeline (software flow)

## Results
Accuracy: 96.25%  
Precision/Recall: >96% average  

## How to Run
pip install -r requirements.txt  
python train_model.py  
python export_onnx.py  



