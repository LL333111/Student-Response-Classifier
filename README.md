# CSC311 Project — LLM Survey Classification

Course project for **CSC311: Introduction to Machine Learning** (University of Toronto, Fall 2025).

## Task
Predict which LLM (ChatGPT, Claude, or Gemini) a survey response refers to using ratings, task selections, and free-text feedback.

## Data
- Ordinal ratings (1–5)
- Multi-select task indicators
- Free-text responses  
- Final feature size: **33**
- Grouped train/val/test split by student ID (60/20/20) to prevent leakage

## Models
- XGBoost
- Random Forest
- Neural Network (DistilBERT + MLP)

## Results
- **Best model:** XGBoost  
- **Validation accuracy:** ~0.75  
- **Test accuracy:** ~0.64  

## Tools
Python, NumPy, Pandas, scikit-learn, XGBoost, PyTorch, HuggingFace

## Notes
Academic use only. Follows UofT academic integrity policy.
