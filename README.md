# FindDefault: Credit Card Fraud Prediction

## Project Overview
This project aims to predict fraudulent credit card transactions using machine learning. The dataset contains transaction details with a target label indicating whether a transaction is fraudulent. The primary goal is to build a robust classification model that can accurately distinguish between fraudulent and non-fraudulent transactions.

## Project Structure
The project is organized into the following structure:

```
├── data
│   ├── raw
│   ├── processed
├── models
├── notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   └── hyperparameter_tuning.ipynb
├── pipeline.py
├── README.md
```

### Folders
- **data**: Contains the raw and processed datasets.
- **models**: Stores trained machine learning models.
- **notebooks**: Includes Jupyter notebooks used for analysis, training, and hyperparameter tuning.

### Files
- **pipeline.py**: Contains the main code pipeline for data processing, model training, and evaluation.
- **README.md**: Project documentation.

## Requirements
To run this project, ensure the following dependencies are installed:

```bash
pandas
numpy
scikit-learn
imbalanced-learn
joblib
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Ensure the dataset is available in the `data/raw` directory.
2. **Run Pipeline**:
    ```bash
    python pipeline.py
    ```
3. **View Results**: Check the `models` directory for the trained model and evaluation metrics.

## Design Choices
- **Random Forest Classifier**: Selected for its robustness and interpretability.
- **SMOTE (Synthetic Minority Oversampling Technique)**: Used to handle class imbalance in the dataset.
- **Cross-Validation**: Applied during hyperparameter tuning to avoid overfitting.

## Future Work
- Experiment with other machine learning models such as Gradient Boosting or Neural Networks.
- Implement real-time fraud detection using streaming data.
- Explore interpretability techniques such as SHAP or LIME.

## Results
The model achieved:
- **Accuracy**: >75%
- **Precision and Recall**: Optimized to reduce false negatives.

---

### Acknowledgments
The project was completed as part of the Capstone program, applying advanced machine learning techniques to solve a practical problem in fraud detection.
