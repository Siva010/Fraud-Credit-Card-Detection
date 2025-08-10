# Fraud-Credit-Card-Detection
Of course\! Here is a well-structured and professional README file for your GitHub project. You can copy and paste this directly into a `README.md` file in your repository.

-----

# Credit Card Fraud Detection

## Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions. It utilizes a real-world dataset to train a classifier that can distinguish between legitimate and fraudulent transactions.

The key challenge in this problem is the highly imbalanced nature of the dataset, where fraudulent transactions are a very small minority. This project demonstrates a complete workflow, from data exploration and preprocessing to model training and, most importantly, nuanced evaluation using metrics appropriate for imbalanced data.

The analysis is performed in the Jupyter Notebook: `Credit Card Fraud Detection.ipynb`.

## Dataset

The project uses the "Credit Card Fraud Detection" dataset available on Kaggle.

  * **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  * **Content:** The dataset contains transactions made by European cardholders in September 2013.
  * **Features:** It has 30 features, including `Time`, `Amount`, and 28 anonymized features (`V1` to `V28`) that are the result of a Principal Component Analysis (PCA) transformation.
  * **Imbalance:** The dataset is highly imbalanced. Out of 284,807 transactions, only 492 (0.172%) are fraudulent.

## Project Workflow

1.  **Data Loading and Exploration:** The `creditcard.csv` dataset is loaded into a pandas DataFrame. An initial analysis is performed to understand the data's structure and the severe class imbalance.

2.  **Data Preprocessing:**

      * The features and target variable (`Class`) are separated.
      * The feature set is scaled using `StandardScaler` from scikit-learn to ensure that all features have a similar scale, which is crucial for algorithms like SVM.

3.  **Model Training:**

      * The data is split into an 80% training set and a 20% testing set.
      * A **Support Vector Classifier (SVC)** is chosen as the classification model and is trained on the preprocessed training data.

4.  **Model Evaluation:**

      * The model's performance is evaluated on the unseen test set.
      * Given the class imbalance, accuracy is not a reliable metric. Therefore, the evaluation focuses on:
          * **Confusion Matrix:** To visualize the number of correct and incorrect predictions for each class.
          * **Classification Report:** To analyze precision, recall, and F1-score for the fraudulent class.

## Results and Key Findings

While the model achieves a high accuracy of over 99.9%, this is misleading. The key performance metrics for the minority class (fraud) are:

| Metric    | Score |
| :-------- | :---- |
| Precision | 0.95  |
| Recall    | 0.63  |
| F1-Score  | 0.76  |

**Confusion Matrix on Test Data:**

|              | Predicted Fraud | Predicted Normal |
| :----------- | :-------------: | :--------------: |
| **Is Fraud** |       55        |        32        |
| **Is Normal**|        3        |      56872       |

**Conclusion:**

  * The model is highly **precise**: When it predicts a transaction is fraudulent, it is correct 95% of the time.
  * The model's **recall** is only 63%. This is a significant finding, as it means the model fails to identify **37%** of the actual fraudulent transactions.
  * This demonstrates the classic **precision-recall tradeoff**. In a real-world scenario, a bank would need to decide whether to optimize for higher recall (catching more fraud at the cost of more false positives) or higher precision.

## How to Run This Project

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is included for easy installation.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

    Then, open the `Credit Card Fraud Detection.ipynb` file to view and run the analysis.

## Potential Future Improvements

This project serves as a solid baseline. To improve the model's performance, especially the recall score, the following could be explored:

  * **Handling Class Imbalance:** Implement resampling techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **Random Undersampling** on the training data.
  * **Experiment with Different Models:** Compare the SVC's performance against other algorithms known to perform well on imbalanced data, such as **Random Forest**, **XGBoost**, or **LightGBM**.
  * **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the chosen model to potentially improve recall and F1-score.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
