# Bank Transaction Fraud Detector 🏦🕵️‍♂️

A machine learning project to detect fraudulent bank transactions using Python and Scikit-learn. This project analyzes a transactional dataset to build and evaluate a robust classification model capable of identifying suspicious activity in real-time.

---

## 📋 Project Workflow

This project follows a complete data science workflow, from data exploration to model evaluation, ensuring a comprehensive analysis.

1.  **Data Loading & Initial Analysis:**
    * The dataset is loaded using Pandas.
    * Initial exploratory data analysis (EDA) is performed to understand the data's structure and identify key features.

2.  **Exploratory Data Analysis (EDA) & Visualization:**
    * Visualizations are created using Matplotlib and Seaborn to understand the distribution of fraudulent vs. normal transactions.
    * The relationships between different features (e.g., purchase value, online orders) and fraudulent activity are explored to uncover patterns.

3.  **Data Preprocessing & Feature Engineering:**
    * The data is prepared for modeling using a Scikit-learn pipeline.
    * Numerical features are scaled using `StandardScaler` to ensure the model performs optimally.

4.  **Model Training:**
    * The dataset is split into training and testing sets.
    * A Logistic Regression model is trained on the data. The `class_weight='balanced'` parameter is used to handle the imbalanced nature of the dataset effectively.

5.  **Model Evaluation:**
    * The model's performance is evaluated on the unseen test set.
    * Key metrics for this problem, including **Precision**, **Recall**, and the **F1-Score**, are analyzed from a detailed `classification_report`.
    * A **Confusion Matrix** is generated to visualize the model's accuracy in distinguishing between classes.

---

## 📊 Dataset

The dataset used for this analysis is `creditcard_sample.csv`. It contains anonymized transaction data with features such as distance from home, purchase price ratio, whether the order was online, and a target column indicating if the transaction was fraudulent.

---

## 🛠️ Technologies Used

* **Python:** The core programming language for the analysis.
* **Pandas:** For data manipulation and loading.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For building and evaluating the machine learning model.
* **Google Colab / Jupyter Notebook:** As the environment for writing and running the code.

---

## 🚀 How to Run

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/bank-transaction-fraud-detector.git](https://github.com/YOUR_USERNAME/bank-transaction-fraud-detector.git)
    ```
2.  **Open the notebook:**
    * Upload the `.ipynb` file and the `creditcard_sample.csv` file to your Google Colab or local Jupyter Notebook environment.
3.  **Run the cells:**
    * Execute the cells in the notebook sequentially to see the analysis and results.

---

## 📈 Model Performance

The final model demonstrates a strong ability to identify fraudulent transactions while maintaining a low rate of false positives.

* **Recall Score:** The model successfully identifies a high percentage of all actual fraudulent transactions, which is the primary goal for a fraud detection system.
* **Precision Score:** The model shows good precision, ensuring that when it flags a transaction as fraudulent, it is correct a high percentage of the time.

The detailed metrics can be found in the classification report and confusion matrix within the notebook.
