# 💰 Loan Approval Prediction

An interactive **Machine Learning web application** that predicts whether a loan application is likely to be approved based on applicant and loan-related information.

The model is built using **Logistic Regression** and deployed with **Streamlit**, providing a simple and interactive interface for making predictions.

## 🚀 Live Demo

🔗 **Streamlit App:** Add your deployed Streamlit URL here

## 📌 Features

* 🔮 Predict loan approval instantly
* 👤 Applicant information input
* 💰 Income and loan amount analysis
* 🏠 Property area selection
* 💳 Credit history consideration
* 📊 Approval and rejection probabilities
* 🤖 Logistic Regression machine learning model
* ⚡ Interactive Streamlit interface
* 📈 Displays processed model inputs

## 🧠 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

### Input Features

The model uses the following features:

* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Gender
* Marital Status
* Dependents
* Education
* Self Employment
* Property Area

Categorical variables are converted into numerical features using one-hot encoding, and numerical features are scaled before being passed to the model.

## 📊 Model Performance

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **86.18%** |
| Recall   | **98.82%** |
| F1 Score | **90.81%** |

> Performance metrics are based on the test dataset used during model development.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Logistic Regression

## 📂 Project Structure

```text
loan-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── loan_model.pkl
    ├── scaler.pkl
    └── feature_names.json
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/loan-prediction.git
```

### 2. Navigate to the project

```bash
cd loan-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

## 🔮 How It Works

1. User enters applicant information.
2. Input data is converted into the required numerical features.
3. The saved scaler preprocesses the input.
4. The trained Logistic Regression model makes a prediction.
5. The application displays:

   * Loan approval/rejection
   * Approval probability
   * Rejection probability

## ☁️ Deployment

This application can be deployed easily using **Streamlit Community Cloud**.

### Basic deployment steps:

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the application.

Make sure the `models` folder and all model files are included in the repository.

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**. The prediction should not be considered a real financial or loan approval decision.

## 👨‍💻 Author

**Ishank Yadav**

B.Tech Computer Science — Artificial Intelligence & Machine Learning

---

⭐ If you found this project useful, consider giving it a star on GitHub!
