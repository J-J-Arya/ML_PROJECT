import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- Data ----------
# Simple dataset: hours studied vs pass (1) / fail (0)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# ---------- Sigmoid Function ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------- Training Function ----------
def train_logistic_regression(X, Y, lr=0.01, epochs=1000):
    m = 0  # weight (slope)
    b = 0  # bias (intercept)
    n = len(X)

    for _ in range(epochs):
        z = m * X + b
        y_pred = sigmoid(z)
        error = y_pred - Y

        # Gradients
        D_m = (1/n) * np.dot(error, X)
        D_b = (1/n) * np.sum(error)

        # Update weights
        m -= lr * D_m
        b -= lr * D_b

    return m, b

# Train the model
m, b = train_logistic_regression(X, Y)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Pass/Fail Predictor", layout="centered")

st.title("📘 Student Pass/Fail Predictor")
st.write("### Logistic Regression Model (From Scratch)")
st.info("This model predicts whether a student will pass or fail based on how many hours they studied.")

# Sidebar Input
st.sidebar.header("📊 Enter Study Hours")
input_hours = st.sidebar.slider("Hours Studied:", min_value=0.0, max_value=12.0, step=0.5)

# Prediction
z = m * input_hours + b
probability = sigmoid(z)
predicted_class = 1 if probability >= 0.5 else 0

# Output
st.metric(label="📈 Probability of Passing", value=f"{probability*100:.2f}%")
st.success("✅ Prediction: Pass") if predicted_class == 1 else st.warning("❌ Prediction: Fail")

# Show sigmoid curve (optional)
st.subheader("📉 Sigmoid Curve Visualization")
x_vals = np.linspace(0, 12, 100)
y_vals = sigmoid(m * x_vals + b)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="Sigmoid Curve", color="blue")
ax.scatter(X, Y, color="red", label="Training Data")
ax.axvline(x=input_hours, color='green', linestyle='--', label="Your Input")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Probability of Passing")
ax.set_title("Logistic Regression Fit")
ax.legend()
st.pyplot(fig)
