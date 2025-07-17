#THIS IS A LINEAR REGRESSION PROBLEM

import streamlit as st

# Training data
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]

# Linear Regression Training from scratch
def train_linear_regression(X, Y, lr=0.012, epochs=1000):    #ILL BE PROUD OF MYSELF BECAUSE lr was 0.1 in CHATGPT WHICH GAVE THE WRONG OUTPUT,BUT NOW I CORRECT AND GOT THE EXACT OUTPUT
    n = len(X)
    m = b = 0

    for _ in range(epochs):
        y_pred = [m*x + b for x in X]
        D_m = (-2/n) * sum([x*(y - y_hat) for x, y, y_hat in zip(X, Y, y_pred)])
        D_b = (-2/n) * sum([(y - y_hat) for y, y_hat in zip(Y, y_pred)])
        m -= lr * D_m
        b -= lr * D_b
    return m, b

# Train model
m, b = train_linear_regression(X, Y)

# Streamlit UI
st.title("🔢 Linear Regression Predictor (From Scratch)")

input_value = st.number_input("Enter a number to predict:", step=1.0, format="%.2f")

if st.button("Predict"):
    prediction = m * input_value + b
    st.success(f"Predicted value: **{round(prediction, 2)}**")