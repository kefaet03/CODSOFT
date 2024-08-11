import joblib
import numpy as np

# Load the trained model
model_path = r"F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\Credit Card Fraud Detection\\fraud_transaction_detector.pkl"
model = joblib.load(model_path)

# Define an imaginary transaction (replace with your feature vector)
imaginary_transaction = np.array([[150.75, 120, 1, 0, 0, 0, 1, 0]])

# Predict if the transaction is fraudulent
prediction = model.predict(imaginary_transaction)

# Output the prediction
if prediction[0] == 1:
    print("The transaction is likely fraudulent.")
else:
    print("The transaction is not likely fraudulent.")
