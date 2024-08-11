import joblib
import numpy as np

model_path = r"F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\Bank Customer Churn Predictor\\rfc_model.pkl"
model = joblib.load(model_path)

imaginary_customer = np.array([[600, 1, 0, 40, 3, 60000, 2, 1, 1, 50000, 0, 1]])

prediction = model.predict(imaginary_customer)

if prediction[0] == 1:
    print("The customer is likely to exit.")
else:
    print("The customer is not likely to exit.")
