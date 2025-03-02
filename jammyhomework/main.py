import pickle
import numpy as np

model_path = "gb_model.pkl"
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)
f = open(model_path, 'rb')
model = pickle.load(f)
f.close()

gender = 1
age = 67
hypertension = 0
heart_disease = 1
ever_married = 1
avg_glucose_level = 228.69
bmi = 36.6

# เตรียมข้อมูลเพื่อทำนาย
input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi]])
# ทำนายผล
prediction = model.predict(input_data)
# ส่งผลการทำนายไปยัง HTML

print(f"prediction = {prediction}")
if prediction == 1:
    print('มีความเสี่ยงเป็นโรคหลอดเลือดสมอง')
elif prediction == 0:
    print('ไม่มีความเสี่ยงเป็นโรคหลอดเลือดสมอง')