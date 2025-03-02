from flask import Flask, request, render_template
import pickle
import numpy as np

# โหลดโมเดล
model_path = "gb_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# สร้างแอป Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # หน้าเว็บที่ให้กรอกข้อมูล

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['married'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])

    # เตรียมข้อมูลเพื่อทำนาย
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi]])

    # ทำนายผล
    prediction = model.predict(input_data)

    # ส่งผลการทำนายไปยัง HTML
    if prediction == 1:
        result = 'มีความเสี่ยงเป็นโรคหลอดเลือดสมอง'
    else:
        result = 'ไม่มีความเสี่ยงเป็นโรคหลอดเลือดสมอง'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
