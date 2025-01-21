from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Select features and target
X = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level']]
y = data['stroke']

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'stroke_model.pkl')

# List to store prediction history
prediction_history = []

@app.route('/')
def home():
    return render_template('index.html')  # Display main page

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data from user
    data = request.get_json(force=True)
    input_data = [data['age'], data['hypertension'], data['heart_disease'], data['avg_glucose_level']]

    # Load trained model
    if os.path.exists('stroke_model.pkl'):
        model = joblib.load('stroke_model.pkl')
    else:
        return jsonify({'error': 'Model not found'}), 404

    # Predict
    prediction = model.predict([input_data])
    output = int(prediction[0])

    # Save to prediction history
    prediction_history.append({'input': input_data, 'prediction': output})

    return jsonify({'prediction': output})

@app.route('/training-data')
def training_data():
    # Display the first 100 training data entries
    return render_template('training_data.html',
                           tables=[X_train.head(100).to_html(classes='data', index=False)],
                           titles=X_train.columns.values)

@app.route('/full-data')
def full_data():
    # Remove unnecessary whitespace and empty rows
    clean_data = data.dropna(how='all').reset_index(drop=True)  # Remove empty rows and reset index

    # Display the cleaned dataset
    return render_template('full_data.html',
                           tables=[clean_data.to_html(classes='data', index=False)],
                           titles=clean_data.columns.values)

@app.route('/prediction-history')
def prediction_history_view():
    # Display the prediction history
    return render_template('prediction_history.html',
                           history=prediction_history)

if __name__ == '__main__':
    app.run(debug=True)
