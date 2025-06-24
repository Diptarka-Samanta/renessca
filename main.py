from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import joblib
import sqlite3
from werkzeug.utils import secure_filename

# --- Flask Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}

# --- Create uploads folder if not exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Models ---
model = joblib.load('cibil_model_final.pkl')  # or 'cibil_model_final.pkl'
scaler = joblib.load('cibil_scaler_final.pkl')  # or 'cibil_scaler_final.pkl'

# --- Try to Import OCR ---
try:
    from your_ocr_script import extract_invoice_data
except ImportError:
    extract_invoice_data = None

# --- DB Setup ---
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- File Check ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# === ROUTES ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                       (username, email, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return redirect(url_for('login_page'))
    conn.close()
    return redirect(url_for('login_page'))

@app.route('/credit-score', methods=['GET'])
def credit_score_form():
    return render_template('credit-score.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        age = float(request.form['age'])
        income = float(request.form['income'])
        delayed = float(request.form['delayed'])
        cards = float(request.form['cards'])

        age = np.clip(age, 18, 100)
        income = np.clip(income, 50000, 1e7)
        delayed = np.clip(delayed, 0, 100)
        cards = np.clip(cards, 0, 20)

        user_df = pd.DataFrame([{
            'Age': age,
            'Annual_Income': income,
            'Num_of_Delayed_Payment': delayed,
            'Num_Credit_Card': cards
        }])

        user_scaled = scaler.transform(user_df)
        score = model.predict(user_scaled)[0]

        penalty = 0
        if delayed >= 90:
            penalty = 300
        elif delayed >= 70:
            penalty = 200
        elif delayed >= 50:
            penalty = 150
        elif delayed >= 30 and income < 200000:
            penalty = 100

        score -= penalty
        score = int(np.clip(score, 300, 900))

        if score >= 750:
            category = "Excellent"
            advice = "You are highly eligible for loans and credit cards at low interest."
        elif score >= 650:
            category = "Good"
            advice = "You have decent creditworthiness. You might get standard credit offers."
        elif score >= 550:
            category = "Fair"
            advice = "You may get limited credit. Improve your score by paying on time."
        else:
            category = "Poor"
            advice = "High risk for lenders. Reduce defaults and increase financial discipline."

        return render_template("result2.html", name=name, score=score, category=category, advice=advice)
    except Exception as e:
        return f"Error in prediction: {str(e)}"

@app.route('/extract', methods=['POST'])
def extract():
    if extract_invoice_data is None:
        return "OCR functionality not available", 500

    if 'invoice' not in request.files:
        return redirect('/')
    file = request.files['invoice']
    if file.filename == '':
        return redirect('/')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = extract_invoice_data(file_path)

        return render_template('result1.html',
                               invoice_number=result.get('Invoice Number', ''),
                               date=result.get('Date', ''),
                               amount=result.get('Amount', ''),
                               vendor=result.get('Vendor', ''))
    return "Invalid file type", 400

@app.route('/result1', methods=['GET'])
def upload_invoice_view():
    return render_template('result1.html',
                           invoice_number="",
                           date="",
                           amount="",
                           vendor="")

# === Run the App ===
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
