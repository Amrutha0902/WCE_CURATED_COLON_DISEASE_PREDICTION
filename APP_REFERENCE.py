# ============================================================================
# COLON DISEASE PREDICTION - COMPLETE APP.PY CODE REFERENCE
# Framework: Flask 3.1.2 | Database: MySQL 8.0+ | ML: Keras VGG16
# ============================================================================

import os
os.environ['KERAS_BACKEND'] = 'jax'

from flask import Flask, request, render_template, redirect, url_for, session, send_file
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import pandas as pd
from functools import wraps
import io
import json

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
app.secret_key = 'colon_disease_prediction_secret_key_2026'

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ammu@221025',  # Change in production
    'database': 'colon_disease_db'
}

# ============================================================================
# LOAD ML MODEL
# ============================================================================
model = load_model('model_1/Vgg.h5')
print('Model Loaded!')

# ============================================================================
# SECTION 1: DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """
    Establish MySQL connection.
    Returns: mysql.connector.connection or None
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def save_prediction_to_db(filename, predicted_class, confidence, image_path, username=None):
    """
    Save prediction to MySQL predictions table.
    
    Args:
        filename (str): Original uploaded filename
        predicted_class (str): CNN classification output
        confidence (float): Model confidence score (0-1)
        image_path (str): Server path to saved image
        username (str): Current user (from session)
    
    Returns: bool
    """
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = """
                INSERT INTO predictions (filename, predicted_class, confidence, image_path, username)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (filename, predicted_class, float(confidence), image_path, username))
            connection.commit()
            print(f"✓ Prediction saved for {username}: {filename} -> {predicted_class} ({confidence:.4f})")
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"✗ Database insert error: {e}")
            return False
    return False

# ============================================================================
# SECTION 2: AUTHENTICATION FUNCTIONS
# ============================================================================

def login_required(f):
    """Decorator to protect routes - requires login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return render_template('auth_required.html', redirect_page=f.__name__)
        return f(*args, **kwargs)
    return decorated_function

def verify_doctor_credentials(username, password):
    """Verify username/password against users table."""
    username = (username or "").strip()
    password = (password or "").strip()

    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT password, role FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            row = cursor.fetchone()
            cursor.close()
            connection.close()

            if not row:
                return False
            stored_password, role = row
            return stored_password == password
        except Error as e:
            print(f"Authentication error: {e}")
            return False
    return False

def user_exists(username):
    """Check if username already exists in users table."""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
            return user is not None
        except Error as e:
            print(f"Database error: {e}")
            return False
    return False

def register_user(username, password, email, role='doctor'):
    """Register new user in users table."""
    username = (username or "").strip()
    password = (password or "").strip()
    email = (email or "").strip()
    role = role or 'doctor'

    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
            cursor.execute(query, (username, password, role))
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"Registration error: {e}")
            return False
    return False

# ============================================================================
# SECTION 3: FLASK ROUTES - PUBLIC
# ============================================================================

@app.route('/')
def index():
    is_logged_in = 'logged_in' in session
    return render_template('index.html', is_logged_in=is_logged_in)

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('signup_username')
    password = request.form.get('signup_password')
    confirm_password = request.form.get('signup_confirm_password')
    email = request.form.get('signup_email')
    
    if not username or not password or not confirm_password or not email:
        return render_template('index.html', error_signup="All fields required", is_logged_in=False)
    
    if password != confirm_password:
        return render_template('index.html', error_signup="Passwords do not match", is_logged_in=False)
    
    if len(password) < 6:
        return render_template('index.html', error_signup="Min 6 characters", is_logged_in=False)
    
    if user_exists(username):
        return render_template('index.html', error_signup="Username exists", is_logged_in=False)
    
    if register_user(username, password, email):
        return render_template('index.html', success_signup="Account created! Login now.", is_logged_in=False)
    else:
        return render_template('index.html', error_signup="Registration failed", is_logged_in=False)

@app.route('/login-home', methods=['POST'])
def login_home():
    username = request.form.get('login_username')
    password = request.form.get('login_password')
    
    if verify_doctor_credentials(username, password):
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('index'))
    else:
        return render_template('index.html', error_login="Invalid credentials", is_logged_in=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    is_logged_in = 'logged_in' in session
    return render_template('details.html', is_logged_in=is_logged_in)

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ============================================================================
# SECTION 4: FLASK ROUTES - PROTECTED (Authentication Required)
# ============================================================================

@app.route('/analysis')
@login_required
def analysis():
    """
    Data Analysis Dashboard.
    Queries predictions table, filters by date range, generates analytics.
    """
    try:
        connection = get_db_connection()
        
        if connection:
            try:
                # Get date range filter from request
                range_option = (request.args.get('range', 'till_date') or 'till_date').lower()
                now = datetime.now()
                
                range_config = {
                    'today': ('Today', datetime.combine(now.date(), datetime.min.time())),
                    'week': ('Last 7 Days', now - timedelta(days=7)),
                    'month': ('Last 30 Days', now - timedelta(days=30)),
                    '6months': ('Last 6 Months', now - timedelta(days=182)),
                    'year': ('Last 12 Months', now - timedelta(days=365)),
                    'till_date': ('Till Date', None)
                }

                if range_option not in range_config:
                    range_option = 'till_date'

                selected_range_label, start_date = range_config[range_option]

                # Query predictions for current user
                username = session.get('username')
                query = "SELECT predicted_class, confidence, prediction_timestamp FROM predictions WHERE username = %s"
                params = [username]

                if start_date:
                    query += " AND prediction_timestamp >= %s"
                    params.append(start_date)

                query += " ORDER BY prediction_timestamp DESC"
                df = pd.read_sql(query, connection, params=tuple(params))
                connection.close()
                
                if df.empty:
                    return render_template('analysis.html', 
                                         error="No predictions yet",
                                         is_logged_in=True,
                                         selected_range=range_option,
                                         selected_range_label=selected_range_label)
                
                # Analytics: Class Distribution
                all_classes = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
                
                class_counts = df['predicted_class'].value_counts()
                total_predictions = len(df)
                avg_confidence = df.groupby('predicted_class')['confidence'].mean()
                
                chart_labels = all_classes
                chart_counts = [int(class_counts.get(cls, 0)) for cls in all_classes]
                chart_confidences = [float(avg_confidence.get(cls, 0)) for cls in all_classes]
                
                chart_data = {
                    'labels': chart_labels,
                    'counts': chart_counts,
                    'avg_confidence': chart_confidences
                }
                
                stats = {
                    'total_predictions': total_predictions,
                    'class_distribution': class_counts.to_dict(),
                    'avg_confidence': {k: f"{v:.4f}" for k, v in avg_confidence.to_dict().items()},
                    'chart_data': json.dumps(chart_data)
                }
                
                return render_template('analysis.html',
                                       stats=stats,
                                       is_logged_in=True,
                                       selected_range=range_option,
                                       selected_range_label=selected_range_label)
                
            except Exception as e:
                print(f"Analysis error: {e}")
                import traceback
                traceback.print_exc()
                return render_template('analysis.html', 
                                     error=f"Error: {str(e)}",
                                     is_logged_in=True,
                                     selected_range='till_date',
                                     selected_range_label='Till Date')
        else:
            return render_template('analysis.html', 
                                 error="Database connection failed",
                                 is_logged_in=True,
                                 selected_range='till_date',
                                 selected_range_label='Till Date')
    except Exception as e:
        print(f"Critical error in analysis: {e}")
        return f"Error: {str(e)}", 500

@app.route('/download-report')
@login_required
def download_report():
    """
    Download predictions as CSV file.
    File naming: colon_disease_predictions_YYYYMMDD_HHMMSS.csv
    """
    connection = get_db_connection()
    
    if connection:
        try:
            username = session.get('username')
            query = "SELECT id, filename, predicted_class, confidence, prediction_timestamp, image_path FROM predictions WHERE username = %s ORDER BY prediction_timestamp DESC"
            df = pd.read_sql(query, connection, params=(username,))
            connection.close()
            
            if df.empty:
                return "No prediction data to download", 404
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            csv_bytes = io.BytesIO()
            csv_bytes.write(csv_buffer.getvalue().encode('utf-8'))
            csv_bytes.seek(0)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'colon_disease_predictions_{timestamp}.csv'
            
            return send_file(
                csv_bytes,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
        except Error as e:
            return f"Database error: {e}", 500
    else:
        return "Connection failed", 500

@app.route('/result', methods=['POST', 'GET'])
@login_required
def result():
    """
    Prediction result page.
    1. Receives image upload
    2. Preprocesses image (224x224)
    3. Runs CNN prediction
    4. Saves to MySQL
    5. Returns result
    """
    if request.method == 'POST':
        f = request.files['image']
        
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        
        # Preprocess image
        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        
        # CNN prediction
        predictions = model.predict(x)
        class_index = np.argmax(predictions, axis=1)[0]
        class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
        predicted_class = class_names[class_index]
        confidence = predictions[0][class_index]
        
        # Save to MySQL
        username = session.get('username')
        save_prediction_to_db(
            filename=f.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            image_path=filepath,
            username=username
        )
        
        result_text = f"Prediction: {predicted_class} with confidence {confidence:.2%}"
        predicted_class_display = predicted_class
        confidence_display = f"{confidence:.2%}"
        
        return render_template(
            'result.html',
            result=result_text,
            predicted_class=predicted_class_display,
            confidence=confidence_display,
            is_logged_in=True
        )
    
    return redirect(url_for('predict'))

# ============================================================================
# FLASK APP LAUNCHER
# ============================================================================

if __name__ == '__main__':
    print("\n=== REGISTERED ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("========================\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
