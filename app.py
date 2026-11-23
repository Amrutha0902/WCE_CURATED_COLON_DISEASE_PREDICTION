import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from dotenv import load_dotenv

load_dotenv()
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

app = Flask(__name__)
app.secret_key = os.environ['FLASK_SECRET_KEY']  # Change in production

# MySQL Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASSWORD'],
    'database': os.getenv('DB_NAME', 'colon_disease_db')
}

# Load your model
model = load_model('model_1/Vgg.h5')

print('Model Loaded!')

# Database Connection Function
def get_db_connection():
    """Establish MySQL database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

# Save Prediction to Database
def ensure_predictions_user_column():
    """Ensure predictions table has a username column to support per-user data."""
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        check_query = """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = %s AND table_name = 'predictions' AND column_name = 'username'
        """
        cursor.execute(check_query, (DB_CONFIG['database'],))
        exists = cursor.fetchone()[0] > 0
        if not exists:
            alter_query = "ALTER TABLE predictions ADD COLUMN username VARCHAR(255) AFTER confidence"
            cursor.execute(alter_query)
            connection.commit()
            print("Added username column to predictions table for per-user tracking")
        cursor.close()
        connection.close()
        return True
    except Error as e:
        print(f"Schema check/add error: {e}")
        connection.close()
        return False


def save_prediction_to_db(filename, predicted_class, confidence, image_path, username=None):
    """Insert prediction result into MySQL database with user association"""
    # Ensure schema supports username; if not, attempt to add it
    ensure_predictions_user_column()

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
            print(f"Prediction saved: {filename} -> {predicted_class} ({confidence:.4f}) for user {username}")
            cursor.close()
            connection.close()
            return True
        except Error as e:
            # Fallback: if username column truly missing or insert fails, log and skip insert to avoid silent data issues
            print(f"Database insert error: {e}")
            connection.close()
            return False
    return False

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return render_template('auth_required.html', redirect_page=f.__name__)
        return f(*args, **kwargs)
    return decorated_function

# Verify Doctor Credentials
def verify_doctor_credentials(username, password):
    """Verify doctor login credentials against MySQL database"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT * FROM users WHERE username = %s AND password = %s AND role = 'doctor'"
            cursor.execute(query, (username, password))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
            return user is not None
        except Error as e:
            print(f"Authentication error: {e}")
            return False
    return False

# Check if user exists
def user_exists(username):
    """Check if username already exists"""
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

# Register new user
def register_user(username, password, email=None, role='doctor'):
    """Register a new user in the database"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Insert only the available columns to match existing schema
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

@app.route('/')
def index():
    is_logged_in = 'logged_in' in session
    return render_template('index.html', is_logged_in=is_logged_in)

@app.route('/signup-page')
def signup_page():
    """Display Signup Page"""
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup():
    """User Signup Route"""
    username = request.form.get('signup_username')
    password = request.form.get('signup_password')
    confirm_password = request.form.get('signup_confirm_password')
    email = request.form.get('signup_email')  # optional; ignored if column absent
    
    # Validation
    if not username or not password or not confirm_password:
        return render_template('index.html', 
                             error_signup="Username and password are required",
                             is_logged_in=False)
    
    if password != confirm_password:
        return render_template('index.html', 
                             error_signup="Passwords do not match",
                             is_logged_in=False)
    
    if len(password) < 6:
        return render_template('index.html', 
                             error_signup="Password must be at least 6 characters",
                             is_logged_in=False)
    
    if user_exists(username):
        return render_template('index.html', 
                             error_signup="Username already exists",
                             is_logged_in=False)
    
    # Register user
    if register_user(username, password, email):
        return render_template('index.html', 
                             success_signup="Account created successfully! You can now login.",
                             is_logged_in=False)
    else:
        return render_template('index.html', 
                             error_signup="Registration failed. Please try again.",
                             is_logged_in=False)

@app.route('/login-page')
def login_page():
    """Display Login Page"""
    return render_template('login.html')

@app.route('/login-home', methods=['POST'])
def login_home():
    """Doctor Login Route from Home Page"""
    username = request.form.get('login_username')
    password = request.form.get('login_password')
    error = None
    
    if verify_doctor_credentials(username, password):
        session['logged_in'] = True
        session['username'] = username
        return render_template('index.html', 
                             success_login="Login successful! Redirecting...",
                             is_logged_in=True)
    else:
        error = "Invalid username or password"
        return render_template('index.html', 
                             error_login=error,
                             is_logged_in=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Legacy Login Route (redirects to home)"""
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Logout Route"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')  # Assuming you have an about.html template

@app.route('/predict')
def predict():
    is_logged_in = 'logged_in' in session
    return render_template('details.html', is_logged_in=is_logged_in)

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Assuming you have a contact.html template

@app.route('/analysis')
@login_required
def analysis():
    """Data Analysis Dashboard - Query predictions and generate visualization"""
    username = session.get('username')
    if not username:
        return redirect(url_for('login_page'))

    # Date range filter
    range_option = request.args.get('range', 'till_date')
    now = datetime.now()
    start_date = None
    if range_option == 'today':
        start_date = datetime.combine(now.date(), datetime.min.time())
    elif range_option == 'week':
        start_date = now - timedelta(days=7)
    elif range_option == 'month':
        start_date = now - timedelta(days=30)
    elif range_option == 'six_months':
        start_date = now - timedelta(days=182)
    elif range_option == 'year':
        start_date = now - timedelta(days=365)
    # till_date -> no start_date filter
    try:
        connection = get_db_connection()
        
        if connection:
            try:
                # Query predictions data
                base_query = "SELECT predicted_class, confidence, prediction_timestamp FROM predictions WHERE username = %s"
                params = [username]
                if start_date:
                    base_query += " AND prediction_timestamp >= %s"
                    params.append(start_date)
                query = base_query + " ORDER BY prediction_timestamp DESC"
                df = pd.read_sql(query, connection, params=tuple(params))
                connection.close()
                
                if df.empty:
                    return render_template('analysis.html', 
                                         error="No prediction data for your account yet. Make some predictions first!",
                                         is_logged_in=True,
                                         selected_range=range_option)
                
                # Data Analysis: Class Distribution
                # Define all possible disease classes
                all_classes = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
                
                class_counts = df['predicted_class'].value_counts()
                total_predictions = len(df)
                avg_confidence = df.groupby('predicted_class')['confidence'].mean()
                
                # Ensure all classes are included (even with 0 count)
                chart_labels = all_classes
                chart_counts = [int(class_counts.get(cls, 0)) for cls in all_classes]
                chart_confidences = [float(avg_confidence.get(cls, 0)) for cls in all_classes]
                
                # Prepare data for Chart.js (client-side rendering)
                chart_data = {
                    'labels': chart_labels,
                    'counts': chart_counts,
                    'avg_confidence': chart_confidences
                }
                
                # Prepare statistics
                stats = {
                    'total_predictions': total_predictions,
                    'class_distribution': class_counts.to_dict(),
                    'avg_confidence': {k: f"{v:.4f}" for k, v in avg_confidence.to_dict().items()},
                    'chart_data': json.dumps(chart_data)  # Send data as JSON for Chart.js
                }
                
                return render_template('analysis.html', stats=stats, is_logged_in=True, selected_range=range_option)
                
            except Exception as e:
                print(f"Analysis error: {e}")
                import traceback
                traceback.print_exc()
                return render_template('analysis.html', 
                                     error=f"Error generating analysis: {str(e)}",
                                     is_logged_in=True,
                                     selected_range=range_option)
        else:
            return render_template('analysis.html', 
                                 error="Database connection failed. Check MySQL configuration.",
                                 is_logged_in=True,
                                 selected_range=range_option)
    except Exception as e:
        print(f"Critical error in analysis route: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/download-report')
@login_required
def download_report():
    """Download all predictions as CSV report"""
    username = session.get('username')
    if not username:
        return redirect(url_for('login_page'))
    connection = get_db_connection()
    
    if connection:
        try:
            # Query all predictions data
            query = """
                SELECT id, filename, predicted_class, confidence, prediction_timestamp, image_path
                FROM predictions
                WHERE username = %s
                ORDER BY prediction_timestamp DESC
            """
            df = pd.read_sql(query, connection, params=(username,))
            connection.close()
            
            if df.empty:
                return "No prediction data available to download for this user.", 404
            
            # Create CSV in memory
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Convert to bytes
            csv_bytes = io.BytesIO()
            csv_bytes.write(csv_buffer.getvalue().encode('utf-8'))
            csv_bytes.seek(0)
            
            # Generate filename with timestamp
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
        return "Database connection failed", 500

@app.route('/result', methods=['POST', 'GET'])
@login_required
def result():
    if request.method == 'POST':
        f = request.files['image']
        username = session.get('username')
        if not username:
            return redirect(url_for('login_page'))
        
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        
        # Read and preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the image
        
        # Make predictions using the model
        predictions = model.predict(x)
        class_index = np.argmax(predictions, axis=1)[0]
        class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
        predicted_class = class_names[class_index]
        confidence = predictions[0][class_index]
        
        # Save prediction to MySQL database
        save_prediction_to_db(
            filename=f.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            image_path=filepath,
            username=username
        )
        
        result_text = f"Prediction: {predicted_class} with confidence {confidence:.2f}"
        
        return render_template('result.html', result=result_text, is_logged_in=True)
    
    return redirect(url_for('predict'))

if __name__ == '__main__':
    # Print all registered routes for debugging
    print("\n=== REGISTERED ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("========================\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="127.0.0.1", port=5000, debug=False)  # Disable reloader to prevent crashes