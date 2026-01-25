# CODE SNIPPETS - CRITICAL IMPLEMENTATIONS

## SNIPPET 1: DATABASE CONNECTION & PREDICTION SAVE

```python
# app.py lines 37-75

import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ammu@221025',
    'database': 'colon_disease_db'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def save_prediction_to_db(filename, predicted_class, confidence, image_path, username=None):
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
            print(f"✓ Prediction saved: {filename} -> {predicted_class} ({confidence:.4f})")
            cursor.close()
            connection.close()
            return True
        except Error as e:
            print(f"✗ Database insert error: {e}")
            return False
    return False
```

---

## SNIPPET 2: CNN INFERENCE & MYSQL SAVE (/result route)

```python
# app.py lines 347-388

@app.route('/result', methods=['POST', 'GET'])
@login_required
def result():
    if request.method == 'POST':
        f = request.files['image']
        
        # 1. SAVE UPLOADED IMAGE
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        
        # 2. PREPROCESS IMAGE (224×224)
        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize
        
        # 3. CNN PREDICTION (VGG16)
        predictions = model.predict(x)
        class_index = np.argmax(predictions, axis=1)[0]
        class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
        predicted_class = class_names[class_index]
        confidence = predictions[0][class_index]
        
        # 4. SAVE TO MYSQL
        username = session.get('username')
        save_prediction_to_db(
            filename=f.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            image_path=filepath,
            username=username
        )
        
        # 5. RENDER RESULT
        result_text = f"Prediction: {predicted_class} with confidence {confidence:.2f}"
        
        return render_template('result.html', result=result_text, is_logged_in=True)
    
    return redirect(url_for('predict'))
```

---

## SNIPPET 3: ANALYTICS QUERY WITH PANDAS

```python
# app.py lines 229-275

@app.route('/analysis')
@login_required
def analysis():
    try:
        connection = get_db_connection()
        
        if connection:
            # MYSQL QUERY WITH PANDAS (per-user)
            username = session.get('username')
            query = "SELECT predicted_class, confidence, prediction_timestamp FROM predictions WHERE username = %s"
            df = pd.read_sql(query, connection, params=(username,))
            connection.close()
            
            if df.empty:
                return render_template('analysis.html', 
                                     error="No predictions yet",
                                     is_logged_in=True)
            
            # AGGREGATE DATA
            all_classes = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
            
            class_counts = df['predicted_class'].value_counts()
            total_predictions = len(df)
            avg_confidence = df.groupby('predicted_class')['confidence'].mean()
            
            # PREPARE FOR CHART.JS
            chart_data = {
                'labels': all_classes,
                'counts': [int(class_counts.get(cls, 0)) for cls in all_classes],
                'avg_confidence': [float(avg_confidence.get(cls, 0)) for cls in all_classes]
            }
            
            stats = {
                'total_predictions': total_predictions,
                'class_distribution': class_counts.to_dict(),
                'avg_confidence': {k: f"{v:.4f}" for k, v in avg_confidence.to_dict().items()},
                'chart_data': json.dumps(chart_data)
            }
            
            return render_template('analysis.html',
                                   stats=stats,
                                   is_logged_in=True)
```

---

## SNIPPET 4: CSV EXPORT (Download Report)

```python
# app.py lines 277-325

@app.route('/download-report')
@login_required
def download_report():
    connection = get_db_connection()
    
    if connection:
        try:
            username = session.get('username')
            
            # QUERY ALL USER PREDICTIONS
            query = "SELECT id, filename, predicted_class, confidence, prediction_timestamp, image_path FROM predictions WHERE username = %s ORDER BY prediction_timestamp DESC"
            df = pd.read_sql(query, connection, params=(username,))
            connection.close()
            
            if df.empty:
                return "No prediction data to download", 404
            
            # CONVERT TO CSV IN MEMORY
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            csv_bytes = io.BytesIO()
            csv_bytes.write(csv_buffer.getvalue().encode('utf-8'))
            csv_bytes.seek(0)
            
            # SEND FILE
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
```

---

## SNIPPET 5: AUTHENTICATION DECORATOR

```python
# app.py lines 67-75

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return render_template('auth_required.html', redirect_page=f.__name__)
        return f(*args, **kwargs)
    return decorated_function

# USAGE:
@app.route('/analysis')
@login_required  # ← Protects route
def analysis():
    # This code only runs if user is logged in
    pass
```

---

## SNIPPET 6: CHART.JS INTEGRATION (analysis.html)

```html
<!-- templates/analysis.html -->

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<script>
    {% if stats and stats.chart_data %}
    const chartData = {{ stats.chart_data|safe }};
    
    const primaryBlueShades = [
        'rgba(30, 64, 175, 0.9)',      // Primary Blue
        'rgba(37, 99, 235, 0.9)',      // Lighter Blue
        'rgba(59, 130, 246, 0.9)',     // Light Blue
        'rgba(96, 165, 250, 0.9)',     // Even Lighter
    ];
    
    // CLASS DISTRIBUTION CHART
    const ctx1 = document.getElementById('classDistributionChart').getContext('2d');
    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: chartData.labels,  // ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
            datasets: [{
                label: 'Prediction Count',
                data: chartData.counts,  // [12, 5, 3, 8]
                backgroundColor: primaryBlueShades,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Predictions by Disease Class'
                },
                legend: { display: false }
            }
        }
    });
    
    // AVERAGE CONFIDENCE CHART
    const ctx2 = document.getElementById('confidenceChart').getContext('2d');
    new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Average Confidence',
                data: chartData.avg_confidence,  // [0.91, 0.87, 0.95, 0.89]
                backgroundColor: primaryBlueShades
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: {
                    max: 1.0,
                    ticks: {
                        callback: (val) => (val * 100).toFixed(0) + '%'
                    }
                }
            }
        }
    });
    {% endif %}
</script>
```

---

## SNIPPET 7: SQL QUERIES REFERENCE

```sql
-- Total predictions by user
SELECT username, COUNT(*) as total_predictions 
FROM predictions 
GROUP BY username;

-- Predictions by disease class
SELECT predicted_class, COUNT(*) as count, AVG(confidence) as avg_conf 
FROM predictions 
GROUP BY predicted_class;

-- Predictions from last 7 days
SELECT * FROM predictions 
WHERE prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY prediction_timestamp DESC;

-- High confidence predictions (>0.9)
SELECT * FROM predictions 
WHERE confidence > 0.9
ORDER BY confidence DESC;

-- User's complete history
SELECT * FROM predictions 
WHERE username = 'doctor123'
ORDER BY prediction_timestamp DESC;

-- Statistics for analysis page
SELECT 
    predicted_class,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    MAX(confidence) as max_confidence,
    MIN(confidence) as min_confidence
FROM predictions
WHERE username = 'doctor123'
GROUP BY predicted_class;
```

---

## SNIPPET 8: INSTALLATION VERIFICATION

```bash
# Check Python packages installed
pip list | findstr Flask mysql Pandas keras

# Expected output:
Flask                      3.1.2
mysql-connector-python     9.1.0
pandas                     2.2.3
keras                      3.8.0
numpy                      2.2.1
Pillow                     11.1.0

# Verify MySQL is running
mysql -u root -p -e "SELECT 1;"
# Should return: 1

# Verify database exists
mysql -u root -p -e "USE colon_disease_db; SHOW TABLES;"
# Should return: predictions, users

# Test Flask app import
python -c "import app; print('Flask app imported successfully')"
```

---

## PRODUCTION CHECKLIST

- [ ] Change `app.secret_key` to random string
- [ ] Update `DB_CONFIG` with production credentials
- [ ] Hash passwords: `from werkzeug.security import generate_password_hash`
- [ ] Set `debug=False` in `app.run()`
- [ ] Add CORS restrictions
- [ ] Enable HTTPS/SSL
- [ ] Setup logging: `import logging`
- [ ] Deploy with Gunicorn: `gunicorn -w 4 app:app`
- [ ] Configure database backups
- [ ] Setup monitoring/alerting
- [ ] Add rate limiting to login route
- [ ] Implement API documentation (Swagger)
