#!/bin/bash
# ============================================================================
# COLON DISEASE PREDICTION - QUICK START TERMINAL COMMANDS
# ============================================================================

# SECTION 1: ENVIRONMENT & DEPENDENCIES
# ============================================================================

echo "=== STEP 1: Create Virtual Environment ==="
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

echo "=== STEP 2: Install Python Packages ==="
pip install -r requirements.txt

# Verify installations:
pip list | findstr Flask mysql-connector-python pandas keras


# SECTION 2: DATABASE SETUP
# ============================================================================

echo "=== STEP 3: Start MySQL Server ==="
# Windows:
net start MySQL80
# Mac:
# brew services start mysql
# Linux:
# sudo systemctl start mysql

echo "=== STEP 4: Create Database & Tables ==="
mysql -u root -p < setup_database.sql
# Enter password when prompted

echo "=== STEP 5: Verify Database ==="
mysql -u root -p -e "USE colon_disease_db; SHOW TABLES; SELECT * FROM users;"


# SECTION 3: APPLICATION EXECUTION
# ============================================================================

echo "=== STEP 6: Run Flask Application ==="
python app.py

# Expected Output:
# Model Loaded!
# === REGISTERED ROUTES ===
# [route list]
# * Running on http://127.0.0.1:5000

# Open browser: http://127.0.0.1:5000


# SECTION 4: TEST WORKFLOW
# ============================================================================

echo "=== TESTING WORKFLOW ==="

# 1. Login to home page
# URL: http://127.0.0.1:5000
# Username: doctor123
# Password: password123

# 2. Go to /predict
# URL: http://127.0.0.1:5000/predict
# Upload: Any JPG/PNG image

# 3. View /result
# Automatic redirect after upload
# Shows: Prediction class + Confidence %

# 4. Check /analysis dashboard
# URL: http://127.0.0.1:5000/analysis
# Filters: Today, Week, Month, 6 Months, Year, Till Date
# Charts: Disease distribution, Average confidence

# 5. Download CSV report
# Button: "Download CSV Report"
# File: colon_disease_predictions_YYYYMMDD_HHMMSS.csv


# SECTION 5: MYSQL VERIFICATION QUERIES
# ============================================================================

echo "=== VERIFY DATA IN MYSQL ==="

# Check users table:
# mysql> SELECT * FROM users;

# Check predictions table (total records):
# mysql> SELECT COUNT(*) as total_predictions FROM predictions;

# Check predictions per user:
# mysql> SELECT username, COUNT(*) as count FROM predictions GROUP BY username;

# Check predictions by disease class:
# mysql> SELECT predicted_class, COUNT(*) as count, AVG(confidence) as avg_conf FROM predictions GROUP BY predicted_class;

# Check recent predictions (last 7 days):
# mysql> SELECT * FROM predictions WHERE prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY);


# SECTION 6: TROUBLESHOOTING
# ============================================================================

echo "=== TROUBLESHOOTING COMMANDS ==="

# Kill existing Python process:
# Windows PowerShell:
Get-Process -Name python | Stop-Process -Force

# Check if MySQL is running:
# Windows:
tasklist | findstr mysql
# Mac:
ps aux | grep mysql

# Check port 5000 is available:
# Windows:
netstat -ano | findstr :5000
# Mac/Linux:
lsof -i :5000

# View Flask debug logs:
# Already printed to console from python app.py

# Check database connection:
mysql -u root -p -h localhost -e "SELECT 'Connection Successful';"


# SECTION 7: DEPLOYMENT NOTES
# ============================================================================

echo "=== PRODUCTION DEPLOYMENT ==="

# Stop debug mode in app.py (line ~396):
# Change: debug=True, use_reloader=False
# To: debug=False, use_reloader=False

# Run with Gunicorn (production WSGI server):
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or run with uWSGI:
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app


# SECTION 8: GIT COMMANDS
# ============================================================================

echo "=== GIT VERSION CONTROL ==="

# Check status:
git status

# Add changes:
git add .

# Commit changes:
git commit -m "Complete implementation: auth, predictions, analytics"

# Push to GitHub:
git push origin main

# View commit history:
git log --oneline


# ============================================================================
# END OF QUICK START GUIDE
# ============================================================================
