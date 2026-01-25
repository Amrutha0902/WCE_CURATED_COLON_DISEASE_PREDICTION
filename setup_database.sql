-- ==========================================
-- COLON DISEASE PREDICTION DATABASE SETUP
-- ==========================================
-- Database: colon_disease_db
-- Tables: predictions, users
-- Purpose: Store CNN predictions + Doctor authentication
-- ==========================================

-- Create database
CREATE DATABASE IF NOT EXISTS colon_disease_db;

-- Use the database
USE colon_disease_db;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(500),
    INDEX idx_class (predicted_class),
    INDEX idx_timestamp (prediction_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create users table for doctor authentication
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'doctor',
    full_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default doctor credentials
-- Username: doctor123, Password: password123
INSERT INTO users (username, password, role, full_name) 
VALUES ('doctor123', 'password123', 'doctor', 'Dr. Demo Account')
ON DUPLICATE KEY UPDATE username=username;

-- Verify table creation
DESCRIBE predictions;
DESCRIBE users;

-- Display success message
SELECT 'Database setup complete! Tables: predictions, users' AS Status;
SELECT 'Default doctor login - Username: doctor123, Password: password123' AS Credentials;
