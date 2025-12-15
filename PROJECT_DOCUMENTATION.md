# ColonPredict - Complete Project Documentation

## 📋 Project Overview

**ColonPredict** is a web-based medical image classification system that uses deep learning to detect colon diseases from endoscopy images. It's built with Flask (Python web framework) and uses a pre-trained VGG16 neural network model to classify images into four categories.

---

## 🎯 Key Features

1. **Image Classification**: Predicts one of 4 colon disease conditions
2. **User-Friendly Web Interface**: Interactive website with navigation and image upload
3. **Real-time Predictions**: Instant results with confidence scores
4. **Responsive Design**: Works on desktop and mobile devices
5. **Deployment Ready**: Can be deployed to cloud platforms (Render, Heroku)

---

## 📊 Disease Classification Categories

The model can classify colon images into 4 categories:
- **Normal**: Healthy colon
- **Ulcerative Colitis**: Inflammatory bowel disease with ulcers
- **Polyps**: Abnormal tissue growths
- **Esophagitis**: Inflammation of the esophagus

---

## 🏗️ Project Architecture

```
Flask/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── model_1/
│   └── Vgg.h5                     # Pre-trained VGG16 model (~60MB)
├── static/
│   ├── css/
│   │   └── style.css              # Website styling
│   └── images/
│       ├── index.jpg              # Home page image
│       ├── about.jpg              # About page image
│       └── customer support.jpg   # Contact page image
├── templates/
│   ├── index.html                 # Home page
│   ├── about.html                 # About/Mission page
│   ├── details.html               # Image upload & prediction page
│   ├── contact.html               # Contact information page
│   └── result.html                # Prediction results page
└── uploads/                        # Stores uploaded images (temporary)
    └── test_*.jpg                  # Sample test images
```

---

## 🔧 Backend: app.py (Flask Application)

### Key Components:

#### 1. **Imports & Setup**
```python
import os
os.environ['KERAS_BACKEND'] = 'jax'  # Use JAX instead of TensorFlow
from flask import Flask, request, render_template, redirect, url_for
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
```
- Uses **JAX** as Keras backend (compatible with Python 3.11+)
- Flask for web routing
- Keras for deep learning model loading
- NumPy for numerical operations

#### 2. **Model Loading**
```python
model = load_model('model_1/Vgg.h5')
```
- Loads pre-trained VGG16 model from `model_1/Vgg.h5`
- VGG16 is a deep convolutional neural network with 16 layers
- Transfer learning: Uses weights trained on ImageNet dataset

#### 3. **Flask Routes**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/about` | GET | About page (mission & features) |
| `/predict` | GET | Image upload page |
| `/contact` | GET | Contact information |
| `/result` | POST/GET | Handle prediction & show results |

#### 4. **Image Prediction Flow** (POST to `/result`)

```
1. User uploads image (PNG/JPG/JPEG)
2. Image saved to uploads/ folder
3. Image resized to 224x224 pixels (VGG16 requirement)
4. Converted to numpy array
5. Normalized (pixel values 0-1)
6. Passed to model.predict()
7. argmax() gets highest confidence class
8. Result displayed with confidence percentage
```

#### 5. **Example Prediction**
```
Input: Colon endoscopy image
↓
Processing: Resize to 224x224, normalize
↓
Model Output: [0.05, 0.85, 0.08, 0.02] (probabilities for each class)
↓
Prediction: "Ulcerative Colitis" with confidence 85%
```

---

## 🎨 Frontend: HTML Templates

### 1. **index.html** - Home Page
- **Navigation bar** with logo and links
- **Hero section** with project title and image
- **Call-to-action button** ("GET STARTED")
- Links to: Home, About, Predictions, Contact

**Key Elements:**
```html
<h1>ColoPredict</h1>
<p>Revolutionising Colon prediction through transfer learning mastery.</p>
<img src="images/index.jpg" class="hero-image">
```

### 2. **about.html** - About Page
- **Mission Statement**: Explains ColonPredict's purpose
- **What We Do**: Describes the technology
- **Image** showcasing the service
- **Navigation buttons**: Next to predictions, Back to home

**Content:**
- Mission: Provide accessible, efficient colon health assessments
- Technology: State-of-the-art deep learning for image analysis
- Goal: Early detection and diagnosis support

### 3. **details.html** - Prediction Page
- **File upload form** for image selection
- **Live preview** of selected image (JavaScript)
- **Submit button** to trigger prediction
- **Form details:**
  - Accepts: .png, .jpg, .jpeg
  - Method: POST to `/result` endpoint
  - multipart/form-data encoding for file upload

**JavaScript Preview:**
```javascript
var loadFile = function(event) {
    var image = document.getElementById('out');
    image.src = URL.createObjectURL(event.target.files[0]);
};
```
Allows users to see image before uploading

### 4. **result.html** - Results Page
- **Prediction result** with confidence percentage
- **Formatted output example:**
  ```
  "Prediction: Ulcerative Colitis with confidence 0.85"
  ```
- **Information section** with medical resource links
- **Book Appointment button** linking to Medtronic's colon disease info
- **Navigation buttons**: Back to predictions, Home

**Styling:**
- White text on colored background
- Centered layout for readability
- Responsive button styling with hover effects

### 5. **contact.html** - Contact Page
- **Contact information:**
  - Email: colonpredict@team.com
  - Phone: +91-9845xxxxxx
- **Customer support image**
- **Feedback message** encouraging user interaction

---

## 🎯 CSS Styling (style.css)

### Design Features:

#### Color Scheme:
- **Primary Color**: #089CB0 (Teal/Cyan)
- **Text Colors**: White (#fff), Black (#000)
- **Accent Colors**: Blue (#008CBA)

#### Layout:
```css
* {
    background: #089CB0;
    font-family: 'Trebuchet MS', Arial, sans-serif;
}
```

#### Navigation Bar:
- Fixed height: 80px
- Flexbox layout for centering
- Logo styled in italic bold
- Hover effects on links (underline + color change)

#### Hero Section:
- 2-column grid layout
- Left column: Text content
- Right column: Images
- Full viewport height (95vh)
- Responsive: Stacks to 1 column on mobile (<768px)

#### Images:
```css
.hero-image {
    width: 100%;
    height: auto;
    border-radius: 2px;  /* Slightly curved corners */
}
```

#### Buttons:
```css
button {
    padding: 1rem 3rem;
    background: #000;
    color: #fff;
    border-radius: 50px;  /* Fully rounded */
}

button:hover {
    background: #fff;
    color: #000;
}
```

#### Forms:
- File input with custom styling
- Submit buttons with hover effects
- Dynamic image preview

---

## 📦 Dependencies (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.1.2 | Web framework |
| keras | 3.12.0 | Deep learning API |
| numpy | 2.3.5 | Numerical computing |
| Pillow | 12.0.0 | Image processing |
| jax[cpu] | 0.8.1 | Keras backend (fast computation) |
| jaxlib | 0.8.1 | JAX library for CPU |
| gunicorn | 21.2.0 | Production web server |
| h5py | 3.15.1 | Read/write HDF5 files (model format) |

---

## 🚀 How It Works - Complete User Journey

### Step 1: User Visits Home Page
```
User visits: http://localhost:5000/
↓
Flask serves: templates/index.html
↓
Page displays with navigation and "GET STARTED" button
```

### Step 2: Navigate to Prediction
```
User clicks "GET STARTED" or "Predictions" nav link
↓
Routes to: /predict
↓
Serves: templates/details.html
↓
Shows file upload form
```

### Step 3: Upload & Preview Image
```
User selects image (.jpg, .png, .jpeg)
↓
JavaScript `loadFile()` function triggers
↓
Image preview displays in browser (no upload yet)
↓
User clicks "Predict" button
```

### Step 4: Backend Processing
```
POST request to: /result
↓
app.py:
  1. Receives uploaded file
  2. Saves to: uploads/filename.jpg
  3. Loads image with Keras
  4. Resizes to 224x224 (VGG input requirement)
  5. Converts to numpy array
  6. Normalizes pixel values (0-255 → 0-1)
↓
Model Prediction:
  7. Passes through VGG16 model
  8. Gets 4 probability scores
  9. Finds maximum: argmax()
  10. Maps to class name
↓
Generates result_text:
"Prediction: [Disease] with confidence [X%]"
```

### Step 5: Display Results
```
Renders: templates/result.html
↓
Shows:
  - "RESULTS" heading
  - Prediction with confidence
  - Medical resource link
  - "Book Appointment" button
  - Navigation options
```

### Step 6: Next Steps
```
User can:
- Navigate back to predict another image
- Visit contact page for support
- Visit home page to navigate elsewhere
- Click "Book Appointment" for medical services
```

---

## 🤖 AI Model Details

### VGG16 Architecture:
- **Input**: 224×224 RGB image
- **Layers**: 16 convolutional/pooling layers
- **Output**: 4 classes (disease categories)
- **Model Size**: ~60MB (Vgg.h5)
- **Weights**: Pre-trained on ImageNet (millions of images)
- **Transfer Learning**: Fine-tuned for colon disease classification

### Model Training Pipeline:
1. **Data**: Thousands of colon endoscopy images
2. **Preprocessing**: 
   - Resizing to 224×224
   - Normalization
   - Data augmentation (rotation, flip, zoom)
3. **Training**: Transfer learning with ImageNet weights
4. **Validation**: Testing on unseen images
5. **Optimization**: Fine-tuning VGG16 final layers

---

## 📁 Data Flow

```
User Input (Image)
        ↓
  Upload Form (HTML)
        ↓
  Flask Backend (app.py)
        ↓
  Image Preprocessing:
  - Resize 224×224
  - Normalize 0-1
  - Convert to array
        ↓
  Keras Model (VGG16)
        ↓
  Prediction (4 probabilities)
        ↓
  Post-processing:
  - argmax() → class index
  - class_names mapping
  - confidence calculation
        ↓
  HTML Result Template
        ↓
  User Sees Prediction
```

---

## 🔐 Security & Validation

### Image Upload:
- **Accepted formats**: .png, .jpg, .jpeg (client-side validation)
- **Storage**: Temporary uploads/ folder
- **Cleanup**: Images can be deleted after prediction
- **Validation**: File type checked before processing

### Error Handling:
- Creates uploads/ directory if missing: `os.makedirs(upload_folder, exist_ok=True)`
- Handles missing files gracefully
- Redirects invalid requests

---

## 📡 Deployment

### Local Development:
```bash
python app.py
# Runs on http://localhost:5000
```

### Production (Render):
```
Procfile: gunicorn app:app
Command: gunicorn starts WSGI server
Port: Environment variable PORT
```

### Environment Variables:
```python
port = int(os.environ.get('PORT', 5000))
```
- Default: 5000 (local)
- On Render: Set by platform (typically 10000+)

---

## 📊 Sample Predictions

### Example 1:
```
Input: Normal colon image
↓
Model Output: [0.92, 0.04, 0.02, 0.02]
↓
Display: "Prediction: Normal with confidence 0.92"
```

### Example 2:
```
Input: Ulcerative colitis image
↓
Model Output: [0.05, 0.88, 0.04, 0.03]
↓
Display: "Prediction: Ulcerative Colitis with confidence 0.88"
```

---

## 🎓 Technologies Used

### Backend:
- **Python 3.11**: Programming language
- **Flask**: Web framework (routing, request handling)
- **Keras 3**: High-level deep learning API
- **JAX**: Fast numerical computation (backend)
- **NumPy**: Array operations
- **Pillow**: Image processing/loading
- **H5PY**: HDF5 file format (model storage)

### Frontend:
- **HTML5**: Structure
- **CSS3**: Styling & layout (Flexbox, Grid)
- **JavaScript**: Image preview functionality
- **Jinja2**: Template rendering (Flask templates)

### Deployment:
- **Git**: Version control
- **GitHub**: Repository hosting
- **Render**: Cloud deployment platform
- **Gunicorn**: WSGI application server

---

## ✅ Checklist - How to Use

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run locally: `python app.py`
3. ✅ Visit: http://localhost:5000
4. ✅ Navigate to Predictions
5. ✅ Upload colon endoscopy image
6. ✅ View prediction results
7. ✅ Deploy to Render for free hosting

---

## 🔮 Future Enhancements

1. **User Authentication**: Save prediction history
2. **Multiple Models**: Compare predictions from different architectures
3. **Confidence Threshold**: Alert for low-confidence predictions
4. **Image Analysis**: Show which areas model focused on (attention maps)
5. **Doctor Dashboard**: For healthcare professionals
6. **Mobile App**: Native iOS/Android app
7. **API Endpoint**: RESTful API for third-party integration
8. **Batch Processing**: Process multiple images at once

---

## 📞 Support & Contact

- **Email**: colonpredict@team.com
- **Phone**: +91-9845xxxxxx
- **Medical Resources**: https://www.medtronic.com/

---

## 📄 License & Credits

**Project**: ColonPredict - Colon Disease Prediction using Deep Learning
**Author**: Amrutha0902
**Repository**: WCE_CURATED_COLON_DISEASE_PREDICTION
**Tech Stack**: Python, Flask, Keras, JAX

---

**End of Documentation**
