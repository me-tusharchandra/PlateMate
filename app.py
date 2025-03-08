import os
import cv2
import json
import requests
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
# from pyzbar.pyzbar import decode
import google.generativeai as genai
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash, Response
import base64

# 1. Load Environment Variables ----------------------------------------
load_dotenv()

# 2. Configure Gemini API ----------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# 3. Flask + SQLAlchemy Setup ------------------------------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///platemate.db'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "platemate_secret_key")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 4. Upload Configuration ----------------------------------------------
UPLOAD_FOLDER = 'uploads'  # Directory to temporarily store uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file types
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 5. Suggested Values for UI -------------------------------------------
SUGGESTED_ALLERGIES = [
    "Peanuts", "Tree Nuts", "Milk", "Eggs", "Fish", "Shellfish", "Soy", 
    "Wheat", "Gluten", "Sesame", "Mustard", "Celery", "Lupin", "Sulfites",
    "Corn", "Nightshades", "Citrus", "Chocolate", "Alcohol"
]

SUGGESTED_HEALTH_CONDITIONS = [
    "Diabetes", "Hypertension", "Heart Disease", "Celiac Disease", 
    "IBS", "Crohn's Disease", "Ulcerative Colitis", "GERD", "Kidney Disease",
    "Gout", "Lactose Intolerance", "Histamine Intolerance", "Insulin Resistance",
    "High Cholesterol", "Fatty Liver Disease", "Thyroid Disorders"
]

# 6. Database Models --------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    mobile = db.Column(db.String(20), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    allergies = db.Column(db.String(255), default="")
    health_conditions = db.Column(db.String(255), default="")
    
    # Relationship with saved products
    saved_products = db.relationship('SavedProduct', backref='user', lazy=True)

class SavedProduct(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    barcode = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100))
    category = db.Column(db.String(100))
    is_safe = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_added = db.Column(db.DateTime, default=db.func.current_timestamp())

class AlternativeProduct(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    barcode = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100))
    category = db.Column(db.String(100))
    for_allergy = db.Column(db.String(100))
    for_condition = db.Column(db.String(100))

# Create database tables
with app.app_context():
    db.create_all()

# 7. Helper Functions -------------------------------------------------
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_barcode(image_path=None, frame=None):
    """
    Reads barcode or QR code from an image file or directly from a video frame.
    Returns a list of tuples (barcode_data, barcode_type) if found, else None.
    """
    try:
        # If we're processing a file
        if image_path and not frame:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image from {image_path}")
                return None
        # If we're processing a video frame
        elif frame is not None:
            image = frame
            if image is None or image.size == 0:
                print("Received empty frame")
                return None
        else:
            print("No image path or frame provided")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try to use pyzbar if available (most reliable)
        try:
            from pyzbar.pyzbar import decode
            barcodes = decode(image)
            if barcodes:
                return [(barcode.data.decode('utf-8'), barcode.type) for barcode in barcodes]
        except ImportError:
            print("pyzbar not available, falling back to OpenCV")
        except Exception as e:
            print(f"Error using pyzbar: {str(e)}")
        
        # Initialize the cv2 QRCode detector
        qr_detector = cv2.QRCodeDetector()
        
        # Detect and decode QR code
        data, bbox, _ = qr_detector.detectAndDecode(gray)
        
        # If QR code is detected
        if data:
            print(f"QR code detected: {data}")
            return [(data, "QR-Code")]
            
        # Enhanced OpenCV-based barcode detection methods
        
        # Method 1: Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Method 2: Try Canny edge detection with better parameters
        edges = cv2.Canny(gray, 50, 200, apertureSize=3)
        
        # Method 3: Try different blurring techniques
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 4: Morphological operations to enhance barcode features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Method 5: Sobel edge detection (good for barcodes)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, sobel_binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Combine methods for contour detection
        for img in [thresh, edges, binary, morph, sobel_binary]:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on area and aspect ratio to find potential barcode regions
            potential_barcodes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Barcodes typically have a specific aspect ratio and minimum area
                if area > 1000 and (aspect_ratio > 2.0 or aspect_ratio < 0.5):
                    potential_barcodes.append((x, y, w, h))
            
            # If we found potential barcodes, try to decode them
            if potential_barcodes:
                print(f"Found {len(potential_barcodes)} potential barcode regions")
                
                # For each potential barcode region, try to extract and decode
                for x, y, w, h in potential_barcodes:
                    # Add padding around the region
                    padding = 10
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(gray.shape[1], x + w + padding)
                    y_end = min(gray.shape[0], y + h + padding)
                    
                    # Extract the region
                    roi = gray[y_start:y_end, x_start:x_end]
                    
                    # Try to enhance the region for better detection
                    _, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Try to detect lines in the region (barcodes have parallel lines)
                    lines = cv2.HoughLinesP(roi_binary, 1, np.pi/180, 50, minLineLength=w/3, maxLineGap=10)
                    
                    # If we found enough parallel lines, it's likely a barcode
                    if lines is not None and len(lines) > 5:
                        # For now, we'll return a manual entry required
                        # In a production app, you might want to implement a more sophisticated
                        # barcode decoding algorithm here
                        return [("MANUAL_ENTRY_REQUIRED", "UNKNOWN")]
        
        # If we have too many potential barcodes, it's likely not a barcode image
        if len(contours) > 5:  # Arbitrary threshold for potential barcode presence
            return [("MANUAL_ENTRY_REQUIRED", "UNKNOWN")]
        
        return None
        
    except Exception as e:
        import traceback
        print(f"Error reading barcode: {str(e)}")
        print(traceback.format_exc())
        return None

def get_product_from_openfoodfacts(barcode):
    """
    Fetches product details using the OpenFoodFacts API.
    Returns a dictionary with product information if found, else None.
    """
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "product" in data and data["product"]:
            product = data["product"]
            title = product.get("product_name", "No product title found")
            brand = product.get("brands", "Unknown brand")
            description = product.get("generic_name", "No description available")
            ingredients = product.get("ingredients_text", "") or ""
            category = product.get("categories", "Unknown category")
            image_url = product.get("image_url", "")
            nutriments = product.get("nutriments", {})
            
            # Extract allergen information
            allergens = product.get("allergens_tags", [])
            allergens_from_ingredients = product.get("allergens_from_ingredients", "")
            
            # Extract ingredient analysis information
            ingredients_analysis = product.get("ingredients_analysis_tags", [])
            
            # Extract detailed ingredients list
            ingredients_list = []
            if "ingredients" in product and isinstance(product["ingredients"], list):
                for ingredient in product["ingredients"]:
                    ingredients_list.append({
                        "id": ingredient.get("id", ""),
                        "text": ingredient.get("text", ""),
                        "percent": ingredient.get("percent_estimate", 0),
                        "vegan": ingredient.get("vegan", "unknown"),
                        "vegetarian": ingredient.get("vegetarian", "unknown"),
                        "from_palm_oil": ingredient.get("from_palm_oil", "unknown")
                    })
            
            # Extract nutrition grade and eco-score
            nutrition_grade = product.get("nutriscore_grade", "")
            eco_score = product.get("ecoscore_grade", "")
            
            # Extract traces information (may contain traces of)
            traces = product.get("traces_tags", [])
            
            return {
                "barcode": barcode,
                "title": title,
                "brand": brand,
                "description": description,
                "ingredients": ingredients,
                "ingredients_list": ingredients_list,
                "category": category,
                "image_url": image_url,
                "nutriments": nutriments,
                "allergens": allergens,
                "allergens_from_ingredients": allergens_from_ingredients,
                "ingredients_analysis": ingredients_analysis,
                "nutrition_grade": nutrition_grade,
                "eco_score": eco_score,
                "traces": traces
            }
    return None

def analyze_product_with_gemini(product_info, user_profile):
    """
    Uses Gemini 2.0 to analyze product safety based on user's allergies and health conditions.
    Returns a detailed analysis with safety assessment and recommendations.
    """
    if not product_info or not user_profile:
        return {"status": "error", "message": "Invalid product or user information."}
    
    # Format the prompt for Gemini
    prompt = f"""
    Analyze this food product for dietary safety based on the user's profile:
    
    PRODUCT INFORMATION:
    - Name: {product_info['title']}
    - Brand: {product_info['brand']}
    - Category: {product_info['category']}
    - Ingredients: {product_info['ingredients']}
    - Nutrition Facts: {json.dumps(product_info.get('nutriments', {}))}
    - Nutrition Grade: {product_info.get('nutrition_grade', 'Not available')}
    - Eco Score: {product_info.get('eco_score', 'Not available')}
    
    ALLERGEN INFORMATION:
    - Declared Allergens: {', '.join(product_info.get('allergens', []))}
    - Allergens from Ingredients: {product_info.get('allergens_from_ingredients', 'None')}
    - May Contain Traces of: {', '.join(product_info.get('traces', []))}
    - Ingredient Analysis: {', '.join(product_info.get('ingredients_analysis', []))}
    
    USER PROFILE:
    - Age: {user_profile['age']}
    - Allergies: {user_profile['allergies']}
    - Health Conditions: {user_profile['health_conditions']}
    
    Please provide:
    1. Is this product safe for the user to consume? (Yes/No/Caution)
    2. Detailed explanation of why it is safe or unsafe
    3. List any ingredients that conflict with the user's allergies or health conditions
    4. Suggest 3-5 alternative products or ingredients that would be safer for this user
    5. Any additional dietary advice specific to this user's profile
    
    Format your response as a JSON object with these keys: 
    "is_safe" (boolean), 
    "safety_level" (string: "Safe", "Caution", or "Unsafe"),
    "explanation" (string), 
    "conflicting_ingredients" (array of strings), 
    "alternatives" (array of strings),
    "dietary_advice" (string)
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        
        # Parse the response - assuming Gemini returns valid JSON
        try:
            # First try to extract JSON if it's wrapped in text
            response_text = response.text
            # Look for JSON pattern between curly braces
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
            else:
                # If no JSON pattern found, try parsing the whole response
                analysis = json.loads(response_text)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from the text
            analysis = {
                "is_safe": "Caution" not in response.text and "Unsafe" not in response.text,
                "safety_level": "Safe" if "Safe" in response.text else "Unsafe" if "Unsafe" in response.text else "Caution",
                "explanation": response.text[:500],  # Truncate if too long
                "conflicting_ingredients": [],
                "alternatives": [],
                "dietary_advice": "Please consult with a healthcare professional."
            }
        
        return analysis
        
    except Exception as e:
        return {
            "is_safe": False,
            "safety_level": "Error",
            "explanation": f"Error analyzing product: {str(e)}",
            "conflicting_ingredients": [],
            "alternatives": [],
            "dietary_advice": "Please try again or consult with a healthcare professional."
        }

def find_alternative_products(product_info, user_profile, analysis):
    """
    Search for alternative products in the database or OpenFoodFacts API
    based on the user's dietary restrictions and the current product category.
    """
    alternatives = []
    seen_products = set()  # Track unique products by title+brand combination
    
    # First check our database for known alternatives
    if 'conflicting_ingredients' in analysis and analysis['conflicting_ingredients']:
        for ingredient in analysis['conflicting_ingredients']:
            db_alternatives = AlternativeProduct.query.filter_by(
                for_allergy=ingredient.lower()
            ).limit(3).all()
            
            for alt in db_alternatives:
                # Create a unique identifier for this product
                product_key = f"{alt.title.lower()}|{alt.brand.lower()}"
                if product_key not in seen_products:
                    seen_products.add(product_key)
                    alternatives.append({
                        "barcode": alt.barcode,
                        "title": alt.title,
                        "brand": alt.brand,
                        "category": alt.category,
                        "reason": f"Safe alternative for {ingredient} allergy"
                    })
    
    # Also check for alternatives based on detected allergens
    user_allergies = [allergy.strip().lower() for allergy in user_profile['allergies'] if allergy.strip()]
    product_allergens = [allergen.replace('en:', '') for allergen in product_info.get('allergens', [])]
    
    for allergen in product_allergens:
        if any(user_allergy in allergen.lower() for user_allergy in user_allergies):
            db_alternatives = AlternativeProduct.query.filter_by(
                for_allergy=allergen.lower()
            ).limit(2).all()
            
            for alt in db_alternatives:
                product_key = f"{alt.title.lower()}|{alt.brand.lower()}"
                if product_key not in seen_products:
                    seen_products.add(product_key)
                    alternatives.append({
                        "barcode": alt.barcode,
                        "title": alt.title,
                        "brand": alt.brand,
                        "category": alt.category,
                        "reason": f"Safe alternative for {allergen} allergy"
                    })
    
    # If we don't have enough alternatives, search OpenFoodFacts
    if len(alternatives) < 3 and product_info.get('category'):
        # Get the category and search for products in the same category
        category = product_info['category'].split(',')[0].strip()
        
        # Create a search query for OpenFoodFacts
        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={category}&search_simple=1&action=process&json=1"
        
        try:
            response = requests.get(search_url)
            if response.status_code == 200:
                data = response.json()
                if 'products' in data and data['products']:
                    # Filter products that don't contain the conflicting ingredients
                    conflicting_ingredients = analysis.get('conflicting_ingredients', [])
                    user_allergens = [allergen.lower() for allergen in user_allergies]
                    
                    for product in data['products'][:15]:  # Check more products
                        ingredients_text = product.get('ingredients_text', '').lower()
                        product_allergens = [allergen.replace('en:', '').lower() for allergen in product.get('allergens_tags', [])]
                        
                        is_safe = True
                        
                        # Check for conflicting ingredients
                        for ingredient in conflicting_ingredients:
                            if ingredient.lower() in ingredients_text:
                                is_safe = False
                                break
                        
                        # Check for user allergens
                        for allergen in product_allergens:
                            if any(user_allergen in allergen for user_allergen in user_allergens):
                                is_safe = False
                                break
                        
                        if is_safe and product.get('product_name'):
                            product_key = f"{product.get('product_name', '').lower()}|{product.get('brands', '').lower()}"
                            if product_key not in seen_products:
                                seen_products.add(product_key)
                                alternatives.append({
                                    "barcode": product.get('code', ''),
                                    "title": product.get('product_name', ''),
                                    "brand": product.get('brands', ''),
                                    "category": product.get('categories', ''),
                                    "reason": "Safe alternative from the same category"
                                })
                            
                            if len(alternatives) >= 5:
                                break
        except Exception as e:
            print(f"Error searching for alternatives: {str(e)}")
    
    return alternatives[:5]  # Return at most 5 alternatives

# 8. Routes -----------------------------------------------------------

# Home route
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/scan_camera')
def scan_camera():
    """
    Render the camera-based barcode scanning page.
    """
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    # Check if pyzbar is available
    try:
        import importlib.util
        pyzbar_spec = importlib.util.find_spec('pyzbar')
        has_pyzbar = pyzbar_spec is not None
        if has_pyzbar:
            print("pyzbar is available and will be used for barcode detection")
        else:
            print("pyzbar is not available, will use OpenCV fallback")
    except ImportError:
        has_pyzbar = False
        print("Error importing pyzbar module")
    
    # Check if OpenCV is properly configured for barcode detection
    has_good_cv_detection = True
    try:
        # Verify QRCodeDetector is available
        detector = cv2.QRCodeDetector()
        print("OpenCV QRCodeDetector is available")
    except Exception as e:
        print(f"OpenCV QRCodeDetector error: {str(e)}")
        has_good_cv_detection = False
    
    # Log information for debugging
    print(f"Camera scan page accessed. Pyzbar available: {has_pyzbar}, OpenCV detection: {has_good_cv_detection}")
    print(f"User agent: {request.user_agent}")
    
    return render_template('scan_camera.html', 
                          has_pyzbar=has_pyzbar, 
                          has_good_cv_detection=has_good_cv_detection)

@app.route('/camera_test')
def camera_test():
    """
    Simple camera test page to verify camera access works properly.
    """
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    print(f"Camera test page accessed. User agent: {request.user_agent}")
    
    return render_template('camera_test.html')

# User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            name = request.form.get('name')
            mobile = request.form.get('mobile')
            age = request.form.get('age')

            if not all([username, name, mobile, age]):
                flash("Username, Name, Mobile, and Age are required fields.", "danger")
                return redirect(url_for('register'))

            # Check if username or mobile already exists
            if User.query.filter_by(username=username).first():
                flash("Username already exists. Please choose a different one.", "warning")
                return redirect(url_for('register'))
                
            if User.query.filter_by(mobile=mobile).first():
                flash("Mobile number already exists. Try logging in instead.", "warning")
                return redirect(url_for('register'))

            # Process allergies and health conditions
            allergies = request.form.getlist('allergies')
            custom_allergy = request.form.get('custom_allergy')
            health_conditions = request.form.getlist('health_conditions')
            custom_health = request.form.get('custom_health')

            if custom_allergy:
                allergies.append(custom_allergy)
            if custom_health:
                health_conditions.append(custom_health)

            # Create new user
            new_user = User(
                username=username,
                name=name,
                mobile=mobile,
                age=int(age),
                allergies=", ".join(allergies),
                health_conditions=", ".join(health_conditions)
            )

            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))

        except IntegrityError:
            db.session.rollback()
            flash("Registration failed. Please try again with different information.", "danger")
        except Exception as e:
            db.session.rollback()
            flash(f"An unexpected error occurred: {str(e)}", "danger")

    return render_template(
        'register.html', 
        suggested_allergies=SUGGESTED_ALLERGIES, 
        suggested_health=SUGGESTED_HEALTH_CONDITIONS
    )

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        mobile = request.form.get('mobile')
        username = request.form.get('username')
        
        if not (mobile or username):
            flash("Please provide either mobile number or username.", "danger")
            return redirect(url_for('login'))
            
        # Try to find user by mobile or username
        user = None
        if mobile:
            user = User.query.filter_by(mobile=mobile).first()
        elif username:
            user = User.query.filter_by(username=username).first()
            
        if user:
            session['user_id'] = user.id
            session['mobile'] = user.mobile
            session['username'] = user.username
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("User not found. Please register first.", "danger")
            
    return render_template('login.html')

# User Logout
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('home'))

# User Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access your dashboard.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
        
    # Get user's recently scanned products
    recent_products = SavedProduct.query.filter_by(user_id=user.id).order_by(SavedProduct.date_added.desc()).limit(5).all()
    
    return render_template('dashboard.html', user=user, recent_products=recent_products)

# Update User Profile
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash("Please log in to access your profile.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Update basic info
            user.name = request.form.get('name', user.name)
            user.age = int(request.form.get('age', user.age))
            
            # Process allergies and health conditions
            allergies = request.form.getlist('allergies')
            custom_allergy = request.form.get('custom_allergy')
            health_conditions = request.form.getlist('health_conditions')
            custom_health = request.form.get('custom_health')

            if custom_allergy:
                allergies.append(custom_allergy)
            if custom_health:
                health_conditions.append(custom_health)
                
            user.allergies = ", ".join(allergies)
            user.health_conditions = ", ".join(health_conditions)
            
            db.session.commit()
            flash("Profile updated successfully!", "success")
            return redirect(url_for('profile'))
            
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {str(e)}", "danger")
    
    # For GET request, prepare the form
    user_allergies = [a.strip() for a in user.allergies.split(",")] if user.allergies else []
    user_health = [h.strip() for h in user.health_conditions.split(",")] if user.health_conditions else []
    
    return render_template(
        'profile.html', 
        user=user,
        user_allergies=user_allergies,
        user_health=user_health,
        suggested_allergies=SUGGESTED_ALLERGIES,
        suggested_health=SUGGESTED_HEALTH_CONDITIONS
    )

# Scan Barcode
@app.route('/scan', methods=['GET', 'POST'])
def scan_barcode():
    if 'user_id' not in session:
        flash("Please log in to scan products.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image_file' not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)
            
        file = request.files['image_file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash("No selected file.", "danger")
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the barcode
            results = read_barcode(filepath)
            
            if not results:
                flash("No barcode detected in the uploaded image.", "danger")
                os.remove(filepath)  # Clean up
                return redirect(request.url)
                
            barcode_data, barcode_type = results[0]
            product_info = get_product_from_openfoodfacts(barcode_data)
            
            os.remove(filepath)  # Clean up
            
            if not product_info:
                flash(f"No product found for barcode {barcode_data}.", "warning")
                return redirect(request.url)
                
            # Prepare user profile for Gemini analysis
            user_profile = {
                "age": user.age,
                "allergies": user.allergies,
                "health_conditions": user.health_conditions
            }
            
            # Analyze with Gemini
            analysis = analyze_product_with_gemini(product_info, user_profile)
            
            # Find alternative products
            alternatives = find_alternative_products(product_info, user_profile, analysis)
            
            # Save the product to user's history
            try:
                saved_product = SavedProduct(
                    barcode=barcode_data,
                    title=product_info['title'],
                    brand=product_info['brand'],
                    category=product_info['category'],
                    is_safe=analysis.get('is_safe', False),
                    user_id=user.id
                )
                db.session.add(saved_product)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Error saving product: {str(e)}")
            
            return render_template(
                'scan_result.html',
                barcode=barcode_data,
                barcode_type=barcode_type,
                product=product_info,
                analysis=analysis,
                alternatives=alternatives,
                user=user
            )
        else:
            flash("Unsupported file type. Please upload a PNG, JPG, or JPEG image.", "danger")
            return redirect(request.url)
    
    return render_template('scan.html')

# Manual Barcode Entry
@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    if 'user_id' not in session:
        flash("Please log in to check products.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        barcode = request.form.get('barcode')
        
        if not barcode:
            flash("Please enter a barcode.", "danger")
            return redirect(request.url)
            
        product_info = get_product_from_openfoodfacts(barcode)
        
        if not product_info:
            flash(f"No product found for barcode {barcode}.", "warning")
            return redirect(request.url)
            
        # Prepare user profile for Gemini analysis
        user_profile = {
            "age": user.age,
            "allergies": user.allergies,
            "health_conditions": user.health_conditions
        }
        
        # Analyze with Gemini
        analysis = analyze_product_with_gemini(product_info, user_profile)
        
        # Find alternative products
        alternatives = find_alternative_products(product_info, user_profile, analysis)
        
        # Save the product to user's history
        try:
            saved_product = SavedProduct(
                barcode=barcode,
                title=product_info['title'],
                brand=product_info['brand'],
                category=product_info['category'],
                is_safe=analysis.get('is_safe', False),
                user_id=user.id
            )
            db.session.add(saved_product)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error saving product: {str(e)}")
        
        return render_template(
            'scan_result.html',
            barcode=barcode,
            barcode_type="MANUAL",
            product=product_info,
            analysis=analysis,
            alternatives=alternatives,
            user=user
        )
    
    return render_template('manual_entry.html')

# View Product History
@app.route('/history')
def product_history():
    if 'user_id' not in session:
        flash("Please log in to view your product history.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    products = SavedProduct.query.filter_by(user_id=user.id).order_by(SavedProduct.date_added.desc()).all()
    
    return render_template('history.html', products=products, user=user)

# View Product Details
@app.route('/product/<barcode>')
def product_details(barcode):
    if 'user_id' not in session:
        flash("Please log in to view product details.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    product_info = get_product_from_openfoodfacts(barcode)
    
    if not product_info:
        flash(f"No product found for barcode {barcode}.", "warning")
        return redirect(url_for('dashboard'))
        
    # Prepare user profile for Gemini analysis
    user_profile = {
        "age": user.age,
        "allergies": user.allergies,
        "health_conditions": user.health_conditions
    }
    
    # Analyze with Gemini
    analysis = analyze_product_with_gemini(product_info, user_profile)
    
    # Find alternative products
    alternatives = find_alternative_products(product_info, user_profile, analysis)
    
    return render_template(
        'product_details.html',
        product=product_info,
        analysis=analysis,
        alternatives=alternatives,
        user=user
    )

# API endpoint for product analysis
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.json
    barcode = data.get('barcode')
    user_id = data.get('user_id')
    
    if not barcode or not user_id:
        return jsonify({"error": "Barcode and user_id are required"}), 400
        
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    product_info = get_product_from_openfoodfacts(barcode)
    
    if not product_info:
        return jsonify({"error": f"No product found for barcode {barcode}"}), 404
        
    # Prepare user profile for Gemini analysis
    user_profile = {
        "age": user.age,
        "allergies": user.allergies,
        "health_conditions": user.health_conditions
    }
    
    # Analyze with Gemini
    analysis = analyze_product_with_gemini(product_info, user_profile)
    
    # Find alternative products
    alternatives = find_alternative_products(product_info, user_profile, analysis)
    
    # Save the product to user's history
    try:
        saved_product = SavedProduct(
            barcode=barcode,
            title=product_info['title'],
            brand=product_info['brand'],
            category=product_info['category'],
            is_safe=analysis.get('is_safe', False),
            user_id=user.id
        )
        db.session.add(saved_product)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving product: {str(e)}")
    
    # Prepare a more detailed response
    response_data = {
        "product": {
            "barcode": barcode,
            "title": product_info['title'],
            "brand": product_info['brand'],
            "category": product_info['category'],
            "image_url": product_info.get('image_url', ''),
            "nutrition_grade": product_info.get('nutrition_grade', ''),
            "eco_score": product_info.get('eco_score', ''),
            "allergens": product_info.get('allergens', []),
            "traces": product_info.get('traces', []),
            "ingredients_analysis": product_info.get('ingredients_analysis', [])
        },
        "analysis": analysis,
        "alternatives": alternatives
    }
    
    return jsonify(response_data)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Add a context processor to make 'now' available to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route('/api/scan_frame', methods=['POST'])
def scan_frame():
    """
    API endpoint to process a video frame for barcode/QR code detection.
    Expects a base64-encoded image in the request.
    """
    if 'user_id' not in session:
        print("API call rejected: User not authenticated")
        return jsonify({"error": "Authentication required"}), 401
    
    if not request.is_json:
        print("API call rejected: Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        print("API call rejected: No image data provided")
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Decode base64 image
        print("Received image data for processing")
        
        # Check if the image data starts with a data URL prefix
        if image_data.startswith('data:image'):
            print("Image data contains data URL prefix")
            image_data = image_data.split(',')[1]
        
        # Decode the base64 data
        try:
            image_bytes = base64.b64decode(image_data)
            print(f"Successfully decoded base64 data, size: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"Error decoding base64 data: {str(e)}")
            return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Convert to OpenCV format
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode image from base64 data")
                return jsonify({"error": "Failed to decode image"}), 400
                
            print(f"Successfully decoded image, dimensions: {frame.shape}")
        except Exception as e:
            print(f"Error converting image to OpenCV format: {str(e)}")
            return jsonify({"error": "Failed to process image"}), 400
            
        # Process the frame to detect barcodes/QR codes
        results = read_barcode(frame=frame)
        
        if not results:
            print("No barcode detected in frame")
            return jsonify({"status": "no_barcode_detected"})
        
        barcode_data, barcode_type = results[0]
        print(f"Detected barcode: {barcode_data} ({barcode_type})")
        
        if barcode_data == "MANUAL_ENTRY_REQUIRED":
            print("Barcode detected but couldn't be read clearly")
            return jsonify({
                "status": "manual_entry_required",
                "message": "Barcode detected but couldn't be read clearly. Try adjusting lighting or position."
            })
        
        # Get product information
        product_info = get_product_from_openfoodfacts(barcode_data)
        
        if not product_info:
            return jsonify({
                "status": "product_not_found",
                "barcode": barcode_data,
                "barcode_type": barcode_type,
                "message": "Barcode detected but product not found in database."
            })
            
        # Get user profile for analysis
        user = User.query.get(session['user_id'])
        user_profile = {
            "allergies": user.allergies.split(',') if user.allergies else [],
            "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
        }
        
        # Analyze product for user
        analysis = analyze_product_with_gemini(product_info, user_profile)
        
        # Return the result
        return jsonify({
            "status": "success",
            "barcode": barcode_data,
            "barcode_type": barcode_type,
            "product": {
                "title": product_info['title'],
                "brand": product_info['brand'],
                "category": product_info['category'],
                "image_url": product_info.get('image_url', '')
            },
            "analysis": analysis,
            "redirect_url": url_for('product_details', barcode=barcode_data)
        })
    except Exception as e:
        import traceback
        print(f"Error in scan_frame: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/direct_camera')
def direct_camera():
    """
    A page that uses server-side camera access with OpenCV and streams it to the browser.
    This is an alternative approach when browser-based camera access isn't working.
    """
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    return render_template('direct_camera.html')

def generate_frames():
    """
    Generator function that captures frames from the camera and yields them as JPEG images.
    """
    # Initialize the camera
    camera = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera properties for better quality
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        
        # Process the frame to detect barcodes
        results = read_barcode(frame=frame)
        
        # If a barcode is detected, draw a rectangle around it and display the data
        if results and results[0][0] != "MANUAL_ENTRY_REQUIRED":
            barcode_data, barcode_type = results[0]
            cv2.putText(frame, f"{barcode_data} ({barcode_type})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Get product information
            product_info = get_product_from_openfoodfacts(barcode_data)
            
            if product_info:
                # Display product name
                cv2.putText(frame, f"Product: {product_info['title']}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save the product to the user's history
                try:
                    user = User.query.get(session['user_id'])
                    
                    # Check if product already exists in user's history
                    existing_product = SavedProduct.query.filter_by(
                        user_id=user.id, barcode=barcode_data).first()
                    
                    if not existing_product:
                        new_product = SavedProduct(
                            barcode=barcode_data,
                            title=product_info['title'],
                            brand=product_info['brand'],
                            category=product_info['category'],
                            user_id=user.id
                        )
                        db.session.add(new_product)
                        db.session.commit()
                except Exception as e:
                    print(f"Error saving product: {str(e)}")
        
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break
        
        # Yield the frame in the response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release the camera when done
    camera.release()

@app.route('/video_feed')
def video_feed():
    """
    Route that returns the video feed from the camera.
    """
    if 'user_id' not in session:
        return "Authentication required", 401
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) 