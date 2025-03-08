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
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash

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

def read_barcode(image_path):
    """
    Reads a barcode from a local image file using OpenCV's QR code detector.
    Returns a list of tuples (barcode_data, barcode_type) if found,
    or None if no barcodes are detected.
    """
    img = cv2.imread(image_path)
    
    # Create QR code detector
    qr_detector = cv2.QRCodeDetector()
    
    # Try to detect and decode QR code
    data, bbox, _ = qr_detector.detectAndDecode(img)
    
    if data:
        # Successfully detected QR code
        return [(data, "QR-Code")]
    
    # If QR detection fails, we can try some basic image processing for linear barcodes
    # This is a simplified approach and may not work for all barcodes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply image processing to enhance barcode visibility
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # For now, we'll just inform the user to enter the barcode manually
    # since reliable barcode detection without pyzbar is complex
    print("Note: Using simplified barcode detection. For better results, fix the ZBar library installation.")
    
    # Check if there are any potential barcode-like patterns
    # This is very basic and won't work for all cases
    edges = cv2.Canny(thresh, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 5:  # Arbitrary threshold for potential barcode presence
        return [("MANUAL_ENTRY_REQUIRED", "UNKNOWN")]
    
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
            
            return {
                "barcode": barcode,
                "title": title,
                "brand": brand,
                "description": description,
                "ingredients": ingredients,
                "category": category,
                "image_url": image_url,
                "nutriments": nutriments
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
    
    # First check our database for known alternatives
    if 'conflicting_ingredients' in analysis and analysis['conflicting_ingredients']:
        for ingredient in analysis['conflicting_ingredients']:
            db_alternatives = AlternativeProduct.query.filter_by(
                for_allergy=ingredient.lower()
            ).limit(3).all()
            
            for alt in db_alternatives:
                alternatives.append({
                    "barcode": alt.barcode,
                    "title": alt.title,
                    "brand": alt.brand,
                    "category": alt.category,
                    "reason": f"Safe alternative for {ingredient} allergy"
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
                    
                    for product in data['products'][:10]:  # Check first 10 products
                        ingredients_text = product.get('ingredients_text', '').lower()
                        is_safe = True
                        
                        for ingredient in conflicting_ingredients:
                            if ingredient.lower() in ingredients_text:
                                is_safe = False
                                break
                        
                        if is_safe and product.get('product_name'):
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
    if 'mobile' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

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
    
    return jsonify({
        "product": product_info,
        "analysis": analysis,
        "alternatives": alternatives
    })

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

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) 