# 1. Standard Library Imports ------------------------------------------
import os
import io
import sys
import json
import uuid
import time
import base64
import ctypes
import random
import string
import tempfile
import threading
import webbrowser
import re  # Add the missing re module for regular expressions
from datetime import datetime, timedelta
from functools import wraps

# Set the path to the zbar library before importing pyzbar
import ctypes.util
original_find_library = ctypes.util.find_library

def custom_find_library(name):
    if name == 'zbar':
        return '/opt/homebrew/lib/libzbar.dylib'
    return original_find_library(name)

ctypes.util.find_library = custom_find_library

# 2. Third-Party Library Imports ---------------------------------------
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pyzbar.pyzbar import decode

# 3. Load Environment Variables ----------------------------------------
import dotenv
dotenv.load_dotenv()

# 4. Configure Gemini API ----------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# 5. Flask + SQLAlchemy Setup ------------------------------------------
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash, Response

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///platemate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Set up the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the database
db = SQLAlchemy(app)

# 6. Upload Configuration ----------------------------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file types
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 7. Suggested Values for UI -------------------------------------------
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

# 8. Database Models --------------------------------------------------
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

# 9. Helper Functions -------------------------------------------------
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
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image from {image_path}")
                return None
        # If we're processing a video frame
        elif frame is not None:
            img = frame
            if img is None or img.size == 0:
                print("Received empty frame")
                return None
        else:
            print("No image path or frame provided")
            return None
            
        # Use pyzbar to decode barcodes
        barcodes = decode(img)
        
        if not barcodes:
            return None
            
        results = []
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            results.append((barcode_data, barcode_type))
        return results
        
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
    # Helper function to extract English fields
    def get_english_field(product, field_name, default=""):
        # First try the English-specific field
        if f"{field_name}_en" in product and product[f"{field_name}_en"]:
            return product[f"{field_name}_en"]
        # Then try the regular field
        elif field_name in product and product[field_name]:
            return product[field_name]
        # Finally return the default value
        return default
    
    # First try direct barcode lookup with language and country parameters
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json?lc=en&cc=us"
    
    try:
        response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if product data exists
            if "product" in data and data["product"]:
                product = data["product"]
                
                # Extract basic product information with English priority
                title = get_english_field(product, "product_name", "No product title found")
                brand = get_english_field(product, "brands", "Unknown brand")
                
                # Improve description handling
                description = get_english_field(product, "generic_name", "")
                
                # If description is empty or in a non-English language, try alternative fields
                if not description or any(c.isalpha() and ord(c) > 127 for c in description):
                    # Try product_name_en if generic_name didn't work
                    if "product_name_en" in product and product["product_name_en"]:
                        description = product["product_name_en"]
                    # Try the first English category as a fallback
                    elif "categories_tags" in product:
                        for tag in product["categories_tags"]:
                            if tag.startswith('en:'):
                                description = tag[3:].replace('-', ' ').capitalize()
                                break
                
                # If still empty, use a default
                if not description:
                    description = "No description available"
                
                ingredients = get_english_field(product, "ingredients_text", "No ingredients information available")
                
                # Handle category with the same improved logic
                category = get_english_field(product, "categories", "Unknown category")
                if category == "Unknown category" or not category.strip() or "," in category:
                    # Try to extract English categories from categories_tags
                    if "categories_tags" in product:
                        categories_tags = product.get("categories_tags", [])
                        if categories_tags:
                            # Extract only English category names from tags
                            categories = []
                            for tag in categories_tags:
                                if tag.startswith('en:'):
                                    # Convert tag to readable format (replace hyphens with spaces, capitalize words)
                                    clean_category = tag[3:].replace('-', ' ')
                                    # Capitalize each word
                                    clean_category = ' '.join(word.capitalize() for word in clean_category.split())
                                    categories.append(clean_category)
                            
                            if categories:
                                category = ", ".join(categories)
                    
                    # If we still don't have a good category, try to clean up the existing one
                    if category == "Unknown category" or not category.strip():
                        if "categories" in product:
                            raw_categories = product.get("categories", "")
                            # Split by commas and clean up each category
                            if raw_categories and "," in raw_categories:
                                cat_parts = [part.strip() for part in raw_categories.split(",")]
                                # Keep only English-looking parts (no special characters)
                                english_parts = []
                                for part in cat_parts:
                                    # Simple heuristic: if it has mostly ASCII characters, it's likely English
                                    if sum(c.isalpha() and ord(c) < 128 for c in part) > len(part) * 0.7:
                                        english_parts.append(part.capitalize())
                                
                                if english_parts:
                                    category = ", ".join(english_parts)
                
                # Get image URL - prioritize English images if available
                image_url = ""
                if "selected_images" in product and "front" in product["selected_images"]:
                    if "display" in product["selected_images"]["front"]:
                        if "en" in product["selected_images"]["front"]["display"]:
                            image_url = product["selected_images"]["front"]["display"]["en"]
                        else:
                            # Fallback to any available image
                            for lang in product["selected_images"]["front"]["display"]:
                                image_url = product["selected_images"]["front"]["display"][lang]
                                break
                
                # If still no image, try the regular image fields
                if not image_url:
                    image_url = product.get("image_url", "")
                if not image_url and "image_front_url" in product:
                    image_url = product.get("image_front_url", "")
                
                # Extract nutriments
                nutriments = product.get("nutriments", {})
                
                # Extract allergen information - clean up the tags to remove 'en:' prefix
                allergens = []
                for allergen in product.get("allergens_tags", []):
                    if allergen.startswith('en:'):
                        allergens.append(allergen[3:].replace('-', ' '))
                    else:
                        allergens.append(allergen.replace('-', ' '))
                
                allergens_from_ingredients = product.get("allergens_from_ingredients", "")
                
                # Extract ingredient analysis information - clean up the tags
                ingredients_analysis = []
                for tag in product.get("ingredients_analysis_tags", []):
                    if tag.startswith('en:'):
                        ingredients_analysis.append(tag[3:].replace('-', ' '))
                    else:
                        ingredients_analysis.append(tag.replace('-', ' '))
                
                # Extract detailed ingredients list
                ingredients_list = []
                if "ingredients" in product and isinstance(product["ingredients"], list):
                    for ingredient in product["ingredients"]:
                        # Try to get English text if available
                        ingredient_text = ingredient.get("text", "")
                        if "id" in ingredient and ingredient["id"].startswith("en:"):
                            ingredient_text = ingredient["id"][3:].replace('-', ' ')
                        
                        ingredients_list.append({
                            "id": ingredient.get("id", ""),
                            "text": ingredient_text,
                            "percent": ingredient.get("percent_estimate", 0),
                            "vegan": ingredient.get("vegan", "unknown"),
                            "vegetarian": ingredient.get("vegetarian", "unknown"),
                            "from_palm_oil": ingredient.get("from_palm_oil", "unknown")
                        })
                
                # Extract nutrition grade and eco-score
                nutrition_grade = product.get("nutriscore_grade", "")
                eco_score = product.get("ecoscore_grade", "")
                
                # Extract traces information (may contain traces of)
                traces = []
                for trace in product.get("traces_tags", []):
                    if trace.startswith('en:'):
                        traces.append(trace[3:].replace('-', ' '))
                    else:
                        traces.append(trace.replace('-', ' '))
                
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
            else:
                # If direct lookup failed, try searching by barcode
                print(f"Direct lookup failed for barcode {barcode}, trying search...")
                search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={barcode}&search_simple=1&action=process&json=1&lc=en&cc=us"
                
                try:
                    search_response = requests.get(search_url, timeout=10)
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        if 'products' in search_data and search_data['products']:
                            # Use the first product from search results
                            product = search_data['products'][0]
                            
                            # Extract the same information as above using the helper function
                            title = get_english_field(product, "product_name", "No product title found")
                            brand = get_english_field(product, "brands", "Unknown brand")
                            
                            # Improve description handling
                            description = get_english_field(product, "generic_name", "")
                            
                            # If description is empty or in a non-English language, try alternative fields
                            if not description or any(c.isalpha() and ord(c) > 127 for c in description):
                                # Try product_name_en if generic_name didn't work
                                if "product_name_en" in product and product["product_name_en"]:
                                    description = product["product_name_en"]
                                # Try the first English category as a fallback
                                elif "categories_tags" in product:
                                    for tag in product["categories_tags"]:
                                        if tag.startswith('en:'):
                                            description = tag[3:].replace('-', ' ').capitalize()
                                            break
                            
                            # If still empty, use a default
                            if not description:
                                description = "No description available"
                            
                            ingredients = get_english_field(product, "ingredients_text", "No ingredients information available")
                            
                            # Handle category with the same improved logic
                            category = get_english_field(product, "categories", "Unknown category")
                            if category == "Unknown category" or not category.strip() or "," in category:
                                # Try to extract English categories from categories_tags
                                if "categories_tags" in product:
                                    categories_tags = product.get("categories_tags", [])
                                    if categories_tags:
                                        # Extract only English category names from tags
                                        categories = []
                                        for tag in categories_tags:
                                            if tag.startswith('en:'):
                                                # Convert tag to readable format (replace hyphens with spaces, capitalize words)
                                                clean_category = tag[3:].replace('-', ' ')
                                                # Capitalize each word
                                                clean_category = ' '.join(word.capitalize() for word in clean_category.split())
                                                categories.append(clean_category)
                                        
                                        if categories:
                                            category = ", ".join(categories)
                                
                                # If we still don't have a good category, try to clean up the existing one
                                if category == "Unknown category" or not category.strip():
                                    if "categories" in product:
                                        raw_categories = product.get("categories", "")
                                        # Split by commas and clean up each category
                                        if raw_categories and "," in raw_categories:
                                            cat_parts = [part.strip() for part in raw_categories.split(",")]
                                            # Keep only English-looking parts (no special characters)
                                            english_parts = []
                                            for part in cat_parts:
                                                # Simple heuristic: if it has mostly ASCII characters, it's likely English
                                                if sum(c.isalpha() and ord(c) < 128 for c in part) > len(part) * 0.7:
                                                    english_parts.append(part.capitalize())
                                                
                                            if english_parts:
                                                category = ", ".join(english_parts)
                            
                            # Get image URL
                            image_url = ""
                            if "selected_images" in product and "front" in product["selected_images"]:
                                if "display" in product["selected_images"]["front"]:
                                    if "en" in product["selected_images"]["front"]["display"]:
                                        image_url = product["selected_images"]["front"]["display"]["en"]
                                    else:
                                        # Fallback to any available image
                                        for lang in product["selected_images"]["front"]["display"]:
                                            image_url = product["selected_images"]["front"]["display"][lang]
                                            break
                            
                            if not image_url:
                                image_url = product.get("image_url", "")
                            
                            nutriments = product.get("nutriments", {})
                            
                            print(f"Found product via search: {title} by {brand}")
                            
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
                        else:
                            print(f"No products found in search for barcode {barcode}")
                            return None
                    else:
                        print(f"Search request failed with status code {search_response.status_code}")
                        return None
                except Exception as e:
                    print(f"Error during search request: {str(e)}")
                    return None
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during request: {str(e)}")
        return None

def analyze_product_with_gemini(product_info, user_profile):
    """
    Uses Gemini 2.0 to analyze product safety based on user's allergies and health conditions.
    Returns a detailed analysis with safety assessment and recommendations.
    """
    if not product_info or not user_profile:
        return {"status": "error", "message": "Invalid product or user information."}
    
    # Ensure user_profile has all required fields
    user_age = user_profile.get('age', 'Not specified')
    user_allergies = user_profile.get('allergies', [])
    user_health_conditions = user_profile.get('health_conditions', [])
    
    # Format allergies and health conditions as strings if they're lists
    if isinstance(user_allergies, list):
        user_allergies = ', '.join(user_allergies) if user_allergies else 'None'
    if isinstance(user_health_conditions, list):
        user_health_conditions = ', '.join(user_health_conditions) if user_health_conditions else 'None'
    
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
    - Age: {user_age}
    - Allergies: {user_allergies}
    - Health Conditions: {user_health_conditions}
    
    IMPORTANT: Ensure your analysis is based ONLY on the product information provided above. DO NOT invent or assume any information not explicitly stated. The product name is "{product_info['title']}" from brand "{product_info['brand']}" - do not change or modify this information in your analysis.
    
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
        
        # Ensure the analysis doesn't contain incorrect product information
        if "explanation" in analysis:
            # Replace any incorrect product references with the actual product name
            incorrect_patterns = [
                (r'blueberry jam', product_info['title']),
                (r'jam', product_info['title']),
                (r'jelly', product_info['title'])
            ]
            
            for pattern, replacement in incorrect_patterns:
                analysis["explanation"] = re.sub(pattern, replacement, analysis["explanation"], flags=re.IGNORECASE)
        
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
        
        # Create a search query for OpenFoodFacts with language and country parameters
        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={category}&search_simple=1&action=process&json=1&lc=en&cc=us"
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'products' in data and data['products']:
                    # Filter products that don't contain the conflicting ingredients
                    conflicting_ingredients = analysis.get('conflicting_ingredients', [])
                    user_allergens = [allergen.lower() for allergen in user_allergies]
                    
                    for product in data['products'][:15]:  # Check more products
                        # Skip products without English names or descriptions
                        if not product.get('product_name') and not product.get('product_name_en'):
                            continue
                            
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
                        
                        if is_safe:
                            # Get product name, preferring English version
                            product_name = product.get('product_name_en', '') or product.get('product_name', '')
                            if not product_name:
                                continue
                                
                            product_key = f"{product_name.lower()}|{product.get('brands', '').lower()}"
                            if product_key not in seen_products:
                                seen_products.add(product_key)
                                alternatives.append({
                                    "barcode": product.get('code', ''),
                                    "title": product_name,
                                    "brand": product.get('brands', ''),
                                    "category": product.get('categories', ''),
                                    "reason": "Safe alternative from the same category"
                                })
                            
                            if len(alternatives) >= 5:
                                break
                                
            # If we still don't have enough alternatives, try US-specific database
            if len(alternatives) < 3:
                us_search_url = f"https://us.openfoodfacts.org/cgi/search.pl?search_terms={category}&search_simple=1&action=process&json=1"
                
                try:
                    us_response = requests.get(us_search_url, timeout=10)
                    if us_response.status_code == 200:
                        us_data = us_response.json()
                        if 'products' in us_data and us_data['products']:
                            for product in us_data['products'][:10]:
                                # Skip products without English names
                                product_name = product.get('product_name_en', '') or product.get('product_name', '')
                                if not product_name:
                                    continue
                                    
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
                                
                                if is_safe:
                                    product_key = f"{product_name.lower()}|{product.get('brands', '').lower()}"
                                    if product_key not in seen_products:
                                        seen_products.add(product_key)
                                        alternatives.append({
                                            "barcode": product.get('code', ''),
                                            "title": product_name,
                                            "brand": product.get('brands', ''),
                                            "category": product.get('categories', ''),
                                            "reason": "Safe US alternative from the same category"
                                        })
                                    
                                    if len(alternatives) >= 5:
                                        break
                except Exception as e:
                    print(f"Error searching for US alternatives: {str(e)}")
        except Exception as e:
            print(f"Error searching for alternatives: {str(e)}")
    
    return alternatives[:5]  # Return at most 5 alternatives

# 10. Routes -----------------------------------------------------------

# Home route
@app.route('/')
def home():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            flash("User not found. Please log in again.", "danger")
            return redirect(url_for('login'))
            
        # Get user's recently scanned products
        recent_products = SavedProduct.query.filter_by(user_id=user.id).order_by(SavedProduct.date_added.desc()).limit(5).all()
        
        return render_template('dashboard.html', user=user, recent_products=recent_products)
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
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        mobile = request.form.get('mobile')
        username = request.form.get('username')
        
        if not mobile and not username:
            flash("Please enter your mobile number or username.", "danger")
            return redirect(request.url)
            
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
            return redirect(url_for('home'))
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
    """
    Handle barcode scanning from uploaded images.
    """
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
            # Generate a unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"upload_{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Make sure the uploads directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            file.save(filepath)
            
            try:
                # Process the barcode
                results = read_barcode(filepath)
                
                if not results:
                    flash("No barcode detected in the uploaded image.", "danger")
                    return redirect(request.url)
                    
                barcode_data, barcode_type = results[0]
                
                # Check if this product has already been scanned by the user
                existing_product = SavedProduct.query.filter_by(
                    user_id=user.id, 
                    barcode=barcode_data
                ).first()
                
                if existing_product:
                    flash(f"You've already scanned this product ({existing_product.title}). Showing existing details.", "info")
                    return redirect(url_for('product_details', barcode=barcode_data))
                
                product_info = get_product_from_openfoodfacts(barcode_data)
                
                if not product_info:
                    flash(f"No product found for barcode {barcode_data}.", "warning")
                    return redirect(request.url)
                
                # Analyze the product with Gemini
                analysis_result = analyze_product_with_gemini(product_info, {
                    "name": user.name,
                    "age": user.age,
                    "allergies": user.allergies.split(',') if user.allergies else [],
                    "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
                })
                
                # Find alternative products
                alternatives = find_alternative_products(product_info, {
                    "name": user.name,
                    "age": user.age,
                    "allergies": user.allergies.split(',') if user.allergies else [],
                    "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
                }, analysis_result)
                
                # Save the product to the user's history
                new_product = SavedProduct(
                    barcode=barcode_data,
                    title=product_info.get('product_name', 'Unknown Product'),
                    brand=product_info.get('brands', ''),
                    category=product_info.get('categories', ''),
                    is_safe=analysis_result.get('is_safe', False),
                    user_id=user.id
                )
                
                db.session.add(new_product)
                db.session.commit()
                
                # Store analysis in session for the product details page
                session['product_info'] = product_info
                session['analysis_result'] = analysis_result
                session['alternatives'] = alternatives
                
                return redirect(url_for('product_details', barcode=barcode_data))
                
            except Exception as e:
                flash(f"Error processing image: {str(e)}", "danger")
                return redirect(request.url)
            finally:
                # Always clean up the file, but use try/except to avoid errors if file doesn't exist
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    print(f"Error removing file {filepath}: {str(e)}")
    
    return render_template('scan.html')

# Manual Entry
@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    if 'user_id' not in session:
        flash("Please log in to manually enter products.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        barcode = request.form.get('barcode')
        
        if not barcode:
            flash("Barcode is required.", "danger")
            return redirect(request.url)
        
        # Check if this product has already been scanned by the user
        existing_product = SavedProduct.query.filter_by(
            user_id=user.id, 
            barcode=barcode
        ).first()
        
        if existing_product:
            flash(f"You've already scanned this product ({existing_product.title}). Showing existing details.", "info")
            return redirect(url_for('product_details', barcode=barcode))
            
        # Get product information
        product_info = get_product_from_openfoodfacts(barcode)
        
        if not product_info:
            flash(f"No product found for barcode {barcode}.", "warning")
            return redirect(request.url)
        
        # Log the product information for debugging
        print(f"Manual entry - Product info for barcode {barcode}: {product_info['title']} by {product_info['brand']}")
            
        # Prepare user profile for Gemini analysis
        user_profile = {
            "name": user.name,
            "age": user.age,
            "allergies": user.allergies.split(',') if user.allergies else [],
            "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
        }
        
        # Analyze with Gemini
        analysis = analyze_product_with_gemini(product_info, user_profile)
        
        # Ensure analysis doesn't override product information
        if "product_name" in analysis:
            del analysis["product_name"]
        
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
        
        return redirect(url_for('product_details', barcode=barcode))
    
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
    
    # Find the saved product in the database
    saved_product = SavedProduct.query.filter_by(user_id=user.id, barcode=barcode).first()
    
    # Get product information from OpenFoodFacts
    product_info = get_product_from_openfoodfacts(barcode)
    
    if not product_info:
        flash(f"No product found for barcode {barcode}.", "warning")
        return redirect(url_for('home'))
    
    # Log the product information for debugging
    print(f"Product info for barcode {barcode}: {product_info['title']} by {product_info['brand']}")
        
    # Prepare user profile for Gemini analysis
    user_profile = {
        "name": user.name,
        "age": user.age,
        "allergies": user.allergies.split(',') if user.allergies else [],
        "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
    }
    
    # Analyze with Gemini
    analysis = analyze_product_with_gemini(product_info, user_profile)
    
    # Ensure analysis doesn't override product information
    if "product_name" in analysis:
        del analysis["product_name"]
    
    # Find alternative products
    alternatives = find_alternative_products(product_info, user_profile, analysis)
    
    return render_template(
        'product_details.html',
        product=product_info,
        saved_product=saved_product,
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
        "name": user.name,
        "age": user.age,
        "allergies": user.allergies.split(',') if user.allergies else [],
        "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
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
        
        # Check if this product has already been scanned by the user
        user = User.query.get(session['user_id'])
        existing_product = SavedProduct.query.filter_by(
            user_id=user.id, 
            barcode=barcode_data
        ).first()
        
        if existing_product:
            print(f"Product already scanned: {existing_product.title}")
            return jsonify({
                "status": "already_scanned",
                "barcode": barcode_data,
                "barcode_type": barcode_type,
                "product": {
                    "title": existing_product.title,
                    "brand": existing_product.brand,
                    "category": existing_product.category
                },
                "message": f"You've already scanned this product ({existing_product.title}).",
                "redirect_url": url_for('product_details', barcode=barcode_data)
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
        user_profile = {
            "name": user.name,
            "age": user.age,
            "allergies": user.allergies.split(',') if user.allergies else [],
            "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
        }
        
        # Analyze product for user
        analysis = analyze_product_with_gemini(product_info, user_profile)
        
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
            print(f"Saved product to history: {product_info['title']}")
        except Exception as e:
            db.session.rollback()
            print(f"Error saving product: {str(e)}")
        
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
    Render the direct camera feed page.
    """
    if 'user_id' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    
    # Check if we need to redirect to a product page
    if 'redirect_to_product' in session:
        barcode = session.pop('redirect_to_product')
        flash("Product already scanned. Showing existing details.", "info")
        return redirect(url_for('product_details', barcode=barcode))
    
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
    
    # Keep track of the last detected barcode to avoid repeated processing
    last_barcode = None
    last_detection_time = 0
    cooldown_period = 3  # seconds
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        
        # Get current time
        current_time = time.time()
        
        # Process the frame to detect barcodes
        results = read_barcode(frame=frame)
        
        # If a barcode is detected
        if results and results[0][0] != "MANUAL_ENTRY_REQUIRED":
            barcode_data, barcode_type = results[0]
            
            # Only process if it's a new barcode or cooldown period has passed
            if barcode_data != last_barcode or (current_time - last_detection_time) > cooldown_period:
                last_barcode = barcode_data
                last_detection_time = current_time
                
                cv2.putText(frame, f"{barcode_data} ({barcode_type})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Check if this product has already been scanned by the user
                try:
                    user = User.query.get(session['user_id'])
                    existing_product = SavedProduct.query.filter_by(
                        user_id=user.id, 
                        barcode=barcode_data
                    ).first()
                    
                    if existing_product:
                        # Display message that product was already scanned
                        cv2.putText(frame, f"Already scanned: {existing_product.title}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "Redirecting to product page...", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Yield a few frames with the message before redirecting
                        for _ in range(10):  # Show message for about 10 frames
                            ret, buffer = cv2.imencode('.jpg', frame)
                            if not ret:
                                break
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            time.sleep(0.1)  # Short delay
                        
                        # Redirect to product details page
                        # Note: We can't directly redirect from here, so we'll set a session flag
                        session['redirect_to_product'] = barcode_data
                        continue
                    
                    # Get product information
                    product_info = get_product_from_openfoodfacts(barcode_data)
                    
                    if product_info:
                        # Display product name
                        cv2.putText(frame, f"Product: {product_info['title']}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Save the product to the user's history
                        try:
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
                except Exception as e:
                    print(f"Error processing barcode: {str(e)}")
        
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

# Admin route to reset the database (for development only)
@app.route('/reset_database', methods=['GET', 'POST'])
def reset_database():
    """
    Reset the database by dropping and recreating all tables.
    This is for development purposes only and should be disabled in production.
    """
    if request.method == 'POST':
        try:
            # Drop all tables
            db.drop_all()
            print("All tables dropped")
            
            # Recreate all tables
            db.create_all()
            print("All tables recreated")
            
            # Clear session
            session.clear()
            
            flash("Database has been reset successfully. All data has been cleared.", "success")
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Error resetting database: {str(e)}", "danger")
    
    return render_template('reset_database.html')

# Delete a saved product
@app.route('/product/delete/<int:product_id>', methods=['POST'])
def delete_product(product_id):
    """
    Delete a product from the user's history.
    """
    if 'user_id' not in session:
        flash("Please log in to manage your products.", "warning")
        return redirect(url_for('login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash("User not found. Please log in again.", "danger")
        return redirect(url_for('login'))
    
    # Find the product
    product = SavedProduct.query.filter_by(id=product_id, user_id=user.id).first()
    
    if not product:
        flash("Product not found or you don't have permission to delete it.", "danger")
        return redirect(url_for('product_history'))
    
    try:
        # Store product info for the success message
        product_title = product.title
        
        # Delete the product
        db.session.delete(product)
        db.session.commit()
        
        flash(f"Product '{product_title}' has been deleted from your history.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting product: {str(e)}", "danger")
    
    # Redirect back to the referring page or history page
    referrer = request.referrer
    if referrer and (url_for('product_history') in referrer or url_for('dashboard') in referrer):
        return redirect(referrer)
    else:
        return redirect(url_for('product_history'))

# Function to open Oculus casting page on startup
def open_oculus_casting():
    # Wait a few seconds to ensure the Flask app has started
    time.sleep(3)
    print("Opening Oculus casting page...")
    webbrowser.open("https://www.oculus.com/casting", new=2)

# Endpoint to manually trigger opening the Oculus casting page
@app.route('/open_oculus', methods=['GET'])
def trigger_oculus_casting():
    """Endpoint to manually trigger opening the Oculus casting page"""
    threading.Thread(target=open_oculus_casting).start()
    return jsonify({"status": "success", "message": "Opening Oculus casting page"}), 200

# API endpoint for Quest 3 integration
@app.route('/status', methods=['GET'])
def status():
    """Simple endpoint to check if the server is running"""
    return jsonify({"status": "online", "message": "PlateMate server is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_quest_image():
    """
    Endpoint to analyze images from Quest 3 headset
    Takes a screenshot of the Chrome browser window showing Oculus casting
    Returns: Analysis result formatted for voice output based on the prompt
    """
    try:
        # Get data from request
        data = request.get_json()
        print("=== QUEST DEBUG: Received analyze request ===")
        
        if not data or 'prompt' not in data:
            print("QUEST DEBUG: Missing prompt in request")
            return jsonify({
                "success": False,
                "error": "Missing required field: prompt"
            }), 400
            
        # Extract the prompt (used for formatting the response)
        format_prompt = data['prompt']
        print(f"QUEST DEBUG: Prompt received: {format_prompt[:50]}...")
        
        # Create a unique filename for this request
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"oculus_screenshot_{timestamp}.jpg")
        
        # Also create a permanent debug image path that won't be deleted
        debug_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "last_quest_screenshot.jpg")
        
        # Make sure the uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        try:
            # Just find and capture the existing Chrome window with Oculus casting
            import pyautogui
            import time
            
            print("QUEST DEBUG: Attempting to find and capture existing Chrome window with Oculus casting...")
            
            # Try to find and focus on the Chrome window with Oculus casting
            if os.name == 'posix':  # macOS
                try:
                    # Use AppleScript to activate Chrome and focus on the Oculus casting tab
                    applescript = '''
                    tell application "Google Chrome"
                        activate
                        set found to false
                        set windowList to every window
                        repeat with aWindow in windowList
                            set tabList to every tab of aWindow
                            repeat with atab in tabList
                                if URL of atab contains "oculus.com/casting" then
                                    set active tab of aWindow to atab
                                    set index of aWindow to 1
                                    set found to true
                                    exit repeat
                                end if
                            end repeat
                            if found then exit repeat
                        end repeat
                    end tell
                    '''
                    
                    # Run the AppleScript to focus on the Oculus casting tab
                    import subprocess
                    print("QUEST DEBUG: Running AppleScript to focus on existing Oculus casting tab")
                    subprocess.run(['osascript', '-e', applescript], check=False)
                    
                    # Give Chrome time to come to the foreground
                    time.sleep(1)
                    
                    # Take a screenshot
                    print("QUEST DEBUG: Taking screenshot of Chrome with Oculus casting")
                    screenshot = pyautogui.screenshot()
                    
                except Exception as e:
                    print(f"QUEST DEBUG: Error with AppleScript: {str(e)}")
                    # Fallback to just taking a screenshot
                    screenshot = pyautogui.screenshot()
            else:  # Windows
                # On Windows, try to find Chrome window with Oculus in the title
                import win32gui
                import win32con
                
                def window_enum_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        window_title = win32gui.GetWindowText(hwnd)
                        if "Chrome" in window_title and ("Oculus" in window_title or "Meta Quest" in window_title or "casting" in window_title):
                            results.append(hwnd)
                
                chrome_windows = []
                win32gui.EnumWindows(window_enum_callback, chrome_windows)
                
                if chrome_windows:
                    # Activate the Chrome window
                    hwnd = chrome_windows[0]
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore if minimized
                    win32gui.SetForegroundWindow(hwnd)  # Bring to front
                    print(f"QUEST DEBUG: Found and activated Chrome window with handle {hwnd}")
                    
                    # Give it time to come to the foreground
                    time.sleep(1)
                    
                    # Get the window dimensions
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    width = right - left
                    height = bottom - top
                    
                    # Take screenshot of the Chrome window
                    screenshot = pyautogui.screenshot(region=(left, top, width, height))
                    print(f"QUEST DEBUG: Captured Chrome window with dimensions: {width}x{height}")
                else:
                    print("QUEST DEBUG: No Chrome window with Oculus found, taking full screenshot")
                    screenshot = pyautogui.screenshot()
            
            print(f"QUEST DEBUG: Screenshot taken, size: {screenshot.width}x{screenshot.height}, mode: {screenshot.mode}")
            
            # Convert to RGB mode if needed (JPEG doesn't support RGBA)
            if screenshot.mode == 'RGBA':
                screenshot = screenshot.convert('RGB')
                print("QUEST DEBUG: Converted from RGBA to RGB for JPEG compatibility")
                
            # Save the screenshot with high quality
            screenshot.save(temp_image_path, quality=100)
            
            # Also save a copy for debugging that won't be deleted
            screenshot.save(debug_image_path, quality=100)
            print(f"QUEST DEBUG: Screenshot saved to {temp_image_path} and {debug_image_path}")
            
            # Try to read barcode directly from the screenshot without preprocessing
            print("QUEST DEBUG: Attempting barcode detection on original image...")
            barcode_results = read_barcode(temp_image_path)
            
            # Get current user profile
            user_profile = None
            if 'user_id' in session:
                user = User.query.get(session['user_id'])
                if user:
                    user_profile = {
                        "name": user.name,
                        "age": user.age,
                        "allergies": user.allergies.split(',') if user.allergies else [],
                        "health_conditions": user.health_conditions.split(',') if user.health_conditions else []
                    }
                    print(f"QUEST DEBUG: Using logged-in user profile: {user.name}")
            else:
                # Create a default user profile if not logged in
                user_profile = {
                    "name": "Guest",
                    "age": "Not specified",
                    "allergies": [],
                    "health_conditions": []
                }
                print("QUEST DEBUG: Using default guest profile")
            
            # If barcode detected, get product info
            if barcode_results and barcode_results[0][0] != "MANUAL_ENTRY_REQUIRED":
                barcode = barcode_results[0][0]
                barcode_type = barcode_results[0][1]
                print(f"QUEST DEBUG: Barcode detected: {barcode}, type: {barcode_type}")
                product_info = get_product_from_openfoodfacts(barcode)
                
                if not product_info:
                    print(f"QUEST DEBUG: No product info found for barcode {barcode}")
                    return jsonify({
                        "success": True,
                        "response": "I couldn't find information about this product. The barcode was detected but no product data was found."
                    }), 200
                
                print(f"QUEST DEBUG: Product info found: {product_info.get('title', 'Unknown')} by {product_info.get('brand', 'Unknown')}")
                
                # Analyze the product with Gemini
                analysis_result = analyze_product_with_gemini(product_info, user_profile)
                print("QUEST DEBUG: Product analyzed with Gemini")
                print(f"QUEST DEBUG: Full analysis result: {json.dumps(analysis_result, indent=2)}")
                
                # Special case handling for common products
                product_name = product_info.get('product_name', '').lower()
                product_category = product_info.get('category', '').lower()
                product_title = product_info.get('title', '').lower()
                product_brand = product_info.get('brand', '').lower()
                
                # Check if this is water
                if ("water" in product_name or "water" in product_category or 
                    "water" in product_title):
                    print("QUEST DEBUG: Product detected as water, marking as safe")
                    analysis_result['is_safe'] = True
                    analysis_result['safety_level'] = "Safe"
                    analysis_result['explanation'] = "This is water, which is generally safe for consumption."
                
                # Check if this is a soft drink
                elif ("soda" in product_name or "soft drink" in product_category or 
                      "limca" in product_name or "limca" in product_title or
                      "pepsi" in product_brand or "coca-cola" in product_brand or
                      "sprite" in product_name or "fanta" in product_name):
                    print("QUEST DEBUG: Product detected as a soft drink")
                    analysis_result['is_safe'] = True
                    analysis_result['safety_level'] = "Caution"
                    analysis_result['explanation'] = "This is a soft drink. While generally safe for consumption, it contains sugar and should be consumed in moderation."
                
                # Check if there was an error in analysis
                elif "Error" in analysis_result.get('safety_level', '') or "error" in analysis_result.get('status', ''):
                    print("QUEST DEBUG: Error in analysis, providing generic response")
                    analysis_result['is_safe'] = True
                    analysis_result['safety_level'] = "Caution"
                    analysis_result['explanation'] = f"This product ({product_info.get('product_name', 'Unknown')}) appears to be a food item. Without complete information, we recommend checking the ingredients list for any allergens or concerns."
                
                # Find alternative products
                try:
                    alternatives = find_alternative_products(product_info, user_profile, analysis_result)
                    print(f"QUEST DEBUG: Found {len(alternatives)} alternative products")
                except Exception as e:
                    print(f"Error searching for alternatives: {str(e)}")
                    alternatives = []
                
                # Format the response for voice output based on the prompt
                is_safe = analysis_result.get('is_safe', False)
                safety_level = analysis_result.get('safety_level', "Unknown")
                
                # Determine safety status text based on safety_level
                if safety_level == "Safe":
                    safety_status = "Safe"
                elif safety_level == "Caution":
                    safety_status = "Use with caution"
                else:
                    safety_status = "Unsafe"
                
                # Create a concise response suitable for voice output
                response_text = f"{safety_status}. {analysis_result.get('summary', analysis_result.get('explanation', ''))}"
                
                # Add alternatives if available (limit to 3) and not completely safe
                if alternatives and len(alternatives) > 0 and safety_level != "Safe":
                    alt_count = min(3, len(alternatives))
                    response_text += f" Here are {alt_count} healthier alternatives: "
                    
                    for i in range(alt_count):
                        alt = alternatives[i]
                        response_text += f"{i+1}. {alt.get('title', 'Unknown product')}. "
                
                print(f"QUEST DEBUG: Response prepared: {response_text}")
                return jsonify({
                    "success": True,
                    "response": response_text,
                    "product_info": {
                        "title": product_info.get('product_name', 'Unknown'),
                        "brand": product_info.get('brands', 'Unknown'),
                        "barcode": barcode,
                        "is_safe": is_safe,
                        "safety_level": safety_level
                    }
                }), 200
            else:
                # No barcode detected
                print("QUEST DEBUG: No barcode detected in the screenshot")
                return jsonify({
                    "success": True,
                    "response": "I couldn't detect a barcode in this image. Please try to get a clearer view of the product barcode."
                }), 200
            
        except Exception as e:
            import traceback
            print(f"QUEST DEBUG: Screenshot processing error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({
                "success": False,
                "error": f"Failed to process screenshot: {str(e)}"
            }), 400
        
    except Exception as e:
        import traceback
        print(f"QUEST DEBUG: Error in analyze_quest_image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500
    finally:
        # Clean up the temporary file, but keep the debug files
        try:
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"QUEST DEBUG: Removed temporary file {temp_image_path}")
        except Exception as e:
            print(f"QUEST DEBUG: Error removing file {temp_image_path}: {str(e)}")

# Run the application
if __name__ == "__main__":
    # Open the Oculus casting page in a new window when the server starts
    def open_oculus_window_on_startup():
        print("Opening Oculus casting page in a new browser window...")
        time.sleep(3)  # Wait a bit for the server to fully start
        try:
            if os.name == 'posix':  # macOS
                # Use AppleScript to open in a new Chrome window
                applescript = '''
                tell application "Google Chrome"
                    make new window
                    set URL of active tab of front window to "https://www.oculus.com/casting"
                    activate
                end tell
                '''
                import subprocess
                subprocess.run(['osascript', '-e', applescript], check=False)
                print("Opened Oculus casting page in Chrome using AppleScript")
            else:  # Windows
                webbrowser.get('chrome').open_new("https://www.oculus.com/casting")
                print("Opened Oculus casting page in Chrome using webbrowser")
        except Exception as e:
            print(f"Error opening Oculus casting page: {str(e)}")
            try:
                # Fallback to default browser
                webbrowser.open_new("https://www.oculus.com/casting")
                print("Opened Oculus casting page in default browser")
            except Exception as e2:
                print(f"Error opening browser: {str(e2)}")

    # Start the browser window in a separate thread to avoid blocking
    threading.Thread(target=open_oculus_window_on_startup, daemon=True).start()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 