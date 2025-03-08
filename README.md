# PlateMate

PlateMate is a dietary allergy monitoring application that helps users identify whether food products are safe for consumption based on their allergies and health conditions. The application uses Gemini 2.0 AI to analyze product ingredients and provide personalized recommendations.

## Features

- **Barcode Scanning**: Scan product barcodes to retrieve detailed information
- **AI-Powered Analysis**: Gemini 2.0 analyzes product safety based on user's health profile
- **Alternative Suggestions**: Get recommendations for safer alternatives when a product is flagged
- **User Profiles**: Store allergies and health conditions for personalized analysis
- **Product History**: Keep track of previously scanned products
- **Manual Barcode Entry**: Enter barcodes manually if scanning is not possible

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Gemini API key from Google AI Studio

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PlateMate.git
   cd PlateMate
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   SECRET_KEY=your_secret_key_for_flask
   ```

5. Create the necessary directories:
   ```
   mkdir -p uploads templates static
   ```

### Running the Application

1. Start the Flask development server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Register an Account**: Create a user profile with your allergies and health conditions
2. **Login**: Access your account using your mobile number or username
3. **Scan Products**: Upload an image of a product barcode or enter the barcode manually
4. **View Analysis**: See if the product is safe for you based on your health profile
5. **Explore Alternatives**: If a product is unsafe, check the suggested alternatives
6. **Update Profile**: Modify your health information as needed

## API Endpoints

The application provides a REST API for programmatic access:

- `POST /api/analyze`: Analyze a product for a specific user
  - Request body: `{"barcode": "1234567890", "user_id": 1}`
  - Response: Product information, safety analysis, and alternatives

## Technologies Used

- **Flask**: Web framework
- **SQLAlchemy**: ORM for database operations
- **OpenCV & pyzbar**: Barcode detection and decoding
- **OpenFoodFacts API**: Product information database
- **Google Gemini 2.0**: AI-powered product analysis
- **SQLite**: Database storage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenFoodFacts for their comprehensive food product database
- Google for the Gemini AI platform