# ACTIVI Video Thumbnail Validator

A comprehensive tool for analyzing and validating YouTube video thumbnails to ensure they meet best practices for visibility, engagement, and brand consistency.

## Features

- **Thumbnail Analysis**
  - Dimension validation (1280x720, 16:9 aspect ratio)
  - Brightness analysis with categorization
  - Color palette extraction
  - Text readability assessment
  - Visual result representation

- **Logo Detection**
  - Template matching for exact logo matches
  - Feature matching (SIFT) for logo variants
  - Overall accuracy scoring
  - Logo placement analysis
  - Actionable suggestions for improvement

## Architecture

The application consists of two main components:

### Backend (Python/FastAPI)

- Image processing with OpenCV and NumPy
- Text detection with Pytesseract
- Visualization with Matplotlib
- RESTful API with FastAPI

### Frontend (React/Next.js)

- Modern UI with Tailwind CSS
- File upload capabilities
- Real-time analysis results display
- Interactive visualization

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Tesseract OCR (for text detection)

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure Tesseract OCR is installed on your system:
   - For macOS: `brew install tesseract`
   - For Ubuntu: `sudo apt-get install tesseract-ocr`
   - For Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the required Node.js packages:
   ```
   npm install
   ```
   or
   ```
   yarn install
   ```

## Running the Application

### Start the Backend Server

1. From the project root directory:
   ```
   cd backend
   python run.py
   ```
   The backend server will start at http://localhost:8000

### Start the Frontend Development Server

1. From the project root directory:
   ```
   cd frontend
   npm run dev
   ```
   or
   ```
   yarn dev
   ```
   The frontend will be available at http://localhost:3000

## Usage

1. Open the application in your web browser at http://localhost:3000
2. Upload a thumbnail image for analysis
3. For logo detection, upload both a thumbnail and a logo image
4. View the detailed analysis results including:
   - Dimension validation
   - Brightness assessment
   - Color palette
   - Text readability
   - Logo detection scores and suggestions

## API Endpoints

- `POST /analyze-thumbnail/`: Analyze thumbnail dimensions, brightness, colors, and text
- `POST /detect-logo/`: Detect and evaluate logo presence in a thumbnail

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
