from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import sys
import tempfile
from pathlib import Path
import json
import base64
import cv2
import numpy as np

# Configure matplotlib to use Agg backend for server environments
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import io

# Add parent directory to path to import the modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the Python modules
import image_analyzer
import logo_detector

app = FastAPI(title="ACTIVI Video Thumbnail Validator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the ACTIVI Video Thumbnail Validator API"}

@app.post("/analyze-thumbnail/")
async def analyze_thumbnail(thumbnail: UploadFile = File(...)):
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / thumbnail.filename
        
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(thumbnail.file, buffer)
        
        try:
            # Load and analyze the image
            cv_img, pil_img = image_analyzer.load_image(str(temp_path))
            
            # Analyze the image
            dimension_results = image_analyzer.check_dimensions(cv_img)
            color_palette = image_analyzer.extract_color_palette(cv_img, n_colors=5)
            brightness_results = image_analyzer.analyze_brightness(cv_img)
            text_results = image_analyzer.analyze_text_readability(pil_img)
            
            # Create results dictionary
            results = {
                'dimensions': dimension_results,
                'brightness': brightness_results,
                'text': text_results
            }
            
            # Create visualization - use a single figure
            plt.close('all')  # Close any existing figures
            fig = plt.figure(figsize=(12, 8))
            
            # Use a modified version of visualize_results that doesn't call plt.show()
            # This prevents issues with the web server context
            image_analyzer.visualize_results(cv_img, color_palette, results, output_path=None)
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')  # Make sure to close all figures
            
            # Convert color palette to serializable format
            serializable_palette = []
            for color, percentage in color_palette:
                serializable_palette.append({
                    "color": color.tolist(),
                    "percentage": float(percentage)
                })
            
            # Prepare response data
            response_data = {
                "dimensions": {
                    "width": dimension_results["width"],
                    "height": dimension_results["height"],
                    "aspect_ratio": float(dimension_results["aspect_ratio"]),
                    "is_1280x720": bool(dimension_results["is_1280x720"]),
                    "is_16:9_ratio": bool(dimension_results["is_16:9_ratio"])
                },
                "brightness": {
                    "brightness_value": float(brightness_results["brightness_value"]),
                    "brightness_percentage": float(brightness_results["brightness_percentage"]),
                    "is_bright": bool(brightness_results["is_bright"]),
                    "brightness_category": brightness_results["brightness_category"]
                },
                "text": {
                    "word_count": text_results["word_count"],
                    "is_concise": bool(text_results.get("is_concise", False)),
                    "readability_score": text_results["readability_score"]
                },
                "color_palette": serializable_palette,
                "visualization": img_base64
            }
            
            if "detected_text" in text_results and text_results["detected_text"]:
                response_data["text"]["detected_text"] = text_results["detected_text"]
            
            return response_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing thumbnail: {str(e)}")

@app.post("/detect-logo/")
async def detect_logo(thumbnail: UploadFile = File(...), logo: UploadFile = File(...)):
    # Create a temporary directory to store the uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_thumbnail_path = Path(temp_dir) / thumbnail.filename
        temp_logo_path = Path(temp_dir) / logo.filename
        
        # Save the uploaded files
        with open(temp_thumbnail_path, "wb") as buffer:
            shutil.copyfileobj(thumbnail.file, buffer)
        
        with open(temp_logo_path, "wb") as buffer:
            shutil.copyfileobj(logo.file, buffer)
        
        try:
            # Load images
            thumbnail_img, logo_img = logo_detector.load_images(
                str(temp_thumbnail_path), 
                str(temp_logo_path)
            )
            
            # Evaluate logo presence
            results = logo_detector.evaluate_logo_presence(thumbnail_img, logo_img)
            
            # Visualize results
            vis_img = logo_detector.visualize_results(thumbnail_img, logo_img, results)
            
            # Convert visualization to base64
            _, buffer = cv2.imencode('.png', vis_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare response data
            response_data = {
                "is_present": results["is_present"],
                "accuracy": float(results["accuracy"]),
                "accuracy_percentage": float(results["accuracy_percentage"]),
                "template_matching": {
                    "score": float(results["template_matching"]["score"]),
                    "location": results["template_matching"]["location"],
                    "dimensions": results["template_matching"]["dimensions"]
                },
                "feature_matching": {
                    "match_ratio": float(results["feature_matching"]["match_ratio"]),
                    "num_matches": results["feature_matching"]["num_matches"]
                },
                "suggestions": results["suggestions"],
                "visualization": img_base64
            }
            
            return response_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error detecting logo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
