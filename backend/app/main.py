from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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

# Import Gemini API module
sys.path.append(str(Path(__file__).parent.parent))
import gemini_api

app = FastAPI(title="ACTIVI Video Thumbnail Validator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yt-thumbnail-validator.vercel.app", "*"],
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
            print(f"[DEBUG] Received file: {thumbnail.filename}")
            print(f"[DEBUG] Saved file at {temp_path}")
            print(f"[DEBUG] File size: {os.path.getsize(temp_path)} bytes")
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
            import traceback
            print("[ERROR] Exception in /analyze-thumbnail:")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error analyzing thumbnail: {str(e)}")

@app.post("/detect-logo/")
async def detect_logo(
    thumbnail: UploadFile = File(...), 
    logo: UploadFile = File(...),
    use_gemini: bool = Form(False),
    api_key: str = Form(None)
):
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
            print(f"[DEBUG] Received thumbnail file: {thumbnail.filename}")
            print(f"[DEBUG] Saved thumbnail at {temp_thumbnail_path}")
            print(f"[DEBUG] Thumbnail file size: {os.path.getsize(temp_thumbnail_path)} bytes")
            print(f"[DEBUG] Received logo file: {logo.filename}")
            print(f"[DEBUG] Saved logo at {temp_logo_path}")
            print(f"[DEBUG] Logo file size: {os.path.getsize(temp_logo_path)} bytes")
            # Check if we should use Gemini API
            if use_gemini:
                print(f"[DEBUG] Using Gemini API for logo detection with API key: {'*****' if api_key else 'None'}")
                
                # Use Gemini API for detection
                gemini_response = gemini_api.detect_logo_with_gemini(
                    str(temp_thumbnail_path),
                    str(temp_logo_path),
                    api_key
                )
                
                # Parse the JSON response from Gemini API
                try:
                    print(f"[DEBUG] Parsing Gemini API response: {gemini_response}")
                    import json
                    results = json.loads(gemini_response)
                    
                    # Ensure we have all required fields
                    if not all(key in results for key in ["is_present", "accuracy", "accuracy_percentage", "suggestions"]):
                        print(f"[ERROR] Missing required fields in Gemini API response")
                        # Set default values for missing fields
                        results.setdefault("is_present", False)
                        results.setdefault("accuracy", 0.0)
                        results.setdefault("accuracy_percentage", 0.0)
                        results.setdefault("suggestions", [])
                        results.setdefault("llm_response", gemini_response)
                except Exception as e:
                    print(f"[ERROR] Failed to parse Gemini API response: {str(e)}")
                    # Create a default results dictionary if parsing fails
                    results = {
                        "is_present": False,
                        "accuracy": 0.0,
                        "accuracy_percentage": 0.0,
                        "suggestions": ["Error parsing Gemini API response"],
                        "llm_response": gemini_response,
                        "error": str(e)
                    }
                
                # Load images for visualization only
                thumbnail_img, logo_img = logo_detector.load_images(
                    str(temp_thumbnail_path), 
                    str(temp_logo_path)
                )
                
                # Create a simplified visualization for Gemini results
                # Just draw a rectangle on the thumbnail to indicate detection
                vis_img = thumbnail_img.copy()
                h, w = vis_img.shape[:2]
                if results["is_present"]:
                    # Draw a green border around the image to indicate detection
                    cv2.rectangle(vis_img, (0, 0), (w-1, h-1), (0, 255, 0), 10)
                    cv2.putText(vis_img, f"Logo detected ({results['accuracy_percentage']:.1f}%)", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Draw a red border
                    cv2.rectangle(vis_img, (0, 0), (w-1, h-1), (0, 0, 255), 10)
                    cv2.putText(vis_img, "Logo not detected", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Use traditional CV methods
                print(f"[DEBUG] Using traditional CV methods for logo detection")
                
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
                "suggestions": results["suggestions"],
                "visualization": img_base64,
                "detection_method": "gemini" if use_gemini else "traditional"
            }
            
            # Add traditional CV specific data if not using Gemini
            if not use_gemini:
                response_data["template_matching"] = {
                    "score": float(results["template_matching"]["score"]),
                    "location": results["template_matching"]["location"],
                    "dimensions": results["template_matching"]["dimensions"]
                }
                response_data["feature_matching"] = {
                    "match_ratio": float(results["feature_matching"]["match_ratio"]),
                    "num_matches": results["feature_matching"]["num_matches"]
                }
            
            # Add Gemini specific data if using Gemini
            if use_gemini:
                if "llm_response" in results:
                    response_data["llm_response"] = results["llm_response"]
                else:
                    response_data["llm_response"] = gemini_response  # Use the raw response if parsed response doesn't have it
            
            return response_data
            
        except Exception as e:
            import traceback
            print("[ERROR] Exception in /detect-logo:")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error detecting logo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
