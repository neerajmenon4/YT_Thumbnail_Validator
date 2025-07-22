"""
Gemini API integration for logo detection in YouTube thumbnails.
This module provides functionality to use Google's Gemini API to detect logos in images.
"""

import os
import base64
import google.genai as genai
from typing import Dict, Any, Optional
from pydantic import BaseModel

class LogoDetectionResponse(BaseModel):
    is_present: bool
    accuracy: float
    accuracy_percentage: float
    llm_response: str
    suggestions: list[str]

def configure_genai(api_key: Optional[str] = None):
    """
    Configure the Google Generative AI client with the API key.
    
    Args:
        api_key: The API key to use. If None, will try to use GEMINI_API_KEY environment variable.
    """
    # Use provided API key or fall back to environment variable
    key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if not key:
        raise ValueError("No API key provided. Please provide an API key or set the GEMINI_API_KEY environment variable.")
    
    return genai.Client(api_key=key)

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def detect_logo_with_gemini(
    thumbnail_path: str, 
    logo_path: str, 
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect if a logo is present in a thumbnail using Google's Gemini API.
    
    Args:
        thumbnail_path: Path to the thumbnail image.
        logo_path: Path to the logo image.
        api_key: Optional API key to use. If not provided, will use environment variable.
        
    Returns:
        Dictionary with detection results.
    """
    try:
        print(f"[DEBUG] Starting Gemini API logo detection")
        print(f"[DEBUG] Thumbnail path: {thumbnail_path}")
        print(f"[DEBUG] Logo path: {logo_path}")
        print(f"[DEBUG] API key provided: {bool(api_key)}")
        
        # Verify files exist
        if not os.path.exists(thumbnail_path):
            raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_path}")
        if not os.path.exists(logo_path):
            raise FileNotFoundError(f"Logo file not found: {logo_path}")
            
        print(f"[DEBUG] Thumbnail file size: {os.path.getsize(thumbnail_path)} bytes")
        print(f"[DEBUG] Logo file size: {os.path.getsize(logo_path)} bytes")
        
        # Configure the API and get client
        print(f"[DEBUG] Configuring Gemini API client")
        client = configure_genai(api_key)
        
        # Encode images to base64
        print(f"[DEBUG] Encoding images to base64")
        thumbnail_b64 = encode_image_to_base64(thumbnail_path)
        logo_b64 = encode_image_to_base64(logo_path)
        
        print(f"[DEBUG] Thumbnail base64 length: {len(thumbnail_b64)}")
        print(f"[DEBUG] Logo base64 length: {len(logo_b64)}")
        
        # Generate the prompt
        prompt = """
        I'm going to show you two images: a YouTube thumbnail and a logo.
        
        Please analyze if the logo appears in the thumbnail image. Consider:
        1. The logo might appear in different sizes, positions, or with slight variations
        2. The logo might be partially visible or integrated with other elements
        3. The logo might have different background colors or slight color variations
        
        Provide a detailed analysis with:
        - Whether the logo is present (yes/no)
        - Confidence level (0-100%)
        - Location description (if found)
        - Any modifications to the logo in the thumbnail
        - Suggestions for better logo placement or visibility (if applicable)
        """
        
        # Set up the model
        # print(f"[DEBUG] Setting up Gemini model")
        # model = client.GenerativeModel('gemini-1.5-flash')  # Using gemini-1.5-flash instead of 2.5
        
        # Create the request with both images
        print(f"[DEBUG] Sending request to Gemini API")
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[
                    genai.types.Content(
                        role="user",
                        parts=[
                            genai.types.Part(text=prompt),
                            genai.types.Part.from_bytes(
                                mime_type="image/png" if thumbnail_path.lower().endswith('.png') else "image/jpeg",
                                data=base64.b64decode(thumbnail_b64)
                            ),
                            genai.types.Part.from_bytes(
                                mime_type="image/png" if logo_path.lower().endswith('.png') else "image/jpeg",
                                data=base64.b64decode(logo_b64)
                            )
                        ]
                    )
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": LogoDetectionResponse,
                }
            )
            print(f"[DEBUG] Received response from Gemini API")
        except Exception as api_error:
            print(f"[ERROR] Gemini API request failed: {str(api_error)}")
            raise
        
        # Process the response
        text_response = response.text
        print(f"[DEBUG] Gemini response: {text_response}")
        
        # # Extract key information from the response
        # is_present = "yes" in text_response.lower() and not ("no" in text_response.lower() and "not present" in text_response.lower())
        
        # # Try to extract confidence level from the text
        # confidence = 0.0
        # confidence_percentage = 0.0
        
        # # Simple confidence extraction (can be improved with more sophisticated parsing)
        # if "confidence" in text_response.lower():
        #     confidence_text = text_response.lower().split("confidence")[1].split("\n")[0]
        #     # Try to extract a percentage
        #     import re
        #     percentage_match = re.search(r'(\d+)%', confidence_text)
        #     if percentage_match:
        #         confidence_percentage = float(percentage_match.group(1))
        #         confidence = confidence_percentage / 100.0
        #     else:
        #         # Look for decimal numbers
        #         decimal_match = re.search(r'(\d+\.\d+)', confidence_text)
        #         if decimal_match:
        #             confidence = float(decimal_match.group(1))
        #             if confidence > 0 and confidence <= 1:
        #                 confidence_percentage = confidence * 100.0
        #             else:
        #                 confidence_percentage = confidence
        #                 confidence = confidence / 100.0
        
        # # Extract suggestions
        # suggestions = []
        # if "suggestion" in text_response.lower():
        #     suggestion_text = text_response.lower().split("suggestion")[1].split("\n")
        #     for line in suggestion_text:
        #         if line.strip() and ":" not in line and len(line) > 10:
        #             suggestions.append(line.strip().capitalize())
        
        # # Prepare the result
        # result = {
        #     "is_present": is_present,
        #     "accuracy": confidence,
        #     "accuracy_percentage": confidence_percentage,
        #     "llm_response": text_response,
        #     "suggestions": suggestions[:3]  # Limit to top 3 suggestions
        # }
        
        return text_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] Exception in detect_logo_with_gemini: {str(e)}")
        print(f"[ERROR] Traceback: {error_traceback}")
        return {
            "is_present": False,
            "accuracy": 0.0,
            "accuracy_percentage": 0.0,
            "error": str(e),
            "error_traceback": error_traceback,
            "llm_response": f"Error processing request: {str(e)}",
            "suggestions": ["Error occurred during logo detection"]
        }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python gemini_api.py <thumbnail_path> <logo_path>")
        sys.exit(1)
    
    thumbnail_path = sys.argv[1]
    logo_path = sys.argv[2]
    
    result = detect_logo_with_gemini(thumbnail_path, logo_path)
    print(f"Logo present: {result['is_present']}")
    print(f"Confidence: {result['accuracy_percentage']}%")
    print(f"Suggestions: {result['suggestions']}")
    print("\nFull response:")
    print(result['llm_response'])
