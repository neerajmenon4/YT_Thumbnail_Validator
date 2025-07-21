#!/usr/bin/env python3
"""
Image Analyzer for YouTube Thumbnails

This script analyzes an image to determine:
1. Color palette analysis
2. Brightness percentage (bright or dull)
3. Verification of 1280x720 size with 16:9 aspect ratio
4. Basic text readability assessment
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pytesseract
from PIL import Image


def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        tuple: (image as numpy array, PIL image)
    """
    # Read image with OpenCV (for analysis)
    cv_img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if cv_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Also load with PIL for some operations
    pil_img = Image.open(image_path)
    
    return cv_img, pil_img


def check_dimensions(image):
    """
    Check if the image has 1280x720 dimensions with 16:9 aspect ratio.
    
    Args:
        image (numpy.ndarray): Image as numpy array
        
    Returns:
        dict: Results containing dimension information
    """
    height, width = image.shape[:2]
    expected_width, expected_height = 1280, 720
    expected_aspect_ratio = 16/9
    
    # Calculate actual aspect ratio
    actual_aspect_ratio = width / height
    
    # Check if dimensions match expected
    is_correct_size = (width == expected_width and height == expected_height)
    
    # Check if aspect ratio is 16:9 (with small tolerance)
    aspect_ratio_tolerance = 0.01
    is_correct_ratio = abs(actual_aspect_ratio - expected_aspect_ratio) < aspect_ratio_tolerance
    
    return {
        'width': width,
        'height': height,
        'aspect_ratio': actual_aspect_ratio,
        'is_1280x720': is_correct_size,
        'is_16:9_ratio': is_correct_ratio
    }


def extract_color_palette(image, n_colors=5):
    """
    Extract the dominant color palette from the image.
    
    Args:
        image (numpy.ndarray): Image as numpy array
        n_colors (int): Number of colors to extract
        
    Returns:
        list: List of dominant colors in RGB format
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Convert from BGR to RGB
    pixels = pixels[:, ::-1]
    
    # Perform K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    
    # Convert to integer RGB values
    colors = colors.astype(int)
    
    # Calculate percentage of each color
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    
    # Create a list of (color, percentage) tuples
    color_percentages = [(colors[i], percentages[i]) for i in range(len(colors))]
    
    # Sort by percentage (descending)
    color_percentages.sort(key=lambda x: x[1], reverse=True)
    
    return color_percentages


def analyze_brightness(image):
    """
    Analyze the brightness of the image.
    
    Args:
        image (numpy.ndarray): Image as numpy array
        
    Returns:
        dict: Results containing brightness information
    """
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the V channel (brightness)
    v_channel = hsv_img[:, :, 2]
    
    # Calculate average brightness (0-255)
    avg_brightness = np.mean(v_channel)
    
    # Convert to percentage (0-100%)
    brightness_percentage = (avg_brightness / 255) * 100
    
    # Determine if image is bright or dull
    # Threshold can be adjusted based on preference
    brightness_threshold = 50
    is_bright = brightness_percentage >= brightness_threshold
    
    return {
        'brightness_value': avg_brightness,
        'brightness_percentage': brightness_percentage,
        'is_bright': is_bright,
        'brightness_category': 'Bright' if is_bright else 'Dull'
    }


def analyze_text_readability(pil_img):
    """
    Analyze text readability in the image.
    
    Args:
        pil_img (PIL.Image): Image as PIL Image object
        
    Returns:
        dict: Results containing text readability information
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("image_analyzer.text_readability")

    try:
        logger.info("Starting OCR text extraction...")
        text = pytesseract.image_to_string(pil_img)
        logger.info(f"Extracted text: {text.strip()}")

        words = text.split()
        word_count = len(words)
        char_count = sum(len(word) for word in words)
        avg_word_length = char_count / max(word_count, 1)
        is_concise = word_count <= 15  # Arbitrary threshold for conciseness
        long_words = [word for word in words if len(word) > 10]
        has_long_words = len(long_words) > 0

        logger.info(f"Word count: {word_count}, Avg word length: {avg_word_length:.2f}, Is concise: {is_concise}, Has long words: {has_long_words}")

        result = {
            'detected_text': text.strip(),
            'word_count': word_count,
            'is_concise': is_concise,
            'avg_word_length': avg_word_length,
            'has_long_words': has_long_words,
            'readability_score': 'Good' if is_concise and not has_long_words else 'Could be improved'
        }
        logger.info(f"Text readability result: {result}")
        return result
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        return {
            'error': f"Text analysis failed: {str(e)}",
            'detected_text': '',
            'word_count': 0,
            'is_concise': False,
            'avg_word_length': 0,
            'has_long_words': False,
            'readability_score': 'Unknown'
        }


def visualize_results(image, color_palette, results, output_path=None):
    """
    Visualize the analysis results.
    
    Args:
        image (numpy.ndarray): Image as numpy array
        color_palette (list): List of dominant colors
        results (dict): Analysis results
        output_path (str, optional): Path to save the visualization
        
    Returns:
        None
    """
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Plot color palette
    plt.subplot(2, 2, 2)
    plt.title('Dominant Color Palette')
    
    # Create a bar for each color
    for i, (color, percentage) in enumerate(color_palette):
        plt.bar(
            i, percentage, 
            color=color/255, 
            width=0.8,
            label=f"RGB{tuple(color)} ({percentage:.1f}%)"
        )
    
    plt.xticks([])
    plt.ylabel('Percentage (%)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.0))
    
    # Plot brightness histogram
    plt.subplot(2, 2, 3)
    plt.title(f"Brightness: {results['brightness']['brightness_percentage']:.1f}% ({results['brightness']['brightness_category']})")
    
    # Convert to grayscale for histogram
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.hist(gray_img.ravel(), 256, [0, 256], color='gray', alpha=0.7)
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Display results as text
    plt.subplot(2, 2, 4)
    plt.title('Analysis Results')
    plt.axis('off')
    
    # Format results as text
    result_text = [
        f"Dimensions: {results['dimensions']['width']}x{results['dimensions']['height']}",
        f"Expected: 1280x720 (16:9)",
        f"Correct Size: {'✓' if results['dimensions']['is_1280x720'] else '✗'}",
        f"Correct Ratio: {'✓' if results['dimensions']['is_16:9_ratio'] else '✗'}",
        f"Brightness: {results['brightness']['brightness_percentage']:.1f}%",
        f"Category: {results['brightness']['brightness_category']}",
        f"\nText Readability: {results['text']['readability_score']}",
        f"Word Count: {results['text']['word_count']}",
        f"Concise: {'✓' if results.get('text', {}).get('is_concise', False) else '✗'}"
    ]
    
    plt.text(0.1, 0.9, '\n'.join(result_text), fontsize=12, va='top')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Only show the plot if not being used in a web context
    # When used in web app, the figure will be captured and converted to base64


def main():
    """Main function to run the image analyzer."""
    parser = argparse.ArgumentParser(description='Analyze image properties for YouTube thumbnails')
    parser.add_argument('image', help='Path to the image to analyze')
    parser.add_argument('--output', '-o', help='Path to save the visualization result')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--colors', '-c', type=int, default=5, help='Number of colors to extract (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Load image
        cv_img, pil_img = load_image(args.image)
        
        # Check dimensions
        dimension_results = check_dimensions(cv_img)
        
        # Extract color palette
        color_palette = extract_color_palette(cv_img, n_colors=args.colors)
        
        # Analyze brightness
        brightness_results = analyze_brightness(cv_img)
        
        # Analyze text readability
        text_results = analyze_text_readability(pil_img)
        
        # Combine results
        results = {
            'dimensions': dimension_results,
            'brightness': brightness_results,
            'text': text_results
        }
        
        # Print results
        print("\n=== Image Analysis Results ===")
        print(f"\nDimensions: {dimension_results['width']}x{dimension_results['height']}")
        print(f"Aspect Ratio: {dimension_results['aspect_ratio']:.2f}")
        print(f"Is 1280x720: {'Yes' if dimension_results['is_1280x720'] else 'No'}")
        print(f"Is 16:9 Ratio: {'Yes' if dimension_results['is_16:9_ratio'] else 'No'}")
        
        print("\nBrightness Analysis:")
        print(f"Brightness: {brightness_results['brightness_percentage']:.2f}%")
        print(f"Category: {brightness_results['brightness_category']}")
        
        print("\nDominant Colors (RGB):")
        for i, (color, percentage) in enumerate(color_palette, 1):
            print(f"Color {i}: RGB{tuple(color)} - {percentage:.2f}%")
        
        print("\nText Readability:")
        print(f"Word Count: {text_results['word_count']}")
        print(f"Is Concise: {'Yes' if text_results.get('is_concise', False) else 'No'}")
        print(f"Readability Score: {text_results['readability_score']}")
        
        if 'detected_text' in text_results and text_results['detected_text']:
            print(f"\nDetected Text:\n{text_results['detected_text']}")
        
        # Visualize results if not disabled
        if not args.no_vis:
            visualize_results(cv_img, color_palette, results, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
