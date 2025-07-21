#!/usr/bin/env python3
"""
Logo Detector for YouTube Thumbnails

This script checks if a logo is present in a YouTube thumbnail image and
evaluates how accurately it appears.
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_images(thumbnail_path, logo_path):
    """
    Load thumbnail and logo images from the given paths.
    
    Args:
        thumbnail_path (str): Path to the thumbnail image
        logo_path (str): Path to the logo image
        
    Returns:
        tuple: (thumbnail_img, logo_img) as numpy arrays
    """
    # Read images
    thumbnail_img = cv2.imread(thumbnail_path)
    logo_img = cv2.imread(logo_path)
    
    # Check if images were loaded successfully
    if thumbnail_img is None:
        raise FileNotFoundError(f"Could not load thumbnail image: {thumbnail_path}")
    if logo_img is None:
        raise FileNotFoundError(f"Could not load logo image: {logo_path}")
    
    return thumbnail_img, logo_img


def template_matching(thumbnail_img, logo_img):
    """
    Perform template matching to find the logo in the thumbnail.
    
    Args:
        thumbnail_img (numpy.ndarray): Thumbnail image
        logo_img (numpy.ndarray): Logo image
        
    Returns:
        tuple: (max_val, max_loc, result) where max_val is the confidence score,
               max_loc is the top-left corner of the detected logo,
               and result is the matching result matrix
    """
    # Convert images to grayscale for template matching
    thumbnail_gray = cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2GRAY)
    logo_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions of the logo
    h, w = logo_gray.shape
    
    # Apply template matching
    result = cv2.matchTemplate(thumbnail_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find the location with maximum correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_val, max_loc, result, (w, h)


def feature_matching(thumbnail_img, logo_img, method="ORB"):
    """
    Perform feature matching to find the logo in the thumbnail using ORB (default) or SIFT.
    
    Args:
        thumbnail_img (numpy.ndarray): Thumbnail image
        logo_img (numpy.ndarray): Logo image
        method (str): Feature detector to use ("ORB" or "SIFT")
    
    Returns:
        tuple: (match_ratio, good_matches, keypoints) where match_ratio is the ratio of good matches,
               good_matches is the list of good matches, and keypoints are the detected keypoints
    """
    # Convert images to grayscale
    thumbnail_gray = cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2GRAY)
    logo_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)

    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(logo_gray, None)
    kp2, des2 = detector.detectAndCompute(thumbnail_gray, None)

    # Check if keypoints were found
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0, [], (kp1, kp2, None)

    if method.upper() == "SIFT":
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        # Brute Force matcher for ORB (Hamming distance)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    match_ratio = len(good_matches) / max(len(kp1), 1)
    return match_ratio, good_matches, (kp1, kp2, matches)



def generate_logo_suggestions(results, thumbnail_img):
    """
    Generate suggestions based on logo detection scores.
    
    Args:
        results (dict): Logo detection results
        
    Returns:
        list: List of suggestions for improving logo visibility
    """
    suggestions = []
    
    # Overall accuracy thresholds
    if results['accuracy'] < 0.2:
        suggestions.append("The logo is barely detectable. Consider using a clearer version of your logo.")
    elif results['accuracy'] < 0.4:
        suggestions.append("Your logo has low visibility. Try placing it in a more prominent position.")
    elif results['accuracy'] < 0.6:
        suggestions.append("Logo visibility could be improved. Consider adjusting its size or position.")
    
    # Template matching specific suggestions
    if results['template_matching']['score'] < 0.1 and results['feature_matching']['match_ratio'] > 0.3:
        suggestions.append("Your logo appears to be modified from the original. Consider using the standard version.")
    
    # Feature matching specific suggestions
    if results['feature_matching']['match_ratio'] < 0.2:
        suggestions.append("Few distinctive logo features were detected. Ensure the logo isn't obscured or heavily modified.")
    
    # Size and placement suggestions based on template matching location
    if results['is_present'] and 'location' in results['template_matching']:
        x, y = results['template_matching']['location']
        width, height = results['template_matching']['dimensions']
        
        # Check if logo is too small
        thumbnail_area = thumbnail_img.shape[0] * thumbnail_img.shape[1]
        logo_area = width * height
        logo_size_ratio = logo_area / thumbnail_area
        
        if logo_size_ratio < 0.05:
            suggestions.append("The logo appears too small. Consider increasing its size for better visibility.")
        elif logo_size_ratio > 0.25:
            suggestions.append("The logo may be too large. Consider reducing its size for better thumbnail composition.")
        
        # Check if logo is in a corner or center
        thumbnail_width = thumbnail_img.shape[1]
        thumbnail_height = thumbnail_img.shape[0]
        center_x = x + width/2
        center_y = y + height/2
        
        # Define regions
        is_left = center_x < thumbnail_width * 0.25
        is_right = center_x > thumbnail_width * 0.75
        is_top = center_y < thumbnail_height * 0.25
        is_bottom = center_y > thumbnail_height * 0.75
        
        if is_left and is_top and results['accuracy'] < 0.5:
            suggestions.append("The logo in the top-left corner has low visibility. Consider using a contrasting background.")
        elif is_right and is_top and results['accuracy'] < 0.5:
            suggestions.append("The logo in the top-right corner has low visibility. Consider using a contrasting background.")
        elif is_left and is_bottom and results['accuracy'] < 0.5:
            suggestions.append("The logo in the bottom-left corner has low visibility. Consider using a contrasting background.")
        elif is_right and is_bottom and results['accuracy'] < 0.5:
            suggestions.append("The logo in the bottom-right corner has low visibility. Consider using a contrasting background.")
    
    # If no specific suggestions, provide general ones based on accuracy
    if not suggestions:
        if results['accuracy'] < 0.7:
            suggestions.append("Consider improving logo visibility for better brand recognition.")
        else:
            suggestions.append("Logo visibility is good.")
    
    return suggestions


def evaluate_logo_presence(thumbnail_img, logo_img, feature_method="ORB"):
    """
    Evaluate if the logo is present in the thumbnail and how accurately.
    
    Args:
        thumbnail_img (numpy.ndarray): Thumbnail image
        logo_img (numpy.ndarray): Logo image
        feature_method (str): Feature detector to use for matching ("ORB" or "SIFT")
    
    Returns:
        dict: Results containing presence information and accuracy scores
    """
    results = {}
    
    # Perform template matching
    tm_score, tm_loc, tm_result, (logo_width, logo_height) = template_matching(thumbnail_img, logo_img)
    results['template_matching'] = {
        'score': tm_score,
        'location': tm_loc,
        'dimensions': (logo_width, logo_height)
    }
    
    # Perform feature matching
    fm_ratio, fm_matches, (kp1, kp2, matches) = feature_matching(thumbnail_img, logo_img, method=feature_method)
    results['feature_matching'] = {
        'match_ratio': fm_ratio,
        'num_matches': len(fm_matches)
    }
    
    # Determine if logo is present based on combined evidence
    # Template matching threshold
    tm_threshold = 0.7
    # Feature matching threshold (minimum number of good matches)
    fm_threshold = 0.1
    
    is_present_tm = tm_score >= tm_threshold
    is_present_fm = fm_ratio >= fm_threshold
    
    # Combine results (consider logo present if either method detects it)
    results['is_present'] = is_present_tm or is_present_fm
    
    # Calculate overall accuracy score (weighted average of both methods)
    tm_weight = 0.6
    fm_weight = 0.4
    
    # Normalize feature matching score to 0-1 range
    normalized_fm_score = min(fm_ratio * 2, 1.0)
    
    results['accuracy'] = tm_weight * tm_score + fm_weight * normalized_fm_score
    
    # Convert accuracy to percentage for display
    results['accuracy_percentage'] = results['accuracy'] * 100
    
    # Generate suggestions based on the results
    results['suggestions'] = generate_logo_suggestions(results, thumbnail_img)
    
    return results


def visualize_results(thumbnail_img, logo_img, results):
    """
    Visualize the logo detection results.
    
    Args:
        thumbnail_img (numpy.ndarray): Thumbnail image
        logo_img (numpy.ndarray): Logo image
        results (dict): Detection results
        
    Returns:
        numpy.ndarray: Image with visualization
    """
    # Create a copy of the thumbnail for visualization
    vis_img = thumbnail_img.copy()
    
    # Draw rectangle around the detected logo using template matching
    if results['is_present']:
        tm_loc = results['template_matching']['location']
        logo_width, logo_height = results['template_matching']['dimensions']
        
        # Draw rectangle
        cv2.rectangle(
            vis_img,
            tm_loc,
            (tm_loc[0] + logo_width, tm_loc[1] + logo_height),
            (0, 255, 0),
            2
        )
        
        # Add text with confidence score
        cv2.putText(
            vis_img,
            f"Accuracy: {results['accuracy']:.2f}",
            (tm_loc[0], tm_loc[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    else:
        # Add text indicating logo not found
        cv2.putText(
            vis_img,
            "Logo not found",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    
    return vis_img


def main():
    """Main function to run the logo detector. Supports ORB (default) or SIFT for feature matching."""
    parser = argparse.ArgumentParser(description='Detect logo in YouTube thumbnail')
    parser.add_argument('thumbnail', help='Path to the YouTube thumbnail image')
    parser.add_argument('logo', help='Path to the logo image')
    parser.add_argument('--output', '-o', help='Path to save the visualization result')
    parser.add_argument('--show', '-s', action='store_true', help='Show the visualization')
    parser.add_argument('--feature-method', choices=["ORB", "SIFT"], default="ORB", help='Feature detection method to use (ORB or SIFT, default: ORB)')
    
    args = parser.parse_args()
    
    try:
        # Load images
        thumbnail_img, logo_img = load_images(args.thumbnail, args.logo)
        
        # Evaluate logo presence
        results = evaluate_logo_presence(thumbnail_img, logo_img, feature_method=args.feature_method)
        
        # Print results
        print(f"Logo presence: {'Detected' if results['is_present'] else 'Not detected'}")
        print(f"Accuracy score: {results['accuracy']:.4f}")
        print(f"Template matching score: {results['template_matching']['score']:.4f}")
        print(f"Feature matching ratio: {results['feature_matching']['match_ratio']:.4f}")
        print(f"Number of feature matches: {results['feature_matching']['num_matches']}")
        
        # Visualize results
        vis_img = visualize_results(thumbnail_img, logo_img, results)
        
        # Show visualization if requested
        if args.show:
            plt.figure(figsize=(12, 8))
            plt.subplot(131)
            plt.title('Thumbnail')
            plt.imshow(cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB))
            
            plt.subplot(132)
            plt.title('Logo')
            plt.imshow(cv2.cvtColor(logo_img, cv2.COLOR_BGR2RGB))
            
            plt.subplot(133)
            plt.title('Detection Result')
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            
            plt.tight_layout()
            plt.show()
        
        # Save visualization if output path is provided
        if args.output:
            output_path = Path(args.output)
            cv2.imwrite(str(output_path), vis_img)
            print(f"Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
