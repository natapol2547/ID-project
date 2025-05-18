import os
import cv2
import numpy as np
import json

CONFIG_FILE_NAME = "perspective_configs.json"
PERS_OUTPUT_DIR = "perspective_corrected_images"

def load_perspective_configurations(config_path=CONFIG_FILE_NAME):
    """Loads perspective configurations from a JSON file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    try:
        with open(config_path, 'r') as f:
            configurations = json.load(f)
        print(f"Successfully loaded {len(configurations)} configurations from '{config_path}'.")
        return configurations
    except Exception as e:
        print(f"Error loading or parsing configuration file '{config_path}': {e}")
        return None


def apply_perspective_correction(image_path, config, output_dir=PERS_OUTPUT_DIR):
    """
    Loads an image, applies perspective correction based on the given config,
    and saves the result.
    """
    if config is None:
        print(f"Skipping {image_path} as its configuration is missing or invalid.")
        return

    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Could not load image '{image_path}'. Skipping.")
        return

    # Extract parameters from the configuration
    # source_points = np.array(config["source_points"], dtype=np.float32) # Not directly needed if matrix is stored
    matrix = np.array(config["transformation_matrix"], dtype=np.float32)
    output_width, output_height = config["output_dimensions"]

    if matrix is None or output_width <= 0 or output_height <= 0:
        print(f"Invalid transformation parameters for '{image_path}'. Skipping.")
        return

    # Apply the perspective transformation using the stored matrix
    img_warped = cv2.warpPerspective(img_original, matrix, (output_width, output_height))

    # --- Save the corrected image ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}. Cannot save image.")
            return

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    corrected_image_name = f"{name}_corrected{ext}"
    output_path = os.path.join(output_dir, corrected_image_name)

    try:
        cv2.imwrite(output_path, img_warped)
        print(f"  Corrected '{base_name}' -> Saved to '{output_path}'")
    except Exception as e:
        print(f"  Error saving corrected image '{output_path}': {e}")
        
    return output_path
        
def correct_perspective_images(image_paths, config_path=CONFIG_FILE_NAME, output_dir=PERS_OUTPUT_DIR):
    """
    Corrects the perspective of images based on configurations loaded from a JSON file.
    """
    configurations = load_perspective_configurations(config_path)
    output_paths = []

    if configurations:
        for image_path in image_paths:
            # Extract the base name of the image to find its corresponding config
            config_entry = configurations.get(image_path)

            if config_entry is None:
                print(f"No configuration found for '{image_path}'. Skipping.")
                continue

            # Apply perspective correction
            output_paths.append(apply_perspective_correction(image_path, config_entry, output_dir))
    
    return output_paths


if __name__ == "__main__":
    # Example usage
    image_paths = [
        "image_1.png",
        "image_2.png",
        # Add more image paths as needed
    ]
    
    correct_perspective_images(image_paths)