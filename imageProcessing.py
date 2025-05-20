import cv2
import os
import math # Need this for ceiling function
import typing
import numpy as np
from pyzbar import pyzbar

SPLIT_IMAGE_DIR = "split_images"

def process_image(image_path:str, middle_shit=False)->str:
    PATH = image_path
    img = cv2.imread(image_path)
    new = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    highMask = new[:,:,2] > new[:,:,2].mean() + new[:,:,2].std()/4
    if middle_shit:
        new[highMask] = new[highMask]*0.8

    blur = cv2.GaussianBlur(new, (15, 15), 0)
    thresh_v = cv2.adaptiveThreshold(
        blur[:, :, 2],
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        0
    )
    new[:, :, 2] = thresh_v

    avgValue = new[:,:,2].mean()
    _, thresh_v = cv2.threshold(new[:,:,2], int(avgValue), 255, cv2.THRESH_BINARY_INV)
    new[:,:,2] = thresh_v

    new = cv2.cvtColor(new, cv2.COLOR_HSV2BGR) #these three lines can me refactored into one, im too lazy tho
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR) 
    iterations = max(min(img.shape[0], img.shape[1]) // 500 ,1)# adaptive iteration
    new = cv2.erode(new, None, iterations=iterations) # * lower iterations to 1 if image is small
    new = cv2.dilate(new, None, iterations=iterations) # * lower iterations to 1 if image is small
    writePath = os.path.splitext(PATH)[0]+"_processed"+os.path.splitext(PATH)[1]
    cv2.imwrite(writePath,new)
    return writePath

def split_image(image_paths, output_dir=SPLIT_IMAGE_DIR, rows=1, cols=1, overlap_percent=0)->typing.Union[list, bool]:
    """
    Splits images into tiles with optional overlapping.

    Args:
        image_paths (list): A list of paths to the input images.
        output_dir (str): Directory to save the split tiles.
        rows (int): Number of rows to split the image into.
        cols (int): Number of columns to split the image into.
        overlap_percent (float): Percentage of overlap between adjacent tiles (0-100).

    Returns:
        list: A list of paths to the saved output tiles, or False if an error occurred.
    """
    # --- Input validation ---
    if not all(isinstance(path, str) for path in image_paths) or not image_paths:
        print("Error: 'image_paths' must be a non-empty list of strings.")
        return False
        
    if not isinstance(output_dir, str) or not output_dir:
        print("Error: 'output_dir' must be a non-empty string.")
        return False

    if not isinstance(rows, int) or rows < 1:
        print("Error: 'rows' must be an integer greater than or equal to 1.")
        return False

    if not isinstance(cols, int) or cols < 1:
        print("Error: 'cols' must be an integer greater than or equal to 1.")
        return False
        
    if not isinstance(overlap_percent, (int, float)) or overlap_percent < 0 or overlap_percent > 100:
         print("Error: 'overlap_percent' must be a number between 0 and 100.")
         return False

    # --- Create output directory ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}. Cannot save image.")
            return False

    output_paths = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from '{image_path}'. Check if it's a valid image file. Skipping.")
            continue # Use continue to process other images in the list if one fails

        img_height, img_width = img.shape[:2] # Get height and width, ignore channels for now

        # print(f"Input image ('{os.path.basename(image_path)}') dimensions: {img_width}x{img_height}")

        overlap_ratio = overlap_percent / 100.0

        # Calculate required tile dimensions based on number of rows/cols and overlap
        # This ensures the tiles cover the whole image with the specified overlap
        if rows > 1:
            # Formula: total_height = rows * tile_height - (rows - 1) * overlap_height
            # Rearranging for tile_height: tile_height = total_height / (rows - (rows - 1) * overlap_ratio)
            denominator_h = rows - (rows - 1) * overlap_ratio
            if denominator_h <= 0: # Should not happen with valid inputs, but check
                 print(f"Error: Invalid overlap calculation for height on '{os.path.basename(image_path)}'. Skipping.")
                 continue
            tile_height_float = img_height / denominator_h
        else:
            tile_height_float = img_height # No vertical split, tile is full height

        if cols > 1:
            # Formula: total_width = cols * tile_width - (cols - 1) * overlap_width
            # Rearranging for tile_width: tile_width = total_width / (cols - (cols - 1) * overlap_ratio)
            denominator_w = cols - (cols - 1) * overlap_ratio
            if denominator_w <= 0: # Should not happen with valid inputs, but check
                 print(f"Error: Invalid overlap calculation for width on '{os.path.basename(image_path)}'. Skipping.")
                 continue
            tile_width_float = img_width / denominator_w
        else:
            tile_width_float = img_width # No horizontal split, tile is full width

        # Calculate overlap in pixels
        overlap_h_pixels_float = tile_height_float * overlap_ratio
        overlap_w_pixels_float = tile_width_float * overlap_ratio

        # Calculate step size for moving to the next tile's starting point
        step_height_float = tile_height_float - overlap_h_pixels_float
        step_width_float = tile_width_float - overlap_w_pixels_float

        # If step size becomes zero or negative due to extreme overlap (should be caught by denominator check),
        # this logic might need adjustment, but with 0-100% overlap, step should be >= 0.
        # If step is 0, it means infinite overlap, which isn't practical for splitting.
        # Let's assume overlap_percent < 100 if rows/cols > 1.

        # print(f"Calculated tile dimensions: {tile_width_float:.2f}x{tile_height_float:.2f}")
        # print(f"Calculated overlap pixels: {overlap_w_pixels_float:.2f}x{overlap_h_pixels_float:.2f}")
        # print(f"Calculated step size: {step_width_float:.2f}x{step_height_float:.2f}")


        tile_count = 0
        original_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        image_extension = os.path.splitext(image_path)[1]
        if not image_extension:
            image_extension = ".png" # Default extension if no extension

        # Loop through the grid positions (rows x cols)
        for r in range(rows):
            for c in range(cols):
                # Calculate floating point coordinates for the current tile based on step size
                y1_float = r * step_height_float
                x1_float = c * step_width_float
                y2_float = y1_float + tile_height_float
                x2_float = x1_float + tile_width_float

                # Convert to integer coordinates by rounding
                y1 = int(round(y1_float))
                x1 = int(round(x1_float))
                y2 = int(round(y2_float))
                x2 = int(round(x2_float))

                # Clamp coordinates to image boundaries
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(img_height, y2)
                x2 = min(img_width, x2)

                # Ensure valid crop dimensions (can happen with extreme rounding or edge cases)
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Calculated tile dimensions for R{r}_C{c} resulted in zero or negative size ({x2-x1}x{y2-y1}). Skipping this tile.")
                    continue

                # Slice the image to get the tile
                tile = img[y1:y2, x1:x2]

                # Construct filename for the tile
                # We use the grid index (R_C) for the filename for clarity
                tile_filename = f"{original_filename_base}_R{r}_C{c}{image_extension}"
                output_path = os.path.join(output_dir, tile_filename)

                # Save the tile
                try:

                    # # Convert the image to grayscale
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    # tile = cv2.GaussianBlur(tile, (5, 5), 0)
                    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    # tile = cv2.filter2D(tile, -1, kernel)

                    # # Now apply thresholding to the grayscale image
                    # thresh, tile = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    # tile = cv2.adaptiveThreshold(tile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

                    cv2.imwrite(output_path, tile)
                    output_paths.append(output_path)
                    tile_count += 1
                    # print(f"Saved tile: {output_path} (Crop: [{y1}:{y2}, {x1}:{x2}])") # Uncomment for verbose logging
                except Exception as e:
                    print(f"Error saving tile {output_path}: {e}")

        print(f"Successfully split '{os.path.basename(image_path)}' into {tile_count} tiles in '{output_dir}'.")

    # Check if any paths were added. If not, and image_paths wasn't empty, something went wrong.
    if not output_paths and image_paths:
        print("Splitting process finished, but no output files were generated. Check for errors above.")
        return False

    return output_paths

def detect_and_order_qrcodes(image_path, num_expected_qrcodes=5):
    """
    Detects multiple QR codes in an image, extracts their data and orientation,
    and orders them based on their horizontal center position.

    Args:
        image_path (str): The path to the image file containing QR codes.
        num_expected_qrcodes (int): The number of QR codes expected in the image.
                                    The output list will have this many elements.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'center',
              'data', and 'orientation' for a detected QR code.
              If no QR code is found for an expected position, it will be None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return [None] * num_expected_qrcodes

    # Convert the image to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect QR codes
    detected_qrcodes = pyzbar.decode(gray)

    qr_info_list = []
    for qrcode in detected_qrcodes:
        # The bounding box corners
        (x, y, w, h) = qrcode.rect
        center_x = x + w // 2
        center_y = y + h // 2

        # Decode the QR code data
        data = qrcode.data.decode("utf-8")

        # Orientation can be inferred from the `polygon` attribute or `rect`
        # For simplicity, we'll just note its presence and data for now.
        # If you need precise orientation (e.g., angle), it's a bit more complex
        # and might involve analyzing the QR code's finder patterns.
        # For this example, we'll assume a "standard" orientation unless you specify more.
        orientation = "standard" # You can refine this if needed

        qr_info_list.append({
            'center': (center_x, center_y),
            'data': data,
            'orientation': orientation
        })

        # Optional: Draw bounding box and text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sort detected QR codes by their horizontal center position (x-coordinate)
    qr_info_list.sort(key=lambda item: item['center'][0])

    # Create the final ordered list with Nones for missing QR codes
    final_ordered_qrcodes = [None] * num_expected_qrcodes

    # This part assumes a relatively even distribution. If QR codes are very
    # unevenly spaced or there are false positives, you might need more
    # sophisticated logic to map detected codes to expected indices.
    # For a row of 5, simple sorting and then filling should work if they are distinct.
    for i, qr_info in enumerate(qr_info_list):
        if i < num_expected_qrcodes:
            final_ordered_qrcodes[i] = qr_info
        else:
            # If more QR codes are detected than expected, we'll ignore the extras
            # or you could handle this differently if it's an error condition.
            break

    # Optional: Display the image with detections (useful for debugging)
    # cv2.imshow("Detected QR Codes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final_ordered_qrcodes