import cv2
import os

SPLIT_IMAGE_DIR = "split_images"

def split_image(image_paths, output_dir=SPLIT_IMAGE_DIR, rows=1, cols=1):
    # --- Save the corrected image ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}. Cannot save image.")
            return
        
    output_paths = []
        
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from '{image_path}'. Check if it's a valid image file. Skipping.")
            return False

        img_height, img_width, channels = img.shape
        if channels not in [1, 3, 4]: # Grayscale, BGR, BGRA
            print(f"Warning: Image '{image_path}' has {channels} channels. Expected 1 (grayscale), 3 (BGR), or 4 (BGRA). Proceeding, but results might be unexpected.")

        # print(f"Input image ('{os.path.basename(image_path)}') dimensions: {img_width}x{img_height}")

        tile_height = img_height // rows
        tile_width = img_width // cols

        if tile_height == 0 or tile_width == 0:
            print(f"Error: Cannot split image '{os.path.basename(image_path)}'. Resulting tile dimension would be zero.")
            print(f"  Image size: {img_width}x{img_height}, Target rows: {rows}, Target cols: {cols}")
            print(f"  Calculated tile size: {tile_width}x{tile_height}. Skipping this image.")
            return False

        # print(f"Calculated tile dimensions for '{os.path.basename(image_path)}': {tile_width}x{tile_height}")

        tile_count = 0
        original_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        image_extension = os.path.splitext(image_path)[1]
        if not image_extension:
            image_extension = ".png" # Default extension

        for r in range(rows):
            for c in range(cols):
                # Calculate coordinates for the current tile
                y1 = r * tile_height
                y2 = (r + 1) * tile_height
                x1 = c * tile_width
                x2 = (c + 1) * tile_width

                # Ensure y2 and x2 do not exceed image dimensions for the last row/column
                if r == rows - 1: # Last row
                    y2 = img_height
                if c == cols - 1: # Last column
                    x2 = img_width

                # Slice the image to get the tile
                tile = img[y1:y2, x1:x2]

                # Construct filename for the tile
                tile_filename = f"{original_filename_base}_R{r}_C{c}{image_extension}"
                output_path = os.path.join(output_dir, tile_filename)
                # print(output_path)

                # Save the tile
                try:
                    cv2.imwrite(output_path, tile)
                    output_paths.append(output_path)
                    tile_count += 1
                    # print(f"Saved tile: {output_path}") # Uncomment for very verbose logging
                except Exception as e:
                    print(f"Error saving tile {output_path}: {e}")
    
        print(f"Successfully split '{os.path.basename(image_path)}' into {tile_count} tiles in '{output_path}'.")
    
    return output_paths

    
    return True