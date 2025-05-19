import cv2
import numpy as np
import json
import os
import argparse

# Global variables for a single image's processing
points_src_current_image = []
img_display_current_image = None
img_original_current_image = None
current_image_index = 0
all_image_configurations = [] # List to store configs for each image

WINDOW_NAME_SELECT = "Select 4 Points (TL, TR, BR, BL), then 'n' for Next or 'd' for Done with this image"
WINDOW_NAME_CORRECTED = "Corrected Perspective (Preview)"
CONFIG_FILE_NAME = "perspective_configs.json"

def select_points_callback(event, x, y, flags, param):
    """Mouse callback function to select points for the current image."""
    global points_src_current_image, img_display_current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_src_current_image) < 4:
            points_src_current_image.append((x, y))
            cv2.circle(img_display_current_image, (x, y), 5, (0, 255, 0), -1)
            if len(points_src_current_image) > 1:
                cv2.line(img_display_current_image, points_src_current_image[-2], points_src_current_image[-1], (0, 255, 0), 2)
            if len(points_src_current_image) == 4:
                cv2.line(img_display_current_image, points_src_current_image[-1], points_src_current_image[0], (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME_SELECT, img_display_current_image)
            print(f"Image {current_image_index + 1}: Point {len(points_src_current_image)} selected: ({x}, {y})")
            if len(points_src_current_image) == 4:
                print(f"Image {current_image_index + 1}: 4 points selected. Press 'd' (Done with this image), 'p' (Preview correction), or 'r' (Reset for this image).")
        else:
            print(f"Image {current_image_index + 1}: Already selected 4 points. Press 'd', 'p', or 'r'.")

def calculate_output_dimensions(src_pts):
    """Calculates reasonable output width and height."""
    if not src_pts or len(src_pts) != 4: return 500, 500 # Default
    tl, tr, br, bl = src_pts
    width_top = np.linalg.norm(np.array(tr) - np.array(tl))
    width_bottom = np.linalg.norm(np.array(br) - np.array(bl))
    max_width = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(np.array(bl) - np.array(tl))
    height_right = np.linalg.norm(np.array(br) - np.array(tr))
    max_height = int(max(height_left, height_right))
    return max_width if max_width > 0 else 500, max_height if max_height > 0 else 500

def preview_or_get_corrected_image(image_to_warp, src_points_for_image, for_preview=True):
    """Applies perspective transform. If for_preview, displays it. Else, returns warped image."""
    if len(src_points_for_image) != 4:
        print("Cannot correct: 4 source points needed.")
        return None

    pts_src_arr = np.array(src_points_for_image, dtype=np.float32)
    out_width, out_height = calculate_output_dimensions(src_points_for_image)
    pts_dst_arr = np.array([
        [0, 0], [out_width - 1, 0],
        [out_width - 1, out_height - 1], [0, out_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts_src_arr, pts_dst_arr)
    img_warped = cv2.warpPerspective(image_to_warp, matrix, (out_width, out_height))

    if for_preview:
        cv2.imshow(WINDOW_NAME_CORRECTED, img_warped)
        print(f"Previewing corrected perspective for image {current_image_index + 1}. Press any key in preview window to close it.")
        cv2.waitKey(0)
        cv2.destroyWindow(WINDOW_NAME_CORRECTED) # Close preview window
    return img_warped, matrix.tolist(), (out_width, out_height)


def reset_points_for_current_image():
    global points_src_current_image, img_display_current_image, img_original_current_image
    points_src_current_image = []
    if img_original_current_image is not None:
        img_display_current_image = img_original_current_image.copy()
        cv2.imshow(WINDOW_NAME_SELECT, img_display_current_image)
    print(f"Image {current_image_index + 1}: Points reset. Select 4 new points.")


def save_configurations_to_json(image_paths):
    global all_image_configurations
    if not all_image_configurations:
        print("No configurations to save.")
        return

    output_data = []
    for i, config in enumerate(all_image_configurations):
        output_data.append({
            "image_path": image_paths[i],
            "source_points": config["source_points"],
            "transformation_matrix": config["matrix"],
            "output_dimensions": config["output_dims"]
        })

    try:
        with open(CONFIG_FILE_NAME, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"All configurations saved to '{CONFIG_FILE_NAME}'")
    except Exception as e:
        print(f"Error saving configurations: {e}")


def process_images(image_files):
    global points_src_current_image, img_display_current_image, img_original_current_image
    global current_image_index, all_image_configurations

    cv2.namedWindow(WINDOW_NAME_SELECT)
    cv2.setMouseCallback(WINDOW_NAME_SELECT, select_points_callback)

    for idx, img_file in enumerate(image_files):
        current_image_index = idx
        points_src_current_image = [] # Reset for each new image

        img_original_current_image = cv2.imread(img_file)
        if img_original_current_image is None:
            print(f"Error: Could not load image '{img_file}'. Skipping.")
            # Add a placeholder or skip configuration for this image
            all_image_configurations.append(None) # Or some indicator of failure
            continue

        img_display_current_image = img_original_current_image.copy()
        cv2.imshow(WINDOW_NAME_SELECT, img_display_current_image)
        cv2.setWindowTitle(WINDOW_NAME_SELECT, f"Image {idx + 1}/{len(image_files)}: Select Points (TL,TR,BR,BL), then 'd' (Done), 'p' (Preview), 'r' (Reset)")

        print(f"\n--- Processing Image {idx + 1}/{len(image_files)}: {img_file} ---")
        print("Click 4 points: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'): # Quit entire process
                print("Quitting...")
                cv2.destroyAllWindows()
                return False # Indicate quit

            elif key == ord('r'): # Reset points for current image
                reset_points_for_current_image()

            elif key == ord('p'): # Preview correction for current image
                if len(points_src_current_image) == 4:
                    preview_or_get_corrected_image(img_original_current_image, points_src_current_image, for_preview=True)
                else:
                    print("Select 4 points first to preview.")

            elif key == ord('d'): # Done with current image
                if len(points_src_current_image) == 4:
                    print(f"Finalizing points for image {idx + 1}.")
                    _, matrix, out_dims = preview_or_get_corrected_image(
                        img_original_current_image,
                        points_src_current_image,
                        for_preview=False # Don't show, just get data
                    )
                    all_image_configurations.append({
                        "image_file_name": os.path.basename(img_file), # Store filename for reference
                        "source_points": points_src_current_image,
                        "matrix": matrix,
                        "output_dims": out_dims
                    })
                    break # Move to next image
                else:
                    user_choice = input("4 points not selected. (s)kip this image or (c)ontinue selecting? [c]: ").lower()
                    if user_choice == 's':
                        print(f"Skipping image {idx + 1}.")
                        all_image_configurations.append(None) # Mark as skipped
                        break
                    # else continue loop for current image

        if current_image_index == len(image_files) -1:
            print("\nAll images processed or skipped.")

    cv2.destroyAllWindows()
    return True # Indicate completion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually select perspective points for multiple images and save configuration.")
    parser.add_argument("--base_name", default="image_", help="Base name for image files (e.g., 'image_').")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for image files (e.g., 0 for image_0).")
    parser.add_argument("--end_index", type=int, default=4, help="Ending index for image files (e.g., 4 for image_4).")
    parser.add_argument("--extension", default=".png", help="Image file extension (e.g., '.png', '.jpg').")
    parser.add_argument("--image_dir", default=".", help="Directory containing the images.")

    args = parser.parse_args()

    image_file_paths = []
    print("Looking for images:")
    for i in range(args.start_index, args.end_index + 1):
        img_name = f"{args.base_name}{i}{args.extension}"
        img_path = os.path.join(args.image_dir, img_name)
        if os.path.exists(img_path):
            image_file_paths.append(img_path)
            print(f"  Found: {img_path}")
        else:
            print(f"  Not found: {img_path} (will be skipped if not available during processing)")

    if not image_file_paths:
        print("No images found with the specified pattern. Exiting.")
        exit()

    print("\n--- Perspective Configuration Tool for Multiple Images ---")
    print("Instructions for each image:")
    print("1. Click to select 4 points in order: TL, TR, BR, BL.")
    print("2. After 4 points:")
    print("   - Press 'd' to finalize points for THIS image and move to the next (or finish).")
    print("   - Press 'p' to preview the correction for THIS image.")
    print("   - Press 'r' to reset points for THIS image.")
    print("   - Press 'q' to quit the entire process at any time.")
    print("At the end, configurations will be saved to a JSON file.")

    if process_images(image_file_paths): # Returns True if not quit early
        save_configurations_to_json(image_file_paths) # Pass original paths for JSON

    print("Done.")