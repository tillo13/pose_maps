import os
from PIL import Image
import numpy as np
import cv2
from controlnet_aux import OpenposeDetector

def process_image(image_path, output_control_path, openpose_detector):
    """Process the input image for pose detection and save the visualized control image."""
    # Load the input image
    image = Image.open(image_path).convert("RGB")

    try:
        # Generate control image with hand and face keypoints
        control_image = openpose_detector(image, hand_and_face=True)
        control_image.save(output_control_path)
        print(f"[INFO] Control image saved to {output_control_path}")

    except Exception as e:
        print(f"[ERROR] Failed to perform pose detection: {e}")
        return

def main():
    # Define the directory containing the input images
    poses_directory = "../shared/poses/"

    # Define the directory to save generated images
    generated_images_directory = "./generated_images_clean_hands_v2/"

    # Create the generated images directory if it doesn't exist
    os.makedirs(generated_images_directory, exist_ok=True)

    try:
        # Initialize OpenPose detector for pose detection including hand and face keypoints
        openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        print("[INFO] Loaded OpenPose with hand and face detection for detailed pose estimation.")
    except Exception as e:
        print(f"[ERROR] Failed to load the OpenPose detector: {e}")
        return

    # Process each image in the poses directory
    for file_name in os.listdir(poses_directory):
        # Check if it's one of the accepted image file formats
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(poses_directory, file_name)
            output_control_path = os.path.join(generated_images_directory, f"{os.path.splitext(file_name)[0]}_control.png")
            process_image(image_path, output_control_path, openpose_detector)

if __name__ == "__main__":
    main()