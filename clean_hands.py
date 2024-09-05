import cv2
import os
import numpy as np
from controlnet_aux import OpenposeDetector
from PIL import Image

def process_image(image_path, output_path):
    # Initialize OpenPose detector for pose detection including hand and face keypoints
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    print(f"[INFO] Loaded OpenPose with hand and face detection for {os.path.basename(image_path)}.")

    # Load the input image
    image = Image.open(image_path).convert("RGB")

    # Perform pose detection, including hands and faces. This should directly provide hand and face keypoints
    control_image = openpose_detector(image, hand_and_face=True)

    # Convert the result to a format suitable for OpenCV if necessary
    if isinstance(control_image, Image.Image):
        control_image = np.array(control_image)

    if control_image.ndim == 3 and control_image.shape[2] == 3:
        output_image = cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR)
    else:
        output_image = control_image

    # Save the output image with detected keypoints
    cv2.imwrite(output_path, output_image)
    print(f"[INFO] Output saved to {output_path}")

def main():
    # Define the directory containing the input images
    poses_directory = "../shared/poses/"

    # Define the directory to save generated images
    generated_images_directory = "./generated_images_clean_hands/"

    # Create the generated images directory if it doesn't exist
    os.makedirs(generated_images_directory, exist_ok=True)

    # Process each image in the poses directory
    for file_name in os.listdir(poses_directory):
        # Check if it's one of the accepted image file formats
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(poses_directory, file_name)
            output_path = os.path.join(generated_images_directory, f"{os.path.splitext(file_name)[0]}_detailed_pose_map.png")
            process_image(image_path, output_path)

if __name__ == "__main__":
    main()