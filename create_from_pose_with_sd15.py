import torch
import os
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import cv2
from controlnet_aux import OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

def process_image(image_path, output_control_path, output_pipe_path, prompt="a person"):
    # Load the model and image
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    image = Image.open(image_path).convert("RGB")

    # Generate control image with hand and face keypoints
    control_image = processor(image, hand_and_face=True)
    control_image.save(output_control_path)
    
    # Load the control model
    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    
    # Load the pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Disable the safety checker
    pipe.safety_checker = None

    # Generate the image
    generator = torch.manual_seed(0)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    generated_image.save(output_pipe_path)

def main():
    # Define the directory containing the input images
    poses_directory = "../shared/poses/"

    # Define the directory to save generated images
    generated_images_directory = "./generated_images_create_from_pose_sd15/"

    # Create the generated images directory if it doesn't exist
    os.makedirs(generated_images_directory, exist_ok=True)

    # Process each image in the poses directory
    for file_name in os.listdir(poses_directory):
        # Check if it's one of the accepted image file formats
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(poses_directory, file_name)
            output_control_path = os.path.join(generated_images_directory, f"{os.path.splitext(file_name)[0]}_control.png")
            output_pipe_path = os.path.join(generated_images_directory, f"{os.path.splitext(file_name)[0]}_posed.png")
            process_image(image_path, output_control_path, output_pipe_path)

if __name__ == "__main__":
    main()