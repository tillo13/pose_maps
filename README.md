# Project README

## Project Overview

This project involves using OpenPose (through ControlNet) to detect human poses in images and to manipulate these poses using Stable Diffusion 1.5. Various scripts are provided in this repository to analyze and process images for different scenarios, ranging from detailed pose maps (including hands and faces) to pose maps without hands.

## Directory Structure
```
.
├── generated_images_clean_hands/
├── clean_hands.py
├── clean_hands_v2.py
├── create_from_pose_with_sd15.py
├── gather_pythons.py
└── no_hands.py
```

## Requirements

- **Python**: 3.x

### Required Python packages (install via pip):
- `opencv-python`
- `numpy`
- `Pillow`
- `controlnet_aux`
- `torch`
- `diffusers`
- `huggingface_hub`

## Python Scripts

### `clean_hands.py`

This script processes images to detect poses, including hands and faces, using OpenPose through ControlNet.

#### Usage

1. Configure the directory containing the input images `poses_directory` and the output directory `generated_images_directory` in the script.
2. Run the script using: 
    ```sh
    python clean_hands.py
    ```

### `clean_hands_v2.py`

This script is a slightly modified version of `clean_hands.py` with better exception handling.

#### Usage

1. Configure the directory containing the input images `poses_directory` and the output directory `generated_images_directory` in the script.
2. Run the script using: 
    ```sh
    python clean_hands_v2.py
    ```

### `create_from_pose_with_sd15.py`

This script processes images and generates new images using the Stable Diffusion model based on the detected poses.

#### Usage

1. Configure the directory containing the input images `poses_directory` and the output directory `generated_images_directory` in the script.
2. Run the script using: 
    ```sh
    python create_from_pose_with_sd15.py
    ```

### `gather_pythons.py`

This script gathers all Python files within the specified root directory and its subdirectories, excluding specified directories.

#### Usage

1. Configure the `root_dir` and `DIRECTORIES_TO_EXCLUDE` in the script. The default value for `root_dir` is the current directory.
2. Run the script using: 
    ```sh
    python gather_pythons.py
    ```

### `no_hands.py`

This script processes images to detect poses excluding hands using OpenPose through ControlNet.

#### Usage

1. Configure the directory containing the input images `poses_directory` and the output directory `generated_images_directory` in the script.
2. Run the script using: 
    ```sh
    python no_hands.py
    ```

## Example

For each script, if the `poses_directory` is set to `../shared/poses/`, the images from this directory will be processed and the results will be saved in the respective output directories. Ensure the input directory contains image files in supported formats (e.g., `.png`, `.jpg`, `.jpeg`, `.webp`).

Sample command to run any of the scripts:
```sh
python clean_hands.py
```

## Contribution

1. Fork the repository.
2. Create your feature branch:
    ```sh
    git checkout -b feature/awesome-feature
    ```
3. Commit your changes:
    ```sh
    git commit -m 'Add awesome feature'
    ```
4. Push to the branch:
    ```sh
    git push origin feature/awesome-feature
    ```
5. Open a Pull Request.

## License

This project is licensed under the MIT License – see the LICENSE file for details.