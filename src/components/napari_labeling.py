import sys
import numpy as np
from pathlib import Path
import napari
from deeplabcut.utils import auxiliaryfunctions
import imageio.v3 as iio

def main():
    if len(sys.argv) != 3:
        print("Usage: python napari_labeling.py <config_path> <image_folder>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    image_folder = Path(sys.argv[2])

    config = auxiliaryfunctions.read_config(str(config_path))  # Use DLC's read_config

    # Load all images in the folder
    image_files = sorted([p for p in image_folder.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    if not image_files:
        print("No image files found in:", image_folder)
        sys.exit(1)

    # Stack images into a 3D numpy array
    images = np.array([iio.imread(str(img)) for img in image_files])

    # Ensure all images are the same shape
    if len(set(img.shape for img in images)) != 1:
        print("Images are not of the same shape. Please check the images.")
        sys.exit(1)

    # Load keypoint skeleton structure if available
    skeleton = config.get("skeleton", [])
    bodyparts = config.get("bodyparts", [])

    # Launch Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(images, name="frames")

    if skeleton and bodyparts:
        print("Skeleton found. You can manually add keypoints and link them based on:")
        print(f"Bodyparts: {bodyparts}")
        print(f"Skeleton: {skeleton}")
    else:
        print("No skeleton or bodypart info found in config.")
        
    #viewer.add_labels

    napari.run()

if __name__ == "__main__":
    main()
