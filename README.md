# PeytonCunningham-FFT-Mountains
Peyton Cunningham 2270-002 Final Project

# FFT Image Processing Project
## Overview
The Fast Fourier Transform (FFT) is a powerful algorithm that is used to analyze the frequency components of signals, which has significant applications in image processing. For this project, I apply FFT to process mountain images and group them based on their visual features. The main idea is to convert images into the frequency domain using FFT, extract key frequency features, and then group similar images based on these features.

By processing images to extract frequency information, patterns that are not immediately obvious in the spatial domain are revealed. Using FFT, the features of the images are enhanced and clustering is performed to group similar images, which is valuable for organizing and understanding large image datasets.

## Goals
Implement Image Preprocessing: Images are converted to grayscale, and resized, and histogram equalization is applied to standardize the input.

**Apply FFT to Images**: Images are transformed into the frequency domain to analyze their frequency components.

**Feature Extraction**: Significant features are extracted from the FFT magnitude spectrum for each image.

**Image Clustering**: Clustering algorithms (K-Means) are used to group images based on their frequency features.

The project involves coding and implementing image preprocessing, FFT application, feature extraction, and clustering algorithms. Python libraries are used, such as numpy, opencv, scipy, and sklearn.

## Running the Project
To run the project, follow these steps:

**Install Dependencies**:
Ensure that the required Python libraries are installed. They can be installed using **pip**:

```bash
# Command to install dependencies
# Copy code
pip install numpy opencv-python-headless scipy scikit-learn matplotlib
```
**Prepare the Dataset**:
Place mountain images in a folder. Note: the images should be in JPEG or PNG format.

**Update the Image Folder Path**:
Modify the image_folder variable in the main function of fft_image_processing.py to point to the folder containing the images.

**Execute the Script**:
Run the Python script from the **command line**:

```bash
# Copy code
python fft_image_processing.py
```
**View Results**:
The script will display clusters of images using **matplotlib.** Each cluster represents a group of images with similar frequency features.

## What the Project Does
In this project, I process images using FFT to analyze their frequency components and then group similar images based on extracted features. 

## What the Script Does:

**Image Preprocessing**: I resize the images, convert them to grayscale, and enhance their contrast.

**FFT Transformation**: I transform each image into the frequency domain using FFT.

**Feature Extraction**: I analyze the magnitude spectrum of the FFT to extract key features.

**Clustering**: I cluster the images based on their frequency features using the K-Means algorithm.

**Visualization**: I display the clustered images to show how they have been grouped.

## Results and Demonstration:

Since the project is executable, results can be seen by directly running the script. The visual output will show images grouped into clusters, illustrating how similar images are organized based on their frequency characteristics.

## Notes on the Project

I chose to work with image processing for this project due to my prior experience with it in the 1300 course. Additionally, I incorporated the K-Means clustering algorithm, drawing from my knowledge gained in the 2800 course. I opted for the **Fast Fourier Transform (FFT)** because of its ability to reveal characteristics and similarities in images that might not be immediately noticeable to myself or others.

## Future Applications
I envision this project as having practical applications in identifying potential new crags and boulders. In climbing, a crag refers to an area with large rock walls used for various forms of roped climbing, and boulders, are freestanding rocks climbed without ropes. In the specialized field of outdoor climbing development, developers often use social media to search for images of rock faces and boulders that could be suitable for climbing. By employing an algorithm like FFT, these images can be pre-sorted to enhance the efficiency of this search, potentially aiding climbers in discovering new climbing spots more effectively.
