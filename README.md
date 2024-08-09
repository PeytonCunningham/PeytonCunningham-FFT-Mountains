# PeytonCunningham-FFT-Mountains
Peyton Cunningham 2270-002 Final Project
FFT Image Processing Project
Overview
The Fast Fourier Transform (FFT) is a powerful algorithm used to analyze the frequency components of signals, and it has significant applications in image processing. In this project, we use FFT to process mountain images and group them based on their visual features. The core idea is to convert images into the frequency domain using FFT, extract key frequency features, and then group similar images based on these features.

Images are processed to extract frequency information which can reveal patterns not immediately obvious in the spatial domain. By using FFT, we can enhance the features of the images and perform clustering to group similar images, which can be valuable in organizing and understanding large image datasets.

Goals
Implement Image Preprocessing: Convert images to grayscale, resize them, and apply histogram equalization to standardize the input.
Apply FFT to Images: Transform images into the frequency domain to analyze their frequency components.
Feature Extraction: Extract significant features from the FFT magnitude spectrum for each image.
Image Clustering: Use clustering algorithms (like K-Means) to group images based on their frequency features.
The project involves coding and implementing image preprocessing, FFT application, feature extraction, and clustering algorithms. You will work with Python libraries such as numpy, opencv, scipy, and sklearn.

Running the Project
To run the project, follow these steps:

Install Dependencies:
Ensure you have the required Python libraries installed. You can install them using pip:

bash
Copy code
pip install numpy opencv-python-headless scipy scikit-learn matplotlib
Prepare the Dataset:
Place your mountain images in a folder. The images should be in JPEG or PNG format.

Update the Image Folder Path:
Modify the image_folder variable in the main function of fft_image_processing.py to point to the folder containing your images.

Execute the Script:
Run the Python script from the command line:

bash
Copy code
python fft_image_processing.py
View Results:
The script will display clusters of images using matplotlib. Each cluster represents a group of images with similar frequency features.

What the Project Does
The project processes images using FFT to analyze their frequency components and then groups similar images based on extracted features. Here's a summary of what the script does:

Image Preprocessing: Images are resized, converted to grayscale, and have their contrast enhanced.
FFT Transformation: Each image is transformed into the frequency domain using FFT.
Feature Extraction: The magnitude spectrum of the FFT is analyzed to extract key features.
Clustering: Images are clustered based on their frequency features using the K-Means algorithm.
Visualization: The clustered images are displayed to show how they have been grouped.
Results and Demonstration
Since the project is executable, you can see the results directly by running the script. The visual output will show images grouped into clusters, illustrating how similar images are organized based on their frequency characteristics.

If you prefer a visual demonstration, you can refer to the following links:

YouTube Video Demonstration (Replace with your actual video link)
Screenshots and Documentation (Replace with your actual documentation link)
Additional Notes
Ensure Images are Preprocessed: The quality of the FFT and clustering depends significantly on how well the images are preprocessed.
Feature Extraction: The choice of features can affect clustering results. Experiment with different feature extraction methods if needed.
Clustering Parameters: Adjust the number of clusters (n_clusters) in the cluster_images function based on your dataset and requirements.
Feel free to experiment with different images and parameters to see how they affect the clustering results. The FFT-based image processing can be a powerful tool for organizing and analyzing image datasets.

