import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Function to load and preprocess images from a folder
def load_and_preprocess_images(images_folder):
    images = []  # List to store preprocessed images
    # Iterate through files in the specified folder
    for filename in os.listdir(images_folder):
        # Check if the file is an image (JPEG or PNG)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Build the full path to the image file
            image_path = os.path.join(images_folder, filename)
            # Load the image in grayscale mode
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Unable to load image {filename}.")
                continue
            # Resize the image to a uniform size (256x256 pixels)
            img = cv2.resize(img, (256, 256))
            # Apply histogram equalization to improve contrast
            img = cv2.equalizeHist(img)
            # Append the preprocessed image to the list
            images.append(img)
    return images  # Return the list of preprocessed images

# Function to apply Fast Fourier Transform (FFT) and return the magnitude spectrum
def apply_fft(image):
    # Compute the FFT of the image
    f_transform = fft2(image)
    # Shift the zero frequency component to the center of the spectrum
    f_transform_shifted = fftshift(f_transform)
    # Compute the magnitude spectrum of the FFT
    magnitude_spectrum = np.abs(f_transform_shifted)
    return magnitude_spectrum  # Return the magnitude spectrum

# Function to extract features from the magnitude spectrum
def extract_features(magnitude_spectrum):
    # Apply a logarithmic transformation to enhance feature visibility
    features = np.log(1 + magnitude_spectrum)
    # Flatten the 2D magnitude spectrum to a 1D array
    features_flat = features.flatten()
    return features_flat  # Return the flattened feature array

# Function to preprocess images, apply FFT, and extract features
def preprocess_and_extract_features(images_folder):
    # Load and preprocess images from the folder
    images = load_and_preprocess_images(images_folder)
    features_list = []  # List to store feature vectors for each image
    # Iterate over each preprocessed image
    for img in images:
        # Apply FFT to the image and get the magnitude spectrum
        magnitude_spectrum = apply_fft(img)
        # Extract features from the magnitude spectrum
        features = extract_features(magnitude_spectrum)
        # Append the features to the list
        features_list.append(features)
    return np.array(features_list)  # Return the array of feature vectors

# Function to cluster images based on FFT features
def cluster_images(features, n_clusters=5):
    # Reduce dimensionality of the feature vectors using PCA
    pca = PCA(n_components=14)
    features_pca = pca.fit_transform(features)
    # Perform K-Means clustering on the PCA-reduced features
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # Explicitly set n_init
    labels = kmeans.fit_predict(features_pca)  # Predict cluster labels for each image
    return labels  # Return the cluster labels

# Function to display images grouped by cluster
def display_clusters(images_folder, labels):
    # Load and preprocess images from the folder
    images = load_and_preprocess_images(images_folder)
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    
    # Iterate over each unique label (cluster)
    for label in unique_labels:
        cluster_images_list = [img for img, l in zip(images, labels) if l == label]
        num_images = len(cluster_images_list)
        
        fig, axes = plt.subplots(1, num_images, figsize=(12, 6))  # Create subplots
        plt.suptitle(f"Cluster {label}")  # Set the title of the figure
        
        # Handle the case where there's only one image in the cluster
        if num_images == 1:
            axes = [axes]  # Make axes iterable
        
        # Iterate over each image in the cluster
        for i, img in enumerate(cluster_images_list):
            axes[i].imshow(img, cmap='gray')  # Display the image in grayscale
            axes[i].axis('off')  # Hide the axis
        
        # Save the plot as a PNG file for this cluster
        plt.savefig(f'cluster_{label}.png')  # Save plot as a PNG file with a unique name
        plt.close()  # Close the plot to free resources

# Main function to execute the image processing and clustering
def main():
    images_folder = '/home/jovyan/PeytonCunningham-FFT-Mountains/images_folder'  # Path to the folder containing images
    # Preprocess images, apply FFT, and extract features
    features = preprocess_and_extract_features(images_folder)
    # Cluster images based on their FFT features
    labels = cluster_images(features, n_clusters=5)  # Adjust number of clusters as needed
    # Display the clustered images
    display_clusters(images_folder, labels)

# Execute the main function if the script is run directly
if __name__ == '__main__':
    main()
