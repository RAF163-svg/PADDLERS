import os
import numpy as np
import cv2
from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import shutil
from tqdm import tqdm
import pandas as pd
import math  # Import math for ceiling function
import matplotlib.pyplot as plt # Added import
from sklearn.manifold import SpectralEmbedding # Added import
from mpl_toolkits.mplot3d import Axes3D # Added for 3D plotting


geshu = 5
name = 'CFD'

image_folder_path = f'Dataset_Processed/{name}/JPEGImages/'
segmentation_folder_path = f'Dataset_Processed/{name}/SegmentationClass/'

# Create parent folder and target folders
parent_folder = f'{name}_clusters'  # Add parent folder
os.makedirs(parent_folder, exist_ok=True)

# Load and process images
def load_and_process_images(folder_path):
    image_paths = []
    features = []
    
    # Get all image files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            image_paths.append(file_path)
            
            # Read image
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # Resize image
            img = cv2.resize(img, (64, 64))
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Extract features (using pixel values as features)
            feature = gray.flatten()
            features.append(feature)
    
    return np.array(features), image_paths

# Main function
def main(num_clusters):
    # Modify output_folders to match the number of clusters passed in
    global output_folders
    output_folders = [f'cluster_{i+1}' for i in range(num_clusters)]
    
    # Recreate folder structure
    for folder in output_folders:
        full_path = os.path.join(parent_folder, folder)
        os.makedirs(os.path.join(full_path, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'SegmentationClass'), exist_ok=True)
    
    # Ensure folders exist
    if not os.path.exists(image_folder_path):
        print(f"Image folder {image_folder_path} does not exist!")
        return
        
    print("Loading and processing images...")
    features, image_paths = load_and_process_images(image_folder_path)
    
    if len(features) == 0:
        print("No valid images found!")
        return
        
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform GMM clustering
    print("Performing GMM clustering...")
    n_clusters = num_clusters
    gmm = GaussianMixture(n_components=n_clusters, 
                           random_state=42,
                           max_iter=100,
                           covariance_type='full')
    gmm_labels = gmm.fit_predict(features_scaled)
    
    # Perform spectral clustering
    print("Performing spectral clustering...")
    n_clusters = num_clusters
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                   assign_labels='discretize',
                                   random_state=42,
                                   affinity='nearest_neighbors')
    labels = clustering.fit_predict(features_scaled)
    
    # Assign images to cluster folders
    print("Assigning images to cluster folders...")
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        dest_cluster_folder = os.path.join(parent_folder, output_folders[label])
        
        # Copy original image
        dest_image_path = os.path.join(dest_cluster_folder, 'JPEGImages', filename)
        shutil.copy(path, dest_image_path)
        
        # Find and copy corresponding segmentation mask (assumed to be in .png format)
        segmentation_filename = base_name + '.png' 
        src_segmentation_path = os.path.join(segmentation_folder_path, segmentation_filename)
        dest_segmentation_path = os.path.join(dest_cluster_folder, 'SegmentationClass', segmentation_filename)
        
        if os.path.exists(src_segmentation_path):
            shutil.copy(src_segmentation_path, dest_segmentation_path)
        else:
            print(f"Warning: Corresponding segmentation mask file not found {src_segmentation_path}")

    print(f"Complete! Images and segmentation masks have been assigned to {n_clusters} cluster folders.")

    # Count and output the number of files in each folder
    print("\nFile count statistics:")
    
    # Create data for Excel
    excel_data = {
        'Cluster': [],
        'Sample Count': []
    }
    
    total_images = 0
    total_masks = 0
    
    for i, folder in enumerate(output_folders):
        full_path = os.path.join(parent_folder, folder)
        jpeg_folder = os.path.join(full_path, 'JPEGImages')
        seg_folder = os.path.join(full_path, 'SegmentationClass')
        
        jpeg_count = len([name for name in os.listdir(jpeg_folder) if os.path.isfile(os.path.join(jpeg_folder, name))])
        seg_count = len([name for name in os.listdir(seg_folder) if os.path.isfile(os.path.join(seg_folder, name))])
        
        total_images += jpeg_count
        total_masks += seg_count
        
        # Add data to Excel data structure
        excel_data['Cluster'].append(folder)
        excel_data['Sample Count'].append(jpeg_count)
        
        print(f"  {folder}:")
        print(f"    JPEGImages: {jpeg_count} files")
        print(f"    SegmentationClass: {seg_count} files")
    
    # Add total to Excel data
    excel_data['Cluster'].append('Total')
    excel_data['Sample Count'].append(total_images)
    
    # Create DataFrame and write to Excel
    df = pd.DataFrame(excel_data)
    excel_file_path = os.path.join(parent_folder, 'clusters_statistics.xlsx')
    df.to_excel(excel_file_path, index=False)
    
    # Print total information
    print("\nTotal:")
    print(f"  Total images: {total_images} files")
    print(f"  Total masks: {total_masks} files")
    
    print(f"\nStatistics have been saved to Excel: {excel_file_path}")

    # Visualize the latent space
    print("\nVisualizing latent space...")
    n_components_visual = 3 # For 3D visualization
    
    # Perform spectral embedding.
    # Use parameters consistent with the SpectralClustering instance if possible.
    # SpectralClustering uses n_components = n_clusters internally for its embedding.
    # For visualization, we explicitly use n_components_visual (e.g., 3).
    embedding_model = SpectralEmbedding(n_components=n_components_visual,
                                     affinity=clustering.affinity, # Use affinity from the clustering model
                                     n_neighbors=clustering.n_neighbors, # Use n_neighbors from the clustering model
                                     random_state=42) # Consistent random state
    
    features_embedded = embedding_model.fit_transform(features_scaled)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d') # Create 3D subplot
    scatter = ax.scatter(features_embedded[:, 0], features_embedded[:, 1], features_embedded[:, 2], 
                         c=labels, cmap='viridis', s=50, alpha=0.7)
    
    # Create a legend for clusters
    # Ensure num_clusters is the actual number of unique labels if some clusters are empty,
    # but for this context, num_clusters (which is n_clusters for SpectralClustering) is appropriate.
    unique_labels_for_legend = np.unique(labels)
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6, num=len(unique_labels_for_legend))
    legend_labels = [f'Cluster {i}' for i in unique_labels_for_legend] # Adjust if labels are not 0-indexed or sequential

    # If labels are 0 to k-1, then this is simpler:
    if len(unique_labels_for_legend) == num_clusters and np.all(unique_labels_for_legend == np.arange(num_clusters)) :
        legend_labels = [f'Cluster {i+1}' for i in range(num_clusters)]
    else: # Fallback for more complex labelings, though SpectralClustering usually gives 0 to k-1.
        legend_labels = [f'Cluster Label {l}' for l in unique_labels_for_legend]


    ax.legend(handles, legend_labels, title="Clusters") # Use ax.legend for 3D plot
    
    ax.set_title(f'Bayesian Clustering Latent Space ({n_components_visual}D) for {name}') # Use ax.set_title
    ax.set_xlabel('Latent Dimension 1') # Use ax.set_xlabel
    ax.set_ylabel('Latent Dimension 2') # Use ax.set_ylabel
    ax.set_zlabel('Latent Dimension 3') # Add Z axis label
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(parent_folder, f'{name}_latent_space_visualization.png')
    plt.savefig(plot_filename)
    print(f"Latent space visualization saved to {plot_filename}")
    # plt.show() # Uncomment to display the plot interactively


    for folder in output_folders:
        full_path = os.path.join(parent_folder, folder)
        xmlfilepath = os.path.join(full_path, 'JPEGImages')
        seg_folder = os.path.join(full_path, 'SegmentationClass')
        txtsavepath = os.path.join(full_path, 'ImageSets', 'Segmentation')
        # Split training and validation sets in ratio 6:2, using all images
        train_ratio = 0.75 # 75% for training set (approximately 6/(6+2) in 6:2)
        
        total_xml = os.listdir(xmlfilepath)

        # Ensure directory exists
        os.makedirs(txtsavepath, exist_ok=True)

        num = len(total_xml)
        train_count = math.ceil(num * train_ratio) # Training set count

        # Open files for writing
        ftrainval = open(txtsavepath + '/train.txt', 'w')
        val = open(txtsavepath + '/val.txt', 'w')

        # Assign directly in sequence
        for i in range(num):
            file_basename_for_set = total_xml[i][:-4] + '\n'
            if i < train_count:
                ftrainval.write(file_basename_for_set)
            else:
                val.write(file_basename_for_set)

        ftrainval.close()
        val.close()

        print(f"Total images: {num}")
        print(f"Training images: {train_count}")
        print(f"Validation images: {num - train_count}")
        print(f"train.txt and val.txt saved to {txtsavepath}")



if __name__ == "__main__":
    main(geshu)