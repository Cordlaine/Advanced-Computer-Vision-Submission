import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re

def load_depth_data(input_path, image_number):
    input_path = os.path.normpath(input_path)
    depth_file = os.path.join(input_path, f'scene_01_{image_number:04d}.npy')
    if os.path.exists(depth_file):
        return depth_file
    else:
        raise FileNotFoundError(f"Depth file for image number {image_number} not found.")

def visualize_depth_data(depth_data, title):
    # Mask to ignore -inf and inf values
    valid_mask = np.isfinite(depth_data)
    valid_data = depth_data[valid_mask]
    
    # Calculate Q1 and Q3
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1
    
    # Define the range for valid data
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data to exclude outliers
    filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
    
    # Get the min and max values of the filtered data
    vmin = np.min(filtered_data)
    vmax = np.max(filtered_data)
    
    print(f"Min depth value: {vmin}, Max depth value: {vmax}")
    
    # Visualize the depth data using Matplotlib with dynamic color scale
    plt.imshow(depth_data, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(label='Depth')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    
def get_centroid_coordinates(image_number, input_path):
    # Path to the JSON file
    json_file_path = os.path.join(input_path, 'centroids.json')
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        centroids = json.load(f)
    
    # Extract coordinates for the specified image number
    coordinates = []
    for key, value in centroids.items():
        if key.startswith(f"Image_{image_number}_"):
            coordinates.append((key, value['centroid_x'], value['centroid_y']))
    
    return coordinates

def retrieve_depth_info(depth_data, x, y):
    # Retrieve the depth value at a specific (x, y) coordinate
    if 0 <= x < depth_data.shape[1] and 0 <= y < depth_data.shape[0]:
        return depth_data[y, x]
    else:
        raise ValueError("Coordinates out of bounds")

def main(input_path='Project/Examples_ZED/depth', results_path='Project/Results/Test', coordinates_path='Project/Results/Test', visualize=True):    
    print("\n") 
    print("=================================") 
    print("===== Depth Retrieval Start =====")
    print("=================================") 
    
    # Extract image numbers from filenames in the directory
    image_numbers = []
    pattern = re.compile(r'^scene_\d+_(\d+)\.npy$')
    
    for filename in os.listdir(input_path):
        match = pattern.match(filename)
        if match:
            image_number = int(match.group(1))
            image_numbers.append((image_number, filename))
    
    # Initialize an empty dictionary to store depth information
    depth_info = {}
            
    for image_number, filename in image_numbers:
        print("\n")    
        print(f"Results for {filename}:\n")
        
        # Load the depth data file corresponding to the image number
        depth_file = load_depth_data(input_path, image_number)
        
        # Load the depth data
        depth_data = np.load(depth_file)
        
        if visualize:
            # Visualize the depth data
            visualize_depth_data(depth_data, title=depth_file)
        
        # Get centroid coordinates for the specified image number
        coordinates = get_centroid_coordinates(image_number, coordinates_path)
        
        print(f"Centroid coordinates for Image_{image_number}: {coordinates}")
        
        # Retrieve and print depth information for each centroid coordinate
        for key, x, y in coordinates:
            depth_value = retrieve_depth_info(depth_data, x, y)
            if np.isnan(depth_value):
                print(f"Value at ({x}, {y}) is NaN. Checking surrounding values:")
                for i in range(max(0, y-1), min(depth_data.shape[0], y+2)):
                    for j in range(max(0, x-1), min(depth_data.shape[1], x+2)):
                        print(f"Value at ({j}, {i}): {depth_data[i, j]}")
                print("\n")
                depth_info[key] = {
                    "centroid_x": x,
                    "centroid_y": y,
                    "depth": None,  # Use None to represent NaN in JSON
                    #"depth_is_nan": True
                }
            else:
                print(f"Depth value at centroid (x={x}, y={y}): {depth_value}")
                depth_info[key] = {
                    "centroid_x": x,
                    "centroid_y": y,
                    "depth": float(depth_value),  # Convert to native Python float
                    #"depth_is_nan": False
                }
        
    # Write the depth information to a new JSON file called depths.json
    depths_json_path = os.path.join(results_path, 'depths.json')
    with open(depths_json_path, 'w') as f:
        json.dump(depth_info, f, indent=4)

    print("\n") 
    print("=================================") 
    print("====== Depth Retrieval End ======")
    print("=================================") 
    
if __name__ == "__main__":
    main(visualize=False)