import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil

def calculate_arbitrary_value(histogram, non_black_percentage):
    # Extract values from the histogram dictionary and convert them to numeric types
    histogram_values = [float(value) for value in histogram.values()]
    # Example formula: sum of histogram values multiplied by non_black_percentage
    result = sum(histogram_values) * non_black_percentage
    return result

def generate_dataset_from_json(input_path):
    print("\n")
    print("Genrating dataset from JSON files...")
    # Path to the JSON files
    centroids_json_file_path = os.path.join(input_path, 'centroids.json')
    histograms_json_file_path = os.path.join(input_path, 'color_histograms.json')
    non_black_percentage_json_file_path = os.path.join(input_path, 'non_black_percentage.json')
    depths_json_file_path = os.path.join(input_path, 'depths.json')

    # Load data from JSON files
    with open(centroids_json_file_path, 'r') as f:
        centroids_data = json.load(f)
    with open(histograms_json_file_path, 'r') as f:
        histograms_data = json.load(f)
    with open(non_black_percentage_json_file_path, 'r') as f:
        non_black_percentage_data = json.load(f)
    with open(depths_json_file_path, 'r') as f:
        depths_data = json.load(f)

    # Find common keys
    common_keys = set(centroids_data.keys()) & set(histograms_data.keys()) & set(non_black_percentage_data.keys()) & set(depths_data.keys())

    # Combine data
    combined_data = {}
    for key in common_keys:
        histogram = histograms_data[key]
        non_black_percentage = non_black_percentage_data[key]
        arbitrary_value = calculate_arbitrary_value(histogram, non_black_percentage)
        
        combined_data[key] = {
            'centroid': centroids_data[key],
            'histogram': histogram,
            'non_black_percentage': non_black_percentage,
            'depth': depths_data[key],
            'arbitrary_value': arbitrary_value
        }

    # Sort the combined data
    sorted_keys = sorted(combined_data.keys(), key=lambda x: (int(re.search(r'Image_(\d+)_Mask', x).group(1)), int(re.search(r'Mask_(\d+)', x).group(1))))
    sorted_combined_data = {key: combined_data[key] for key in sorted_keys}
    
    print("\n")
    print("Dataset generated.")
    print("\n")

    return sorted_combined_data

    
def draw_arbitrary_value(combined_data, input_path, output_path):
    print("\n")
    print("Drawing arbitrary value on images...")
    # Group data by image number
    grouped_data = {}
    for key in combined_data:
        image_num = int(re.search(r'Image_(\d+)_Mask', key).group(1))
        if image_num not in grouped_data:
            grouped_data[image_num] = []
        grouped_data[image_num].append(combined_data[key])
    
    # Process each image
    for image_num, data_list in grouped_data.items():
        # Load the corresponding "Combined_Masked_Pixels_[NUM]" image
        image_filename = f"Combined_Masked_Pixels_{image_num}.jpg"
        image_path = os.path.join(input_path, image_filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image {image_filename} not found.")
            continue
        
        # Draw all data points on the image
        for data in data_list:
            centroid = data['centroid']
            arbitrary_value = data['arbitrary_value']
            depth = data['depth']
            
            # Draw the centroid
            centroid_x = int(centroid['centroid_x'])
            centroid_y = int(centroid['centroid_y'])
            cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            
            # Draw the arbitrary value
            cv2.putText(image, f"Value: {arbitrary_value:.2f}", (centroid_x + 10, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw the depth value
            depth_value = depth['depth']
            if depth_value is not None:
                cv2.putText(image, f"Depth: {depth_value:.2f}mm", (centroid_x + 10, centroid_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(image, "Depth: None", (centroid_x + 10, centroid_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save the modified image
        output_filename = f"Annotated_Combined_Masked_Pixels_{image_num}.jpg"
        output_image_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_image_path, image)
        print(f"Saved annotated image: {output_filename}")
    print("\n")
    print("Arbitrary calculation done.")
    print("\n")
    
def generate_final_result(input_path, image_directory, combined_data):
    # Create the "Final_Results" folder
    final_results_path = os.path.join(input_path, "Final_Results")
    os.makedirs(final_results_path, exist_ok=True)

    # Find all Annotated_Combined_Masked_Pixels_[NUM] and Result_[NUM] images
    annotated_images = [f for f in os.listdir(input_path) if f.startswith("Annotated_Combined_Masked_Pixels_")]
    result_images = [f for f in os.listdir(input_path) if f.startswith("Result_")]

    # Process each image group
    for annotated_image in annotated_images:
        # Extract the image number
        image_num = int(re.search(r'Annotated_Combined_Masked_Pixels_(\d+)', annotated_image).group(1))
        result_image = f"Result_{image_num}.jpg"
        combined_image = f"Combined_Masked_Pixels_{image_num}.jpg"
        scene_image = f"scene_01_{image_num:04d}.png"

        # Create the "Image [NUM]" folder
        image_folder = os.path.join(final_results_path, f"Scene_01_{image_num:04d}")
        os.makedirs(image_folder, exist_ok=True)

        # Create the "Raw" subfolder
        raw_folder = os.path.join(image_folder, "Raw")
        os.makedirs(raw_folder, exist_ok=True)

        # Copy the relevant images to the "Raw" subfolder
        for filename in [annotated_image, result_image, combined_image, scene_image]:
            src_path = os.path.join(input_path, filename) if filename != scene_image else os.path.join(image_directory, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, raw_folder)
            else:
                print(f"File {filename} not found, skipping.")

        # Write the relevant data to a JSON file
        relevant_data = {key: data for key, data in combined_data.items() if int(re.search(r'Image_(\d+)_Mask', key).group(1)) == image_num}
        json_filename = f"Data_{image_num}.json"
        json_path = os.path.join(raw_folder, json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(relevant_data, json_file, indent=4)
        print(f"Saved JSON data: {json_filename}")

        # Load the images
        annotated_img = cv2.imread(os.path.join(input_path, annotated_image))
        scene_img = cv2.imread(os.path.join(image_directory, scene_image))

        if annotated_img is None or scene_img is None:
            print(f"Error loading images for Image_{image_num}")
            continue

        # Resize images to a reasonable size for comparison
        height = max(annotated_img.shape[0], scene_img.shape[0])
        width = annotated_img.shape[1] + scene_img.shape[1]
        comparison_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the images side by side
        comparison_img[:annotated_img.shape[0], :annotated_img.shape[1]] = annotated_img
        comparison_img[:scene_img.shape[0], annotated_img.shape[1]:] = scene_img

        # Save the side-by-side comparison image
        comparison_image_path = os.path.join(image_folder, f"Comparison_{image_num}.png")
        cv2.imwrite(comparison_image_path, comparison_img)
        print(f"Saved comparison image: {comparison_image_path}")

def main(input_path='Project/Results/Pipeline/RUN_4', results_path='Project/Results/Pipeline/RUN_4', image_directory='Project/Examples_ZED/RGB_left'):
    print("\n") 
    print("=================================") 
    print("===== Interpretation Start ======")
    print("=================================") 
    
    data = generate_dataset_from_json(input_path)
    draw_arbitrary_value(data, input_path, results_path)
    generate_final_result(input_path, image_directory, data)
                
    print("\n") 
    print("=================================") 
    print("====== Interpretation End =======")
    print("=================================")

if __name__ == "__main__":
    main()