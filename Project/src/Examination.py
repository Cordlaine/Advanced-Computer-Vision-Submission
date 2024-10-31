import os
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

def image_selector(input_path):
    pattern = re.compile(r'^Image_(\d+)_Mask_(\d+)\.jpg$')
    matching_files = []
    for filename in os.listdir(input_path):
        match = pattern.match(filename)
        if match:
            image_number = int(match.group(1))
            matching_files.append((image_number, os.path.join(input_path, filename)))
    return matching_files

def generate_color_histogram(image_path, ignore_black=True, visualize=True):
    image = cv2.imread(image_path)
    image_path = os.path.basename(image_path)
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    if visualize:  
        plt.figure()
        plt.title(f'Color Histogram for {image_path}')
        plt.xlabel('Bins')
        plt.ylabel('# of Pixels')
    
    print("\n")    
    print(f"Color channel results for {image_path}:\n")
    
    mean_values = {}
    for (channel, color) in zip(channels, colors):
        if ignore_black:
            mask = channel > 0
            channel = channel[mask]
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        if visualize:  
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        
        mean_val = np.mean(channel)
        mean_values[color.upper()] = mean_val
        print(f"Mean {color.upper()} value: {mean_val:.2f}")
    
    if visualize:    
        plt.show()
    
    return mean_values

def calculate_non_black_percentage(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_pixels = gray_image.size
    non_black_pixels = np.count_nonzero(gray_image)
    non_black_percentage = (non_black_pixels / total_pixels) * 100
      
    print(f"Non black percent: {non_black_percentage}")
    
    return non_black_percentage

def main(input_path='Project/Results/Test', results_path='Project/Results/Test', visualize=True):
    print("\n") 
    print("=================================") 
    print("==== Color Examination Start ====")
    print("=================================") 
    
    selected_images = image_selector(input_path)
    selected_images.sort()  # Sort by image number
    results_color = {}
    results_size = {}
    
    for image_number, image_path in selected_images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        mean_values = generate_color_histogram(image_path, visualize=visualize)
        non_black_percentage = calculate_non_black_percentage(image_path)
        results_color[image_name] = mean_values
        results_size[image_name] = non_black_percentage
        
    # Write the results dictionary to a JSON file
    if results_color:
        with open(os.path.join(results_path, 'color_histograms.json'), 'w') as json_file:
            json.dump(results_color, json_file, indent=4)
    
    if results_size:
        with open(os.path.join(results_path, 'non_black_percentage.json'), 'w') as json_file:
            json.dump(results_size, json_file, indent=4)
                
    print("\n") 
    print("=================================") 
    print("===== Color Examination End =====")
    print("=================================")

if __name__ == "__main__":
    main(visualize=False)