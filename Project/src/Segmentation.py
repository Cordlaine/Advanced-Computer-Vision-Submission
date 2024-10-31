import os
import glob
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
import json
import re


def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    

def load_local_images(path):
    path = os.path.normpath(path)
    images = glob.glob(os.path.join(path, '*.jpeg'))
    images.extend(glob.glob(os.path.join(path, '*.jpg')))
    images.extend(glob.glob(os.path.join(path, '*.png')))
    return images


def load_remote_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return [img]


def load_model(model_used):
    if model_used == "yolov8":
        return YOLO("yolov8m-seg.pt") 
    else:
        raise ValueError("Unsupported model type")


def main(model_used="yolov8", conf=0.5, input_path='Project/Examples', results_path='Project/Results/Test', visualize=True):
    print("\n") 
    print("=================================") 
    print("==== Mask Segmentation Start ====")
    print("=================================") 
    
    # Load images
    images = load_local_images(input_path)
    
    # Pre check images
    if not images:
        print("No images found.")
        exit()  
    
    # Load model
    model = load_model(model_used)

    # Prepare results path
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    # Dictionary to store centroid coordinates
    centroids = {}

    # Define colors
    dark_blue = (139, 0, 0)  # Dark blue in BGR format
    orange = (0, 165, 255)   # Orange in BGR format

    for i, image_path in enumerate(images):
        # Extract the image number if the naming follows the pattern 'scene_XX_0001.png'
        image_name = os.path.basename(image_path)
        match = re.match(r'^scene_\d+_(\d+)\.png$', image_name)
        if match:
            try:
                image_number = int(match.group(1))
                i = image_number
            except ValueError:
                pass  # If conversion fails, keep the original index
        
        print("\n")    
        print(f"Operating on {image_path}...\n")
            
        img = cv2.imread(image_path)
        img_copy = img.copy()   
        img_mask = img.copy() # Visualize all masks on the original image
        combined_masked_pixels = np.zeros_like(img)
        results = model.predict(img, conf=conf)
        
        interest_flag = False
        
        if visualize:
            # Display the original image
            cv2.imshow(f"Original: Image {i}: {image_path}", img_copy)
            cv2.waitKey(0)
        
        for result in results:
            if result.masks:  # Check if masks are not None
                for mask_index, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Only process masks of type "apple"
                    if class_name != "apple":
                        continue
                    
                    # Set the flag to True if at least one mask is found
                    interest_flag = True
                    
                    points = np.int32([mask])
                    
                    # Create a binary mask
                    binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(binary_mask, points, 255)
                    
                    # Access the pixels in the original image that are contained by the mask
                    masked_pixels = cv2.bitwise_and(img_copy, img_copy, mask=binary_mask)
                    
                    # Overlay the masked pixels onto the combined image
                    combined_masked_pixels = cv2.add(combined_masked_pixels, masked_pixels)
                    
                    # Calculate the moments of the binary mask
                    moments = cv2.moments(binary_mask)

                    # Compute the centroid coordinates
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                    else:
                        cX, cY = 0, 0

                    # Store the coordinates in the dictionary
                    centroids[f"Image_{i}_Mask_{mask_index}"] = {"centroid_x": cX, "centroid_y": cY}

                    # Fill the mask with dark blue color                 
                    cv2.fillPoly(img_mask, points, dark_blue)
                    
                    # Draw bounding box
                    start_point = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                    end_point = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                    cv2.rectangle(img_mask, start_point, end_point, dark_blue, 2)
                    
                    # Add label with class name and confidence
                    confidence = box.conf[0]  # Assuming box.conf contains the confidence score
                    label = f"{class_name}: {confidence:.2f}"
                           
                    # Draw the centroid on the image
                    cv2.circle(img_mask, (cX, cY), 5, orange, -1)         
                    cv2.putText(img_mask, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, dark_blue, 2)  
                    
                    # Save the masked pixels image                  
                    cv2.imwrite(os.path.join(results_path, f'Image_{i}_Mask_{mask_index}.jpg'), masked_pixels)
                
            else:
                print("No masks found for this result.")
        
        # Only save the resulting images if at least one mask was found
        if interest_flag:        
            cv2.imwrite(os.path.join(results_path, f'Result_{i}.jpg'), img_mask)
            cv2.imwrite(os.path.join(results_path, f'Combined_Masked_Pixels_{i}.jpg'), combined_masked_pixels)

            # Visualize the results 
            if visualize:        
                cv2.imshow(f"Masks Image {i}: {image_path}", img_mask)
                cv2.imshow(f"Combined Masked Pixels {i}", combined_masked_pixels)
                cv2.waitKey(0)        

            # Write the centroids dictionary to a JSON file
            if centroids:
                with open(os.path.join(results_path, 'centroids.json'), 'w') as f:
                    json.dump(centroids, f, indent=4)
        
    print("\n") 
    print("=================================") 
    print("===== Mask Segmentation End =====")
    print("=================================") 


if __name__ == "__main__":
    main(visualize=False)