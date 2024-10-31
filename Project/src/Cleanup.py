import os
import json
import glob

run = 5

specifier = 'Pipeline'
working_directory = f'Project/Results/{specifier}/RUN_{run}'

def main(input_path = working_directory, full_cleanup = True):
    print("\n") 
    print("=================================") 
    print("======== Cleanup Start ==========")
    print("=================================") 
    
    directory = input_path
    
    if full_cleanup:
        # Remove all files in the directory outside of "Final_Results" subfolder
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file != 'Final_Results':
                os.remove(file_path)       
        
        print("\n") 
        print("=================================") 
        print("======== Cleanup End ============")
        print("=================================") 
        
        return

    # Initialize an empty dictionary to store the merged data
    merged_data = {}

    # Iterate through each JSON file in the directory
    for json_file in glob.glob(os.path.join(directory, '*.json')):
        with open(json_file, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                if key in merged_data:
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value

    # Reorder the keys in each entry so that centroid_x, centroid_y, depth come first
    for key, value in merged_data.items():
        reordered_value = {}
        if 'centroid_x' in value:
            reordered_value['centroid_x'] = value.pop('centroid_x')
        if 'centroid_y' in value:
            reordered_value['centroid_y'] = value.pop('centroid_y')
        if 'depth' in value:
            reordered_value['depth'] = value.pop('depth')
        reordered_value.update(value)
        merged_data[key] = reordered_value

    # Write the merged dictionary to a new JSON file
    merged_file_path = os.path.join(directory, 'merged_data.json')
    with open(merged_file_path, 'w') as merged_file:
        json.dump(merged_data, merged_file, indent=4)

    # Delete the original JSON files
    for json_file in glob.glob(os.path.join(directory, '*.json')):
        if json_file != merged_file_path:
            os.remove(json_file)
    
    print("\n") 
    print("=================================") 
    print("======== Cleanup End ============")
    print("=================================") 
        
if __name__ == "__main__":
    main()