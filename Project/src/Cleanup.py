import os
import json
import glob

run = 5

specifier = 'Pipeline'
working_directory = f'Project/Results/{specifier}/RUN_{run}'

def main(input_path = working_directory, full_cleanup = True):
    if full_cleanup:
        print("\n") 
        print("=================================") 
        print("======== Cleanup Start ==========")
        print("=================================") 
        
        directory = input_path
    
        # Remove all files in the directory outside of "Final_Results" subfolder
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file != 'Final_Results':
                os.remove(file_path)       
        
        print("\n") 
        print("=================================") 
        print("======== Cleanup End ============")
        print("=================================") 
        
    else:    
        print("\n") 
        print("=================================") 
        print("===== Cleanup not active ========")
        print("=================================") 
        
        
if __name__ == "__main__":
    main()