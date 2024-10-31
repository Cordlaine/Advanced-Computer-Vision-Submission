import Segmentation
import Retreive_Depth
import Examination
import Interpretation
import Cleanup

# Set the run number
run = 1

# Tidy up working directory; Expect final results only (./Results/Final_Results will be generated)
full_cleanup = True

# Visialization of intermediate results
visualize = False

# Set the working directory
specifier = 'Pipeline'
working_directory = f'Project/Results/{specifier}/RUN_{run}'

# Input directories
image_directory = f'Project/Examples_ZED/RGB_left'
depth_directory = f'Project/Examples_ZED/depth'

def main():
    # Ensure Segmentation runs first and completes
    Segmentation.main(input_path=image_directory, results_path=working_directory, visualize=visualize)
    
    # Then run Retreive_Depth
    Retreive_Depth.main(input_path=depth_directory, results_path=working_directory, coordinates_path=working_directory, visualize=visualize)
    
    # Then run Examination
    Examination.main(input_path=working_directory, results_path=working_directory, visualize=visualize)
    
    # Then run Interpretation
    Interpretation.main(input_path=working_directory, results_path=working_directory, image_directory=image_directory)
    
    # Finally run Cleanup
    Cleanup.main(input_path=working_directory, full_cleanup=full_cleanup)


if __name__ == "__main__":
    main()