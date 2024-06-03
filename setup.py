import yaml
import os

# Function to read the YAML file
def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to write to the YAML file
def write_config(file_path, config):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def main():
    # Get the root path of the current script
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the DLC directory
    dlc_path = os.path.join(root_path, "automated_analysis")
    
    # Define the path to the YAML file
    yaml_file_path = os.path.join(dlc_path, "config.yaml")
    
    # Read the YAML file
    config = read_config(yaml_file_path)
    
    # Update the configuration
    config['project_path'] = dlc_path
    
    # Write the updated config back to the YAML file
    write_config(yaml_file_path, config)
    
    print("Updated config file with project path.")

if __name__ == "__main__":
    main()
