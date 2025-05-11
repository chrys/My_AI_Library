import configparser
import os

# def read_config(section: str, key: str) -> str:
#     """
#     Read sensitive data from the configuration file.

#     Business Requirement:
#     - Read sensitive data securely.

#     Technical Requirements:
#     - Read sensitive data from file ~/.config.ini in the form of:
#       [Section]
#       key = value

#     Args:
#         section (str): The section in the config file.
#         key (str): The key within the section to retrieve the value for.

#     Returns:
#         str: The value associated with the given section and key.

#     Raises:
#         FileNotFoundError: If the config file does not exist.
#         KeyError: If the section or key is not found in the config file.
#     """
#     # Determine the environment
#     environment = os.getenv('WHAT_ENV', 'local')

#     # Set the config path based on the environment
#     if environment == 'production':
#         config_path = os.path.expanduser('/etc/django-secrets/.config.ini')
#     else:
#         config_path = os.path.expanduser('~/.config.ini')
    
#     # Check if the config file exists
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
    
#     # Initialize the config parser
#     config = configparser.ConfigParser()
    
#     # Read the config file
#     config.read(config_path)
    
#     # Retrieve the value from the specified section and key
#     try:
#         value = config[section][key]
#     except KeyError as e:
#         raise KeyError(f"Section or key not found in config file: {e}")
    
#     return value

def read_config(section: str, key: str) -> str:
    #read from .env file
    from dotenv import load_dotenv
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, '.env')
    dotenv_loaded = load_dotenv(env_path)
    value_loaded = os.getenv(key)
    
    if value_loaded is None:
        error_msg = f"Configuration key '{key}' not found in environment variables. "
        if os.path.exists(project_root): # Check if .env file exists, even if not loaded or key not in it
            if dotenv_loaded:
                error_msg += f"A .env file was loaded from '{project_root}', but the key was not present there or was empty."
            else:
                # This case might occur if .env exists but is unreadable, though unlikely with load_dotenv's default.
                error_msg += f"A .env file exists at '{project_root}', but the key was not found (or file was not loaded for other reasons)."
        else:
            error_msg += f"No .env file was found at '{project_root}' to load development/local variables from."
        raise KeyError(error_msg)
    return value_loaded

#main to test read_config
if __name__ == "__main__":
    try:
        # Test with a known section and key
        section = 'DEFAULT'
        key = 'postgres2'
        value = read_config(section, key)
        print(f"Value for {section}.{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    