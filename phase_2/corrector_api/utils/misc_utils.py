
import json

def read_json_file(file_path):
    """
    Reads a JSON file from the given file path and returns the data.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    data (dict): The data loaded from the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as error:
        print(f"Error decoding JSON: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None