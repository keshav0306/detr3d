import pickle

def load_and_analyze_pickle(file_path):
    """
    Load and analyze a pickle file.

    :param file_path: Path to the pickle file.
    """
    try:
        # Load the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Analyze the data
        print("Data type:", type(data))
        print(data['infos'][0])
        
        if isinstance(data, dict):
            print("Dictionary keys:", list(data.keys()))
            for key, value in data.items():
                print(f"Key: {key}, Value type: {type(value)}")
                # if isinstance(value, (list, dict)):
                #     print(f"Sample content of '{key}':", value if len(value) <= 5 else str(value)[:500], "...")
        
        elif isinstance(data, list):
            print("List length:", len(data))
            print("Sample items in list:", data[:5])
        
        else:
            print("Data content:", data)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.UnpicklingError:
        print(f"Error unpickling file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Provide the path to your pickle file
file_path = "/ssd_scratch/cvit/keshav/nuscenes/nuscenes_infos_temporal_train.pkl"

# Call the function
load_and_analyze_pickle(file_path)
