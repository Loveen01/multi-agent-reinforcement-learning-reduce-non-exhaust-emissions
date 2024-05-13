import json 
from utils.utils import CustomEncoder
import os 

def save_data_under_path(data:dict, path:str, file_name:str):
    full_path = os.path.join(path, file_name)
    # try: 
    with open(full_path, 'w') as f:
        json.dump(obj=data, fp=f, cls=CustomEncoder, indent=4, sort_keys=True)
        # print(f"File has been successfully saved under {full_path}")
    # except TypeError as e: 
    #     print(f"Error serialising data to JSON: {e}")
    # except IOError as e:
    #     print(f"Error writing to file {full_path}: {e}")