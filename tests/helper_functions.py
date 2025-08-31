
import sys
import os
from fit_parameters import run_driver
from lib.utils.helper_functions import get_input_reader
from pathlib import Path

def run_driver_wrapper(session_dir):
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    input_file_path = session_dir / Path("inputs") / Path("user_input.xml")
    input_reader = get_input_reader(input_file_path)
    return run_driver(session_dir,input_reader)




