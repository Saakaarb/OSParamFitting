
import sys
import os
from driver_script import run_driver

def run_driver_wrapper(session_dir):
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    return run_driver(session_dir)




