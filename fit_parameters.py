from pathlib import Path
from lib.LLM.classes import OpenAI_model
import os
import shutil
import sys
from lib.utils.helper_functions import fit_generic_system
from lib.utils.helper_functions import get_input_reader

def run_driver(session_dir,input_reader):
    """Run the driver script with session-specific paths"""
    print("Launching driver script")

    # Use session-specific paths
    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.xml"
    path_to_output_dir = session_path / input_reader.output_dirname
    path_to_eq_system = session_path / input_reader.generated_dirname / "user_model.py"

    if path_to_output_dir.exists():
        shutil.rmtree(path_to_output_dir)
    path_to_output_dir.mkdir()

    reference_file_paths = []
    reference_file_paths.append(Path(os.getcwd()) / Path("lib/utils/output_sample.py"))

    dev_instr_filename = Path(os.getcwd()) / Path("lib/LLM/developer_instructions.txt")
    output_filename = session_path / "generated" / "generated_script.py"

    # Generate necessary scripts
    code_model = OpenAI_model(
        path_to_input,
        reference_file_paths,
        api_key_string="OPENAI_ENV_KEY",
        dev_instr_filename=dev_instr_filename,
        role="code_writer"
    )
    code_model.generate_system(path_to_eq_system, output_filename)

    # Run the fitting process
    print("Launching fitting process...")
    generated_dir = session_path / "generated"
    fit_generic_system(path_to_input, path_to_output_dir, generated_dir,session_path)

if __name__ == "__main__":
    
    if not os.path.isdir("sessions"):
        raise ValueError("No sessions directory found. Please create a sessions directory and make a subdir structure as shown in the README.md file")

    sessions_root = Path("sessions")

    if len(sys.argv) == 2:
        session_dir = sessions_root / Path(sys.argv[1])
        
        if not os.path.isdir(session_dir):
            raise ValueError(f"Session directory {session_dir} does not exist")
    else:
        # Find the most recently created session directory
        
        if not sessions_root.exists() or not any(sessions_root.iterdir()):
            print("No session_dir provided and no sessions/ directory found or it is empty.")
            print("Usage: python driver_script.py <session_dir>")
            sys.exit(1)
        # Get all directories in sessions/, sort by creation time descending
        session_dirs = [d for d in sessions_root.iterdir() if d.is_dir()]
        
        session_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
        session_dir = str(session_dirs[0])
        print(f"No session_dir provided. Using most recently created session: {session_dir}")

    input_file_path = session_dir / Path("inputs") / Path("user_input.xml")
    input_reader = get_input_reader(input_file_path)

    run_driver(session_dir,input_reader) 
