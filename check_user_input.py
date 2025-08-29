from pathlib import Path
from lib.LLM.classes import OpenAI_model
import os
import sys
from lib.utils.helper_functions import get_input_reader

def check_input(session_dir,input_reader):
    """Check user input with session-specific paths"""
    print("Launching script to check user input")

    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.xml"
    path_to_user_code = session_path / input_reader.generated_dirname / "user_model.py"
    path_to_output = session_path / input_reader.generated_dirname / "raw_user_input_check.txt"

    path_to_input_check_model_output = session_path / input_reader.generated_dirname / "raw_user_input_check.txt"
    reference_file_paths_check_model_output = []
    path_to_output_check_model_output = session_path / input_reader.generated_dirname / "user_input_check.txt"
    
    # First check of user input
    if path_to_output.exists():
        path_to_output.unlink()

    # check if output file exists, if so delete it
    if path_to_output_check_model_output.exists():
        path_to_output_check_model_output.unlink()


    dev_instr_filename = Path(os.getcwd()) / Path("lib/LLM/user_file_check_instructions.txt")

    
    reference_file_paths = []
    reference_file_paths.append(path_to_user_code)

    input_check_model = OpenAI_model(
        path_to_input,
        reference_file_paths,
        api_key_string="OPENAI_ENV_KEY",
        dev_instr_filename=dev_instr_filename,
        role="code_input_checker"
    )
    print("Checking user inputs....")
    input_check_model.check_generated_files(path_to_output)
    print("Finished first check of user input")

    # refining the output of model
    # currently this is just a single iteration workflow
    # in the future this can be replaced by an agentic setup 

    

    # check model output
    dev_instr_filename_check_output = Path(os.getcwd()) / Path("lib/LLM/model_output_check_inputcheck_instructions.txt")
    output_check_model = OpenAI_model(
        path_to_input_check_model_output,
        reference_file_paths_check_model_output,
        api_key_string="OPENAI_ENV_KEY",
        dev_instr_filename=dev_instr_filename_check_output,
        role="model_output_checker"
    )
    print("Checking model report....")
    output_check_model.check_model_output_inputcheck(path_to_output_check_model_output)
    print("Finished checking model report")

    # print resulting file to terminal
    with open(path_to_output_check_model_output, "r") as file:
        print(file.read())
    


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_ENV_KEY")

    if not os.path.isdir("sessions"):
        raise ValueError("No sessions directory found. Please create a sessions directory and make a subdir structure as shown in the README.md file")

    sessions_root = Path("sessions")

    if len(sys.argv) == 2:
        session_dir = sessions_root / Path(sys.argv[1])
        
        if not os.path.isdir(session_dir):
            raise ValueError(f"Session directory {session_dir} does not exist")
    else:

        session_dirs = [d for d in sessions_root.iterdir() if d.is_dir()]
        
        if not session_dirs:
            print("No session directories found in ./sessions.")
            sys.exit(1)
        # Sort by modification time, descending
        session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        session_dir = str(session_dirs[0])
        print(f"No session_dir provided. Using most recent session: {session_dir}")
    # Pass the API key to check_input via environment or directly if needed
    input_file_path = session_dir / Path("inputs") / Path("user_input.xml")
    input_reader = get_input_reader(input_file_path)
    check_input(session_dir,input_reader)

        
