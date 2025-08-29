from pathlib import Path
from lib.LLM.classes import OpenAI_model
import os
import sys
from lib.utils.helper_functions import get_input_reader

def create_user_model(session_dir,input_reader):
    """Check user input with session-specific paths"""
    print("Launching script to create user model")

    # check if necessary dirs exist in session_dir
    if not os.path.isdir(session_dir / input_reader.user_input_dirname):
        print(f"User input directory {session_dir / input_reader.user_input_dirname} does not exist, making it...")
        os.makedirs(session_dir / input_reader.user_input_dirname)
    if not os.path.isdir(session_dir / input_reader.generated_dirname):
        print(f"Generated directory {session_dir / input_reader.generated_dirname} does not exist, making it...")
        os.makedirs(session_dir / input_reader.generated_dirname)
    if not os.path.isdir(session_dir / input_reader.output_dirname / "inputs"):
        print(f"Output directory {session_dir / input_reader.output_dirname / 'inputs'} does not exist, making it...")
        os.makedirs(session_dir / input_reader.output_dirname / "inputs")

    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.xml"
    path_to_user_model = session_path / input_reader.generated_dirname / "user_model.py"

    path_to_user_input_sample = Path(os.getcwd()) / Path("lib/utils/user_input_sample.xml")
    path_to_user_model_sample = Path(os.getcwd()) / Path("lib/utils/user_model_sample_unpopulated.py")


    dev_instr_filename = Path(os.getcwd()) / Path("lib/LLM/user_model_generation_instructions.txt")

    reference_file_paths = []
    

    user_model_generator = OpenAI_model(
        path_to_input,
        reference_file_paths,
        api_key_string="OPENAI_ENV_KEY",
        dev_instr_filename=dev_instr_filename,
        role="user_model_generator"
    )
    print("Generating user model....")
    user_model_generator.generate_user_model(path_to_input,path_to_user_input_sample,path_to_user_model_sample,path_to_user_model)
    print(f"Written user model skeleton to {path_to_user_model} using {path_to_input},\
           now you can write the python code for the user model in the file")


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
    create_user_model(session_dir,input_reader)

        
