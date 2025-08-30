import os
from openai import OpenAI
from pathlib import Path

# Base Class for LLM interfacing

class LLMBase():
    """
    Base class for Large Language Model (LLM) interfacing in the OSParamFitting framework.

    This abstract base class provides a common interface for different LLM vendors and models.
    It defines the basic structure for API key management, client initialization, and core
    LLM operations like code generation, system checking, and user model creation.

    The class serves as a foundation for vendor-specific implementations (e.g., OpenAI)
    and ensures consistent behavior across different LLM providers.

    Attributes:
        name (str, optional): Problem name identifier
        vendor (str, optional): Model vendor/provider name
        model (str, optional): Specific model identifier
        API_key (str, optional): API key for authentication
        client: LLM client instance for making API calls
    """
    def __init__(self):

        self.name=None # problem name

        # Vendor and model info
        #---------------------------
        self.vendor=None # model vendor
        self.model=None # model number
        self.API_key=None # env API key name 
        #---------------------------

        # input eq system information

        # client setup
        self.client=None


    def set_API_key(self,API_key_string:str=None)-> None:
        
        print("Note: Set the env key as an environemtn variable")
        self.API_key=os.getenv(API_key_string)

    def init_client(self)-> OpenAI:

        return self._init_client()

    def generate_user_model(self,input_file_path: Path,input_sample_path: Path,user_model_sample_path: Path,user_model_path: Path)-> None:
        return self._generate_user_model(input_file_path,input_sample_path,user_model_sample_path,user_model_path)

    # Provide path to created file and generate system
    def generate_system(self,system_path: Path,output_file_path: Path=None)-> None:

        return self._generate_system(system_path,output_file_path)
    # check the user input for errors
    def check_generated_files(self,output_file_path: Path)-> None:

        return self._check_generated_files(output_file_path)

    def check_model_output_inputcheck(self,output_file_path: Path,num_iterations: int=1)-> None:

        return self._check_model_output_inputcheck(output_file_path,num_iterations)

class OpenAI_model(LLMBase):
    """
    Concrete implementation of LLMBase for OpenAI's GPT models.

    This class provides OpenAI-specific functionality for the OSParamFitting framework,
    including GPT-4 integration for code generation, error checking, and user model
    creation. It handles API authentication, client setup, and OpenAI-specific API calls.

    The class is designed to work with OpenAI's GPT-4.1 model and provides methods
    for generating Python code, checking user inputs, and creating model skeletons
    based on XML configuration files.

    Attributes:
        input_file_path (Path): Path to the input XML configuration file
        reference_file_paths (list[Path]): List of reference files for context
        api_key_string (str): Environment variable name containing the API key
        dev_instr_filename (Path): Path to developer instructions file
        role (str): Role identifier for the LLM model
        vendor (str): Set to "openai" for this implementation
        model (str): Set to "gpt-4.1" for this implementation
        vector_store_ids (list): Vector store identifiers (currently unused)
        dev_instructions_filename (str, optional): Developer instructions filename
        dev_instr (str): Content of developer instructions
        client (OpenAI): OpenAI client instance
    """
    # Sets API key and initializes client
    def __init__(self, input_file_path: Path, reference_file_paths: list[Path], api_key_string: str, dev_instr_filename: Path, role: str):
        """
        Initialize the OpenAI_model instance with configuration and API setup.

        Args:
            input_file_path (Path): Path to the input XML configuration file
            reference_file_paths (list[Path]): List of reference files for context
            api_key_string (str): Environment variable name containing the OpenAI API key
            dev_instr_filename (Path): Path to developer instructions file
            role (str): Role identifier for the LLM model

        Raises:
            ValueError: If the API key is not found in the specified environment variable

        The constructor sets up the OpenAI client, validates the API key, and configures
        the model for GPT-4.1 usage. It also initializes the developer instructions.
        """
        super().__init__()
        self.input_file_path = input_file_path
        self.reference_file_paths = reference_file_paths
        self.api_key_string = api_key_string
        self.dev_instr_filename = dev_instr_filename
        self.role = role
        
        # Get API key from environment
        self.API_key = os.getenv(api_key_string)
        if not self.API_key:
            raise ValueError(f"API key not found in environment variable {api_key_string}")

        #TODO move this s.t it is user specified at input
        print("WARNING: hardcoded model name and vendor")
        self.vendor="openai"
        self.model="gpt-4.1"
        self.vector_store_ids=[None] 

        # instructions input (dev and user instructions)
        self.dev_instructions_filename=None
        
        self._init_client()

    def _init_client(self)-> None:
        self.client = OpenAI(api_key=self.API_key)


    def _set_developer_instructions(self)-> None:
        """
        Load and set developer instructions from the specified file.

        Reads the developer instructions file and stores its content in the dev_instr
        attribute. These instructions guide the LLM's behavior during code generation
        and validation tasks.
        """
        with open(self.dev_instr_filename,"r",encoding="utf=8") as f:

            self.dev_instr=f.read()

    def _set_user_instructions(self)-> None:

            raise NotImplementedError

    # query the model to generate the required diffeq system
    def _generate_system(self,system_path: Path,output_file_path: Path)-> None:
        """
        Generate system code using OpenAI's GPT-4.1 model.

        Args:
            system_path (Path): Path to the system definition file
            output_file_path (Path): Path where the generated system code will be written

        This method uses GPT-4.1 to analyze the system definition, XML configuration,
        and output sample to generate appropriate system code. The generated code is
        written to the specified output file.

        The process involves:
        1. Loading developer instructions
        2. Reading system, XML, and sample files
        3. Sending a structured prompt to GPT-4.1
        4. Writing the generated response to the output file
        """
        print("Starting code generation")

        self._set_developer_instructions()

        # setup vector store
        #file_path_list=self.reference_file_paths+[system_path]

        # load xml as text
        with open(self.input_file_path, "r") as f:
            xml_content = f.read()

        with open(system_path,"r") as f:
            system_content=f.read()

        with open(self.reference_file_paths[0],"r") as f:
            output_sample_content=f.read()

        response= self.client.responses.create(
                    model="gpt-4.1",
                    instructions=self.dev_instr,
                    input= [{
                            "role":"user",
                            "content":[
                                {"type":"input_text", "text":f"Given to you is the .py corresponding to the problem system: {system_content}. Further, the user has input settings in the form of an XML, here it is: {xml_content}. Here is an example of the output of the system: {output_sample_content}"},

                    ],
                    }],
                    )
        
        with open(output_file_path,"w") as f:

            f.write(response.output_text)
            f.close()

        print("Finished writing code")

    def _check_generated_files(self,output_file_path: Path)-> None:
        """
        Check generated files for errors using OpenAI's GPT-4.1 model.

        Args:
            output_file_path (Path): Path where the validation report will be written

        This method uses GPT-4.1 to analyze user-uploaded Python code against the
        XML configuration context to identify errors and provide a comprehensive
        validation report.

        The process involves:
        1. Loading developer instructions
        2. Reading the XML configuration and user code
        3. Sending an analysis prompt to GPT-4.1
        4. Writing the validation report to the output file
        """
        print("Checking user inputs.....")

        self._set_developer_instructions()
        
        # in this case, contains just 1 .py file
        file_path_list=self.reference_file_paths

        with open(self.input_file_path, "r") as f:
            xml_content = f.read()

        with open(file_path_list[0],"r") as f:
            user_code=f.read()

        print("Checking user uploads")
        response= self.client.responses.create(
                    model="gpt-4.1",
                    instructions=self.dev_instr,
                    input= [{
                            "role":"user",
                            "content":[
                                {"type": "input_text",
                                  "text":f"Analyze the given .py file for errors and provide a report on the issues found, with respect to the XML context provided, here is the XML context: {xml_content}, and here is the .py file: {user_code}"},


                    ],
                    }],
                    )
        with open(output_file_path,"w") as f:

            f.write(response.output_text)
            f.close()

        print("Finished checking user input")

    def _check_model_output_inputcheck(self,output_file_path: Path,num_iterations: int=1)-> None:
        """
        Check and refine model output through iterative validation using GPT-4.1.

        Args:
            output_file_path (Path): Path where the refined output will be written
            num_iterations (int, optional): Number of refinement iterations. Defaults to 1.

        This method performs iterative refinement of model outputs using GPT-4.1.
        It can run multiple iterations to improve the quality and clarity of the
        generated reports or outputs.

        The process involves:
        1. Loading developer instructions
        2. Reading the current model output
        3. Running multiple refinement iterations (if specified)
        4. Writing the final refined output to the output file
        """
        print("Checking model report.....")

        self._set_developer_instructions()

        with open(self.input_file_path, "r") as f:
            model_writeout_content = f.read()
        
        
        for i_iter in range(num_iterations):
            print(f"Checking user uploads: Iteration {i_iter+1} of {num_iterations}")
            
            response= self.client.responses.create(
                        model="gpt-4.1",
                        instructions=self.dev_instr,
                        input= [{
                                "role":"user",
                                "content":[
                                    {"type": "input_text",
                                    "text":f"Analyze the given report containing a list of critical errors and warnings, and re write the report based on the instructions provided. Here is the report:{model_writeout_content}"},
                        ],
                        }],
                        )
            model_writeout_content=response.output_text

        with open(output_file_path,"w") as f:

            f.write(response.output_text)
            f.close()

    def _generate_user_model(self,input_file_path: Path,input_sample_path: Path,user_model_sample_path: Path,user_model_path: Path)-> None:
        """
        Generate a user model skeleton using OpenAI's GPT-4.1 model.

        Args:
            input_file_path (Path): Path to the input XML configuration file
            input_sample_path (Path): Path to a sample input file for reference
            user_model_sample_path (Path): Path to a sample user model for reference
            user_model_path (Path): Path where the generated user model will be written

        This method uses GPT-4.1 to generate a user model skeleton based on the XML
        configuration and sample files. It analyzes the context and creates appropriate
        Python code templates that users can populate with their specific problem definitions.

        The process involves:
        1. Loading developer instructions
        2. Reading XML configuration and sample files
        3. Sending a generation prompt to GPT-4.1
        4. Overwriting any existing user model file
        5. Writing the generated skeleton to the output file

        Note:
            If a user model already exists at the target path, it will be overwritten.
        """
        print("Generating user model....")

        self._set_developer_instructions()

        with open(input_file_path, "r") as f:
            xml_content = f.read()

        with open(user_model_sample_path,"r") as f:
            user_model_sample_content=f.read()

        with open(input_sample_path,"r") as f:
            input_sample_content=f.read()

        print("Checking user uploads")
        response= self.client.responses.create(
                    model="gpt-4.1",
                    instructions=self.dev_instr,
                    input= [{
                            "role":"user",
                            "content":[
                                {"type": "input_text",
                                  "text":f"Analyze the given XML context: {xml_content}, and generate a user model skeleton based on the context. Here is a sample user model skeleton generated for the robertson equation system: {user_model_sample_content}. Here is the corresponding sample input file: {input_sample_content}"},
                    ],
                    }],
                    )
        
        if os.path.exists(user_model_path):
            print(f"Overwriting existing user model at {user_model_path}")
            os.remove(user_model_path)

        with open(user_model_path,"w") as f:

            f.write(response.output_text)
            f.close()

        print("Finished generating user model skeleton")