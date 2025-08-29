import os
import openai
from openai import OpenAI
import base64
import requests
from io import BytesIO

# Base Class for LLM interfacing

class LLMBase():

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


    def set_API_key(self,API_key_string=None):
        
        print("Note: Set the env key as an environemtn variable")
        self.API_key=os.getenv(API_key_string)

    def init_client(self):

        return self._init_client()

    def generate_user_model(self,input_file_path,input_sample_path,user_model_sample_path,user_model_path):
        return self._generate_user_model(input_file_path,input_sample_path,user_model_sample_path,user_model_path)

    # Provide path to created file and generate system
    def generate_system(self,system_path,output_file_path=None):

        return self._generate_system(system_path,output_file_path)
    # check the user input for errors
    def check_generated_files(self,output_file_path):

        return self._check_generated_files(output_file_path)

    def check_model_output_inputcheck(self,output_file_path,num_iterations=1):

        return self._check_model_output_inputcheck(output_file_path,num_iterations)

class OpenAI_model(LLMBase):
   
    # Sets API key and initializes client
    def __init__(self, input_file_path, reference_file_paths, api_key_string, dev_instr_filename, role):
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

    def _init_client(self):
        self.client = OpenAI(api_key=self.API_key)


    def _set_developer_instructions(self):
        
        with open(self.dev_instr_filename,"r",encoding="utf=8") as f:

            self.dev_instr=f.read()

    def _set_user_instructions(self):

            raise NotImplementedError

    # query the model to generate the required diffeq system
    def _generate_system(self,system_path,output_file_path):

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

    def _check_generated_files(self,output_file_path):
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

    def _check_model_output_inputcheck(self,output_file_path,num_iterations=1):

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

    def _generate_user_model(self,input_file_path,input_sample_path,user_model_sample_path,user_model_path):

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