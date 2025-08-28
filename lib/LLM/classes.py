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

    # Provide path to created file and generate system
    def generate_system(self,system_path,output_file_path=None):

        return self._generate_system(system_path,output_file_path)
    # check the user input for errors
    def check_generated_files(self,output_file_path):

        return self._check_generated_files(output_file_path)

    def check_model_output_inputcheck(self,output_file_path):

        return self._check_model_output_inputcheck(output_file_path)

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
        print("WARNING: hardcoded model name ")
        self.vendor="openai"
        self.model="gpt-4.1"
        self.vector_store_ids=[None] 

        # instructions input (dev and user instructions)
        self.dev_instructions_filename=None
        
        self._init_client()

    def _init_client(self):
        self.client = OpenAI(api_key=self.API_key)

    # convert input image to base64 for input to OpenAI model
    def _encode_image(self,image_path):
        with open(image_path, "rb") as image_file:

            self.base64_image= base64.b64encode(image_file.read()).decode("utf-8")

    def _setup_vector_store(self,files_path_list):

        self._create_vector_store()

        self._upload_files(files_path_list)

    # vector store commands
    # need to create a new vector store and delete it after use
    def _create_vector_store(self):

        # store created vector store path
        vector_store = self.client.vector_stores.create(
        name="knowledge_base"
            )
        self.vector_store_ids[0]=vector_store.id


    # upload files to vector store
    def _upload_files(self,files_path_list):

        # function to prepare the file for upload
        def create_file(file_path):
            if file_path.startswith("http://") or file_path.startswith("https://"):
                # Download the file content from the URL
                response = requests.get(file_path)
                file_content = BytesIO(response.content)
                file_name = file_path.split("/")[-1]
                file_tuple = (file_name, file_content)
                result = self.client.files.create(
                    file=file_tuple,
                    purpose="assistants"
                )
            else:
                # Handle local file path
                with open(file_path, "rb") as file_content:
                    result = self.client.files.create(
                        file=file_content,
                        purpose="assistants"
                    )
            print("Resultant file ID:",result.id)
            return result.id        

        for file_path in files_path_list:

            file_id=create_file(str(file_path))
            print("File ID:",file_id)
            print("Vector store:",self.vector_store_ids)
            result=self.client.vector_stores.files.create(
                vector_store_id=self.vector_store_ids[0],
                file_id=file_id
                )
            print("File upload result:",result)

    def _delete_vector_store(self):

        self.client.vector_stores.delete(vector_store_id=self.vector_store_ids[0])

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
        file_path_list=self.reference_file_paths+[system_path]

        self._setup_vector_store(file_path_list)
        #TODO set user instr
        print("TODO: set user instructions")

        # load xml as text
        with open(self.input_file_path, "r") as f:
            xml_content = f.read()


        response= self.client.responses.create(
                    model="gpt-4.1",
                    instructions=self.dev_instr,
                    input= [{
                            "role":"user",
                            "content":[
                                {"type":"input_text", "text":f"Given to you is the .py corresponding to the problem system. Further, the user has input settings in the form of an XML, here it is: {xml_content}"},
                                #{
                                #    "type":"input_image",
                                #    "image_url":f"data:image/png;base64,{self.base64_image}"

                                #},


                    ],
                    }],
                    tools=[{
                            "type":"file_search",
                            "vector_store_ids":self.vector_store_ids
                            }]
                    )
        
        with open(output_file_path,"w") as f:

            f.write(response.output_text)
            f.close()

        print("Finished writing code")

        self._delete_vector_store()

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

    def _check_model_output_inputcheck(self,output_file_path):

        print("Checking model report.....")

        self._set_developer_instructions()

        with open(self.input_file_path, "r") as f:
            model_writeout_content = f.read()
        
        for i_iter in range(1):
            print(f"Checking user uploads {i_iter+1} of 3")
            
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

