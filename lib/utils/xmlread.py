# XML Reader file
from copy import deepcopy
import re
import warnings

class XMLReader():
    
    def __init__(self):

        # model information
        self.n_search_axes=None
        self.min_axis_values=[]
        self.max_axis_values=[]
        self.axis_logscale=[]
        self.trainable_parameter_names=[]
        self.fixed_parameter_names=[]
        self.fixed_parameter_values=[]
        self.integrated_variable_names=[]
        self.integrated_variable_init_values=[]

        # TODO read a boundary condition for DAEs

        #  population optimizer settings
        self.n_particles=None
        self.n_iters_pop=None
        self.processors=None #TODO not implemented 
        # Gradient based optimizer settings
        self.n_iters_grad=None
        self.stepsize_rtol=None
        self.stepsize_atol=None
        self.init_timestep=None
        self.max_steps=None
        self.init_time=None
        self.init_value_lr=None
        self.end_value_lr=None
        self.transition_steps_lr=None
        self.decay_rate_lr=None
        self.error_loss=5000.0
        #-------------
        # Description of data
        self.filename_data = None
        self.data_column_index=[]
        self.data_column_names=[]
        #------------

        self.write_results= False
        self.output_dir=None

    def get_trailing_number(self,s):
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else None

    def read_XML(self,root):

        for child in root:
        
            if child.tag=="EXPERIMENT":
            
                for child2 in child:
              
                    if child2.tag =="FILENAME":
                        for child3 in child2:
                            name=child3.text.split('=')[0].strip()
                            value=child3.text.split('=')[1].strip()  
                #for key,value in child2.attrib.items():
                
                            if name=="FILENAME_DATA":
                            
                                self.filename_data=value
                    elif child2.tag == "COLUMN_INFO":
                        for child3 in child2:

                            for child4 in child3:
                                name=child4.text.split('=')[0].strip()
                                value=child4.text.split('=')[1].strip()  
                            
                            if name=="NAME":

                                self.data_column_names.append(value)
                            elif name=="COLUMN_INDEX":

                                self.data_column_index.append(int(value))

            elif child.tag=="MODEL":

                for child2 in child:

                    if child2.tag=="TRAINABLE_PARAMETER_DESCRIPTION":
                        
                        for child3 in child2:
                       
                            for child4 in child3:
                                name=child4.text.split('=')[0].strip()
                                value=child4.text.split('=')[1].strip()
                                if name=="PARAMETER_NAME":
                                    self.trainable_parameter_names.append(value)         
                                elif name=="MIN_VAL":
                                    self.min_axis_values.append(float(value)) 
                                elif name=="MAX_VAL":

                                    self.max_axis_values.append(float(value)) 
                                elif name=="LOGSCALE":

                                    if value=="N":
                                        self.axis_logscale.append(int(0))
                                    elif value=="Y":
                                        self.axis_logscale.append(int(1))
                                    else:
                                        raise ValueError("Unknown logscale value")
                                else:
                                    raise ValueError("Unknown entry inside PARAM")

                    elif child2.tag=="INTEGRATED_SYSTEM_DESCRIPTION":

                        for child3 in child2:

                            for child4 in child3:

                                name=child4.text.split('=')[0].strip()
                                value=child4.text.split('=')[1].strip()

                                if name=="NAME":

                                    self.integrated_variable_names.append(value)
                                elif name=="INIT_VAL":
                                    self.integrated_variable_init_values.append(float(value))
                                    

                    elif child2.tag=="TRAINABLE_PARAMETERS":

                        for child3 in child2:

                            name=child3.text.split('=')[0].strip()
                            value=child3.text.split('=')[1].strip()
                            
                            if name=="N_TRAINABLE_PARAMETERS":

                               self.n_search_axes=int(value)
                            else:
                                raise ValueError
                
                    elif child2.tag=="FIXED_PARAM_DESCRIPTION":
            
                        for child3 in child2:

                            for child4 in child3:
                                name=child4.text.split('=')[0].strip()
                                value=child4.text.split('=')[1].strip()
                                
                                if name=="NAME":

                                    self.fixed_parameter_names.append(value)
                                elif name=="VALUE":
                                    self.fixed_parameter_values.append(float(value))

            elif child.tag=="POPULATION_OPT":
                
                for child2 in child:
                    

                    if child2.tag=="SETTINGS":
                    
                        for child3 in child2:
                            name=child3.text.split('=')[0].strip()
                            value=child3.text.split('=')[1].strip()
                        
                            if name=="NUM_PARTICLES":
                                self.n_particles=int(value)
                            elif name=="NUM_ITERS":
                            
                                self.n_iters_pop=int(value)   
                            elif name=="PROCESSORS":
                                self.processors=int(value)
                            else:
                                raise ValueError
                        


            elif child.tag=="GRADIENT_OPT":

                for child2 in child:

                    if child2.tag=="SETTINGS":
                    
                        for child3 in child2:
                        
                            name=child3.text.split('=')[0].strip()
                            value=child3.text.split('=')[1].strip()
                    
                            if name=="NUM_ITERS":
                                self.n_iters_grad=int(value)
                            elif name=="STEPSIZE_RTOL":
                                # Handle comma-separated values
                                rtol_values = [float(x.strip()) for x in value.split(',')]
                                self.stepsize_rtol=rtol_values

                            elif name=="STEPSIZE_ATOL":
                                # Handle comma-separated values
                                atol_values = [float(x.strip()) for x in value.split(',')]
                                self.stepsize_atol=atol_values
                            elif name=="INITIAL_TIMESTEP":
                                self.init_timestep=float(value)
                            elif name=="INITIAL_TIME":
                                self.init_time=float(value)
                            elif name=="MAX_STEPS":
                                self.max_steps=int(value)
                            elif name=="INIT_VALUE_LR":
                                self.init_value_lr=float(value)
                            elif name=="END_VALUE_LR":
                                self.end_value_lr=float(value)
                            elif name=="TRANSITION_STEPS_LR":
                                self.transition_steps_lr=float(value)
                            elif name=="DECAY_RATE_LR":
                                self.decay_rate_lr=float(value)
            elif child.tag=="PLOTTING_INFO":

                for child2 in child:

                    name=child2.text.split('=')[0].strip()
                    value=child2.text.split('=')[1].strip()

                    if name =="WRITE_RESULTS":
                        if value=="Y":
                            self.write_results=True
                        elif value=="N":
                            self.write_results=False
    # check is all declared names by the user are unique
    # raise ValueError if false
    def check_name_uniqueness(self):
        combined_list= self.trainable_parameter_names+self.fixed_parameter_names+self.integrated_variable_names
        if len(combined_list) > len(set(combined_list)):

            raise ValueError("The provided names of every simulation element (trainable parameters, fixed parameters and integrated variable name) is not unique. Please provide unique names for each.")
