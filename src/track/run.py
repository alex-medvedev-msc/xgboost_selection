import os
import uuid
import json


class Run:
    def __init__(self, runs_dir, trait_name, snapshot=None):
        if not os.path.exists(runs_dir):
            os.mkdir(runs_dir)
        
        self.trait_name = trait_name
        if snapshot is None:
            self.dir = f'{runs_dir}/{trait_name}/{uuid.uuid4()}'
        else:
            self.dir = f'{runs_dir}/{trait_name}/{snapshot}'
            
        self.models_dir = f'{self.dir}/models'
        self.results_dir = f'{self.dir}/results'
        self.params_file = f'{self.dir}/params.json'
        
        for d in [f'{runs_dir}/{trait_name}', self.dir, self.models_dir, self.results_dir]:
            if not os.path.exists(d):
                os.mkdir(d)
            
    
    def model_path(self, model_name):
        return f'{self.models_dir}/{model_name}'
    
    def results_path(self, result_name):
        return f'{self.results_dir}/{result_name}'
        
    def save_params(self, description, params):
        data = {
            'description': description,
            'params': params
        }
        with open(self.params_file, 'w') as file:
            json.dump(data, file)
            
    def add_data_to_params(self, to_add):
        data = self.get_params()
        if 'additional_data' not in data:
            data['additional_data'] = [to_add]
        else:
            data['additional_data'].append(to_add)
            
        with open(self.params_file, 'w') as file:
            json.dump(data, file)
            
    def get_params(self):
        with open(self.params_file, 'r') as file:
            return json.load(file)
            
    def __str__(self):
        return self.dir
            
    def __repr__(self):
        return self.dir
    

