import yaml

def read_one_block_of_yaml_data(filename):
      with open(f'{filename}.yaml','r') as f:
          output = yaml.safe_load(f)
      return(output)