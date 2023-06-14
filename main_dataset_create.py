from scripts.dataset_builder import download_and_clean_dataset
from scripts.dataset_builder import remove_over_context_length_and_create_valid
import sys


if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    print("\n NOTE: Write the name of the .yaml file containing the dataset's specs!! Example: !python dataset_builder.py dataset")

print("\n The file you are testing is: {}.yaml".format(file_name))

download_and_clean_dataset(file_name)
print('\nDataset downloaded and cleaned')
print('Now removing any sentences too long or too short and creating a validation set')
remove_over_context_length_and_create_valid(file_name)