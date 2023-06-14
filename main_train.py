from scripts.train import train
import sys



if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    print("\n NOTE: Write the name of the .yaml file containing the training specs!! Example: !python main_train.py train")

print("\n The file you are testing is: {}.yaml".format(file_name))
    
train(file_name)