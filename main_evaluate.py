from scripts.load_and_run_experiment import load_and_run
import sys



if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    print("\nNOTE: Write the name of the .yaml file containing the experiment's specs!! Example: !python main_evaluation.py experiment")

print("\nThe file you are testing is: {}.yaml".format(file_name))

random_seed = 200
load_and_run(file_name,random_seed)