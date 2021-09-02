import subprocess
import os
from os import path

input_dir = "./input_dir/"
input_script = "./input_gen.py"
script_folder = "./scripts"
batch_size = 20
application_count = 5

def main():
    if not path.isdir(input_dir):
        os.mkdir(input_dir)

    first_level_subdirectory = [
        f.name for f in os.scandir(script_folder) if f.is_file()]
    
    for algorithm in first_level_subdirectory:
        directory_name = f"{input_dir}{algorithm}/"
        if not path.isdir(directory_name):
            os.mkdir(directory_name)
        
        algorithm_program = f"{script_folder}/{algorithm}"

        run_algorithm_eval(directory_name, algorithm_program)


def run_algorithm_eval(directory_name, algorithm_program):
    for i in range(1, application_count + 1):
        run_scripts(i, f"{directory_name}{i}/", batch_size, input_script, algorithm_program)


def run_scripts(count, directory, batch_size, input_script, allocation_program):
    if not path.isdir(directory):
            os.mkdir(directory)
    
    for i in range(1, batch_size + 1):
        instance_dir = f"{directory}{i}/"
        if not path.isdir(instance_dir):
            os.mkdir(instance_dir)
        application_topology = f"{instance_dir}application_topology_batch_{count}_{i}"
        output_topology = f"{instance_dir}algorithm_output_{count}_{i}"
        subprocess.call(["python3", input_script, f"{count}", application_topology])
        subprocess.call([allocation_program, application_topology, output_topology])

    return


if __name__ == "__main__":
    main()