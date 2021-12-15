import subprocess
import os
from os import path

input_dir = "./output_dir/"
temp_dir = "./input_dir"
input_script = "./input_gen.py"
lower_bound_folder = "./lower_bound_dir"
script_folder = "./scripts"
batch_size = 20
application_count = 20
regen_input = True
run_sim = False

MOBILE_GPU = "ADRENO_640"
EDGE_GPU = "RTX_3060"
CLOUD_GPU = "RTX_3060"

def main():
    if not path.isdir(input_dir):
        os.mkdir(input_dir)

    if not path.isdir(temp_dir):
        os.mkdir(temp_dir)

    if not path.isdir(lower_bound_folder):
        os.mkdir(lower_bound_folder)

    first_level_subdirectory = [
        f.name for f in os.scandir(script_folder) if f.is_file()]
    
    if regen_input == True:
        #CREATING INPUT FILES
        for i in range(1, application_count + 1):
            application_batch = f"{temp_dir}/{i}"
            if not path.isdir(application_batch):
                os.mkdir(application_batch)

            for x in range(1, batch_size + 1):
                application_topology = f"{application_batch}/application_topology_batch_{i}_{x}"
                subprocess.call(["python3", input_script, f"{i}", application_topology, MOBILE_GPU, EDGE_GPU, CLOUD_GPU ])

    if run_sim == True:
        for algorithm in first_level_subdirectory:
            if algorithm.startswith('.'):   # to ignore any hidden file
                continue
            directory_name = f"{input_dir}{algorithm}/"
            if not path.isdir(directory_name):
                os.mkdir(directory_name)
            
            algorithm_program = f"{script_folder}/{algorithm}"

            run_algorithm_eval(directory_name, algorithm_program)


def run_algorithm_eval(directory_name, algorithm_program):
    for i in range(1, application_count + 1):
        lower_bound_count_dir = f"{lower_bound_folder}/{i}"
        if not os.path.isdir(lower_bound_count_dir):
            os.mkdir(lower_bound_count_dir)
        run_output(i, f"{directory_name}{i}/", batch_size, input_script, algorithm_program, lower_bound_count_dir)


def run_output(count, directory, batch_size, input_script, allocation_program, lower_bound_count_dir):
    if not path.isdir(directory):
            os.mkdir(directory)
    
    for i in range(1, batch_size + 1):
        instance_dir = f"{directory}{i}/"
        if not path.isdir(instance_dir):
            os.mkdir(instance_dir)
        application_topology = f"{temp_dir}/{count}/application_topology_batch_{count}_{i}"
        lower_bound_file = f"{lower_bound_count_dir}/lower_bound_vals_{count}_{i}"
        output_topology = f"{instance_dir}algorithm_output_{count}_{i}"

        subprocess.call([allocation_program, application_topology, output_topology, lower_bound_file])

    return


if __name__ == "__main__":
    main()