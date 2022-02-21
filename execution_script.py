import subprocess
import os
from os import path

meta_folder = "./results"
input_dir = f"{meta_folder}/output_dir/"
temp_dir = f"{meta_folder}/input_dir"
input_script = "./input_gen.py"
algorithm_program = "./script/algorithm_simulator"
batch_size = 20
application_count = 20
regen_input = True
run_sim = False
algorithm_modes = ["reactive_basic", "reactive_mobile", "preallocation", "partition", "proactive"]
application_types = {
    "dnn": 1,
    "generic": 0
}

sim_times = {
    "dnn": 4,
    "generic": 10
}

offload_probability = [0.1, 0.9]

USE_DNNS = 1

MOBILE_GPU = "ADRENO_640"
EDGE_GPU = "RTX_3060"
CLOUD_GPU = "RTX_3060"

def main():
    if not path.isdir(input_dir):
        os.mkdir(input_dir)

    for k in application_types.keys():
        if not path.isdir(f"{input_dir}/{k}"):
            os.mkdir(f"{input_dir}/{k}")

    if not path.isdir(temp_dir):
        os.mkdir(temp_dir)

    for k in application_types.keys():
        if not path.isdir(f"{temp_dir}/{k}"):
            os.mkdir(f"{temp_dir}/{k}")

    if not path.isdir(meta_folder):
        os.mkdir(meta_folder)
    
    if regen_input == True:
        #CREATING INPUT FILES
        for key in application_types.keys():
            for i in range(1, application_count + 1):
                application_batch = f"{temp_dir}/{key}/{i}"
                if not path.isdir(application_batch):
                    os.mkdir(application_batch)

                for x in range(1, batch_size + 1):
                    application_topology = f"{application_batch}/application_topology_batch_{i}_{x}"
                    subprocess.call(["python3", input_script, f"{i}", application_topology, f'{application_types[key]}', MOBILE_GPU, EDGE_GPU, CLOUD_GPU, f'{sim_times[key]}' ])

    if run_sim == True:
        for key in application_types.keys():
            for algorithm_mode in algorithm_modes:
                directory_name = f"{input_dir}/{key}/{algorithm_mode}/"
                if not path.isdir(directory_name):
                    os.mkdir(directory_name)

                run_algorithm_eval(directory_name, algorithm_mode, algorithm_program, key)


def run_algorithm_eval(directory_name, algorithm_mode, algorithm_program, application_type):
    for i in range(1, application_count + 1):
        run_output(i, f"{directory_name}{i}/", batch_size, algorithm_program, algorithm_mode, application_type)


def run_output(count, directory, batch_size, allocation_program, algorithm_mode, application_type):
    if not path.isdir(directory):
            os.mkdir(directory)
    
    for i in range(1, batch_size + 1):
        instance_dir = f"{directory}{i}/"
        if not path.isdir(instance_dir):
            os.mkdir(instance_dir)
        application_topology = f"{temp_dir}/{application_type}/{count}/application_topology_batch_{count}_{i}"
        output_topology = f"{instance_dir}algorithm_output_{count}_{i}.json"

        if os.path.isfile(output_topology):
            print(f"{algorithm_mode} {count} {i} exists, skipping")
            continue

        print(f"{algorithm_mode} {count} {i}")
        subprocess.call([allocation_program, application_topology, output_topology, algorithm_mode])

    return


if __name__ == "__main__":
    main()