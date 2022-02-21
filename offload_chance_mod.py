from hashlib import new
import os
from random import random


input_dir = "./input_dir"
offload_chance = [0.3, 0.7]
import random
copy_dir_name = f"{input_dir}_{int(offload_chance[0] * 100)}"


def main():
    application_types = [
        f.name for f in os.scandir(input_dir) if f.is_dir()]
    
    if not os.path.exists(copy_dir_name):
        os.mkdir(copy_dir_name)
        
    for application_type in application_types:
        modify_input_each_application_type(application_type, f"{copy_dir_name}/{application_type}", f"{input_dir}/{application_type}")


def modify_input_each_application_type(application_types, c_dir, i_dir):

    if not os.path.exists(c_dir):
        os.mkdir(c_dir)

    set_sizes = [
        f.name for f in os.scandir(i_dir) if f.is_dir()]

    for set_size in set_sizes:
        modify_input_each_set_size(f"{c_dir}/{set_size}", f"{i_dir}/{set_size}", set_size)
    return


def modify_input_each_set_size(c_dir, i_dir, set_size):
    if not os.path.exists(c_dir):
        os.mkdir(c_dir)

    instances = [f.name for f in os.scandir(i_dir) if f.is_file()]
    
    for instance in instances:
        modify_offload_chance_for_instance(f"{i_dir}/{instance}", f"{c_dir}/{instance}")
    return


def modify_offload_chance_for_instance(instance_path, copy_path):

    f = open(instance_path, "r")
    lines = [line.strip() for line in f.readlines()]
    f.close()
    
    offset = 2
    app_count = int(lines[1])
    new_arr = lines[0:offset]
    lines = lines[offset:]
    
    for i in range(0, app_count):
        task_count = lines[0]
        new_arr = new_arr + lines[0:4]
        lines = lines[4:]
        task_range = (int(task_count) - 2) * 2
        for j in range(0, task_range, 2):
            new_arr.append(lines[j][: len(lines[j]) - 1] + str(random.choices([0, 1], offload_chance)[0]))
            new_arr.append(lines[j + 1])
        new_arr = new_arr + lines[task_range: task_range + 2]
        lines = lines[task_range + 2:]
        
    f = open(copy_path, "w")
    f.writelines([line + "\n" for line in new_arr])
    f.close()
    return


if __name__ == "__main__":
    main()