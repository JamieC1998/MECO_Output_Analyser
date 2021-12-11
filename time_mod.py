import os
from os import path

input_directory = "./input_dir"
time_val = 20

def main():
    first_level_directory = [f.name for f in os.scandir(input_directory) if f.is_dir()]
    for first_level_dir in first_level_directory:
        file_list = [f.name for f in os.scandir(f"{input_directory}/{first_level_dir}") if f.is_file()]
        for input_files in file_list:
            lines = []
            with open(f"{input_directory}/{first_level_dir}/{input_files}", "r") as f:
                lines = f.readlines()
            lines[0] = f"{time_val}\n"
            with open(f"{input_directory}/{first_level_dir}/{input_files}", "w") as f:
                f.writelines(lines)
            
    return

if __name__ == "__main__":
    main()