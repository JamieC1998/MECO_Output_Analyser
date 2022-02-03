import sys
import json
import os

output_dir = "output_dir"
result_file = "violations.json"
nodes = {
    0: sys.maxsize,
    1: 2,
    2: 1,
    3: 2,
    4: 2
}


def main():
    first_level_subdirectory = [
        f.name for f in os.scandir(output_dir) if f.is_dir()]

    result = { k: iterate_through_folders(f'{output_dir}/{k}') for k in first_level_subdirectory }
    dump_violations(result)
    return


def iterate_through_folders(folder):
    app_sets = [
        f.name for f in os.scandir(folder) if f.is_dir()]
    res = {k: iterate_through_app_sets(f'{folder}/{k}') for k in app_sets}
    return res

def iterate_through_app_sets(folder):
    instances = [f.name for f in os.scandir(folder) if f.is_dir()]
    res = {k: get_file(f'{folder}/{k}') for k in instances}
    return res

def get_file(path):
    file = [f.name for f in os.scandir(path) if f.is_file()][0]
    return file_to_check(f'{path}/{file}')

def file_to_check(path):
    rf = read_file(path)
    rf.append('====================\n')
    parsed_file = parseFile(rf[3: len(rf)])
    resource_violations = {k: checkForResourceViolations(
        k, v) for k, v in parsed_file.items()}

    return resource_violations


def dump_violations(violations):
    with open(result_file, "w") as f:
        json.dump(violations, f)


def checkForResourceViolations(node, node_list):
    resource_events = []
    for task in node_list:
        resource_events.append(
            {"task_name": task["task_name"], "time": task["start_time"], "type": "increase"})
        resource_events.append(
            {"task_name": task["task_name"], "time": task["finish_time"], "type": "decrease"})

    resource_events.sort(key=lambda x: x["time"])
    capacity = nodes[node]
    violations = []
    current_tasks = []
    current_usage = 0
    for time_event in resource_events:
        if time_event["type"] == "increase":
            current_tasks.append(time_event["task_name"])
            current_usage += 1
            if current_usage > capacity:
                violations.append(
                    {"tasks": [name for name in current_tasks], "time": time_event["time"]})
        else:
            current_tasks.remove(time_event["task_name"])
            current_usage -= 1
    return violations


def parseFile(file):
    res = {k: [] for k in nodes.keys()}
    counter = 0

    while len(file) > 0:
        task_name = file[1].split(' ')[2].strip()
        vertex_type = file[14].split(" ")[2].strip()

        offset = 0
        if vertex_type == 'mobile':
            offset = 2
        elif vertex_type == 'edge':
            offset = 1

        node_id = int(file[16 + offset].split(' ')[2].strip())
        start_time = float(file[18 + offset].split(' ')[2].strip())
        finish_time = float(file[19 + offset].split(' ')[2].strip())
        upload_start = float(file[21 + offset].split(' ')[3].strip())
        upload_finish = float(file[22 + offset].split(' ')[3].strip())

        res[node_id].append({"task_name": task_name, "start_time": start_time,
                            "finish_time": finish_time, "upload_start": upload_start, "upload_finish": upload_finish})

        file = file[24 + offset:]

    return res


def read_file(path):
    res = []
    counter = 0
    with open(path, "r") as f:
        res = f.readlines()
    return res


if __name__ == "__main__":
    main()
