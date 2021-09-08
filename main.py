import json
import os
import numpy as np
import matplotlib.pyplot as plt

output_folder = "./outputs"

def generate_meta_values(input_directory, algorithm, algorithm_top):
    first_level_subdirectory = [
        f.name for f in os.scandir(input_directory) if f.is_dir()]
    first_level_subdirectory.sort(key=lambda x: int(x))

    root_results = {}
    for directory in first_level_subdirectory:
        second_level_directory_path = f"{input_directory}/{directory}"
        second_level_directories = [f.name for f in os.scandir(
            second_level_directory_path) if f.is_dir()]
        second_level_directories.sort(key=lambda x: int(x))

        group_results = {}

        for subdirectory in second_level_directories:
            subdirectory_path = f"{second_level_directory_path}/{subdirectory}"
            simulator_results = f"{subdirectory_path}/algorithm_output_{directory}_{subdirectory}"
            application_topology = f"{algorithm_top}/{directory}/application_topology_batch_{directory}_{subdirectory}"
            result = generate_result(application_topology, simulator_results)
            group_results[subdirectory] = result
        root_results[directory] = group_results

    return generate_meta_analysis(root_results, input_directory, algorithm)


def generate_graphs(algorithm_meta_values):
    graph_time_taken(algorithm_meta_values)
    graph_task_completion(algorithm_meta_values)
    graph_applications_completion(algorithm_meta_values)
    return


def graph_task_completion(algorithm_meta_values):
    #Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()
    
    x_pos = []
    for algorithm, meta_values in algorithm_meta_values.items():
        task_completion_list = [np.array([instance['task_completed'][0] / instance['task_completed'][1] for instance in items['raw_data'].values()]) for items in meta_values.values()]
        task_completion_mean = [np.mean(item) for item in task_completion_list]
        task_completion_std = [np.std(item) for item in task_completion_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(task_completion_mean))
        else:
            x_pos = [i+width_val for i in x_pos]

        ax.bar(x_pos, task_completion_mean, yerr=task_completion_std, capsize=10, width=width_val, label=algorithm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(task_completion_std) + 1)])

    ax.set_ylabel('Percentage of Tasks Completed')
    ax.set_title(f'Mean Task completion')
    ax.yaxis.grid(True)
    plt.legend([key for key in algorithm_meta_values.keys()], loc=4)
    plt.savefig(f"{output_folder}/mean_task_completion.pdf")


def graph_applications_completion(algorithm_meta_values):
    #Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()
    
    x_pos = []
    for algorithm, meta_values in algorithm_meta_values.items():
        application_completion_list = [np.array([instance['completed_application_ratio'][0] / instance['completed_application_ratio'][1] for instance in items['raw_data'].values()]) for items in meta_values.values()]
        application_completion_mean = [np.mean(item) for item in application_completion_list]
        application_completion_std = [np.std(item) for item in application_completion_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(application_completion_mean))
        else:
            x_pos = [i+width_val for i in x_pos]

        ax.bar(x_pos, application_completion_mean, yerr=application_completion_std, capsize=10, width=width_val, label=algorithm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(application_completion_std) + 1)])

    ax.set_ylabel('Percentage of Applications Completed')
    ax.set_title(f'Mean Application completion')
    ax.yaxis.grid(True)
    plt.legend([key for key in algorithm_meta_values.keys()], loc=4)
    plt.savefig(f"{output_folder}/mean_application_completion.pdf")


def graph_time_taken(algorithm_meta_values):
    #Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()
    
    x_pos = []
    for algorithm, meta_values in algorithm_meta_values.items():
        time_values_list = [np.array([instance['time_ratio'][0] / instance['time_ratio'][1] for instance in items['raw_data'].values()]) for items in meta_values.values()]
        time_values_mean = [np.mean(item) for item in time_values_list]
        time_values_std = [np.std(item) for item in time_values_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(time_values_mean))
        else:
            x_pos = [i+width_val for i in x_pos]

        ax.bar(x_pos, time_values_mean, yerr=time_values_std, capsize=10, width=width_val, label=algorithm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(time_values_std) + 1)])

    ax.set_ylabel('Time Taken as Percentage')
    ax.set_title(f'Mean time completion')
    ax.yaxis.grid(True)
    plt.legend([key for key in algorithm_meta_values.keys()], loc=4)
    plt.savefig(f"{output_folder}/mean_time_completion.pdf")


def generate_meta_analysis(root_results, input_directory, algorithm):
    meta_values = {}
    for key, value in root_results.items():
        time_percentage = 0
        completed_application_rate = 0
        tasks_completion_rate = 0

        meta_values[key] = {}
        meta_values[key]['raw_data'] = []
        for instance in value.values():
            time_percentage = time_percentage + \
                (instance['time_ratio'][0] / instance['time_ratio'][1])
            completed_application_rate = completed_application_rate + \
                (instance['completed_application_ratio'][0] /
                 instance['completed_application_ratio'][1])
            tasks_completion_rate = tasks_completion_rate + \
                (instance['task_completed'][0] / instance['task_completed'][1])
            meta_values[key]['raw_data'].append(instance)
        instance_count = len(value.keys())

        time_percentage = time_percentage / instance_count
        completed_application_rate = completed_application_rate / instance_count
        tasks_completion_rate = tasks_completion_rate / instance_count
        meta_values[key] = {
            'time_percentage': time_percentage,
            'tasks_completion_rate': tasks_completion_rate,
            'completed_application_rate': completed_application_rate,
            'raw_data': value
        }

    output_str = ""
    for key, items in meta_values.items():
        output_str = output_str + f"{key}\nTime Percentage: {items['time_percentage']}\nAverage Task Completion: {items['tasks_completion_rate']}\nAverage Application Completion Percentage: {items['completed_application_rate']}\n\n"
    
    with open(f"{input_directory}/{algorithm}_output_file", "w") as f:
        f.write(output_str)

    with open(f"{input_directory}/{algorithm}_output_json.json", "w") as f:
        f.write(json.dumps(meta_values, indent = 4) )
    return meta_values


def generate_result(input_file, output_file):
    input_lines = read_file(input_file)

    # output_lines = read_file("output.txt")
    # output_lines = read_file("min_mobile_prior_output.txt")
    output_lines = read_file(output_file)

    total_time = float(input_lines[0])
    number_of_applications = int(input_lines[1])
    input_lines = input_lines[2:len(input_lines)]

    input_application_list = convert_input_to_json(
        input_lines, number_of_applications)
    output_application_list = convert_output_to_json(
        output_lines[3:len(output_lines)])

    results_dict = generate_analysis(total_time, input_application_list,
                                     output_application_list)
    return results_dict


def generate_analysis(total_time, input_list, output_list):
    results_dict = dict()

    total_task_count = 0
    completed_tasks = len(output_list)
    for application in input_list:
        total_task_count = total_task_count + application['task_count']

    # print(f'Time Window: {total_time}')
    completion_time = finish_time(output_list)
    if len(output_list) == 0:
        completion_time = total_time
    # print(f'Finish Time: {completion_time}')

    results_dict['time_ratio'] = [completion_time, total_time]

    completed_application_count = applications_completed(
        input_list, output_list)

    results_dict['completed_application_ratio'] = [
        completed_application_count, len(input_list)]
    # print(f'{completed_tasks}/{total_task_count} tasks completed')

    results_dict['task_completed'] = [completed_tasks, total_task_count]

    finish_time_per_application(input_list, output_list)

    nodes = applications_per_node(input_list, output_list)

    nodes = highest_concurrent_task_per_node(nodes, output_list, input_list)
    overlapping_tasks_by_task(nodes, output_list, input_list)
    nodes = average_task_per_node(nodes)
    nodes = longest_task_per_node(nodes)
    nodes = shortest_task_per_node(nodes)
    longest_task_per_application(input_list, output_list)
    shortest_task_per_application(input_list, output_list)
    return results_dict


def finish_time(output_list):
    last_finish_time = 0
    for allocation in output_list:
        if allocation['finish_time'] > last_finish_time:
            last_finish_time = allocation['finish_time']
    return last_finish_time


def finish_time_per_application(input_list, output_list):
    finish_times = []
    for i in range(0, len(input_list)):
        finish_time = 0
        for task_a in input_list[i]['tasks']:
            for allocation in output_list:
                if task_a['name'] == allocation['task']['name']:
                    if allocation['finish_time'] > finish_time:
                        finish_time = allocation['finish_time']
        finish_times.append(finish_time)

    # print("")
    # for i in range(0, len(finish_times)):
    #     print(f"Application {i} finish time: {finish_times[i]}")
    return


def shortest_task_per_application(input_list, output_list):
    shortest_tasks = []
    for i in range(0, len(input_list)):
        shortest_task = {}
        shortest_task_duration = float('inf')
        for task_a in input_list[i]['tasks']:
            for allocation in output_list:
                if task_a['name'] == allocation['task']['name']:
                    duration = allocation['finish_time'] - \
                        allocation['start_time']
                    if duration < shortest_task_duration:
                        shortest_task = allocation
                        shortest_task_duration = duration
        shortest_tasks.append(shortest_task)

    # print("")
    # for i in range(0, len(shortest_tasks)):
    #     task = shortest_tasks[i]
    #     if task:
    #         print(f"Shortest Task for App: {i}\t{task['task']['name']}\tStart Time: {task['start_time']}\tFinish Time: {task['finish_time']}\tNode: {task['vertex']['id']}")
    return


def longest_task_per_application(input_list, output_list):
    longest_tasks = []
    for i in range(0, len(input_list)):
        longest_task = {}
        longest_task_duration = 0
        for task_a in input_list[i]['tasks']:
            for allocation in output_list:
                if task_a['name'] == allocation['task']['name']:
                    duration = allocation['finish_time'] - \
                        allocation['start_time']
                    if duration > longest_task_duration:
                        longest_task = allocation
                        longest_task_duration = duration
        longest_tasks.append(longest_task)

    # print("")
    # for i in range(0, len(longest_tasks)):
    #     task = longest_tasks[i]
    #     if task:
    #         print(f"Longest Task for App: {i}\t\t{task['task']['name']}\tStart Time: {task['start_time']}\tFinish Time: {task['finish_time']}\tNode: {task['vertex']['id']}")
    return


def shortest_task_per_node(nodes):
    for key in nodes.keys():
        shortest_task = {}
        shortest_duration = float('inf')

        for task in nodes[key]['tasks']:
            duration = task['finish_time'] - task['start_time']
            if shortest_duration > duration:
                shortest_task = task
                shortest_duration = duration
        nodes[key]['shortest_task'] = shortest_task

    # print("")
    # for node in nodes.values():
    #     shortest_task = node['shortest_task']
    #     print(f"Shortest Task on Node: {node['id']}\t{shortest_task['task']['name']}\tStart Time: {shortest_task['start_time']}\tFinish Time: {shortest_task['finish_time']}")

    return nodes


def longest_task_per_node(nodes):
    for key in nodes.keys():
        longest_task = {}
        longest_duration = 0

        for task in nodes[key]['tasks']:
            duration = task['finish_time'] - task['start_time']
            if longest_duration < duration:
                longest_task = task
                longest_duration = duration
        nodes[key]['longest_task'] = longest_task

    # print("")
    # for node in nodes.values():
    #     longest_task = node['longest_task']
    #     print(f"Longest Task on Node: {node['id']}\t\t{longest_task['task']['name']}\tStart Time: {longest_task['start_time']}\tFinish Time: {longest_task['finish_time']}")
    return nodes


def average_task_per_node(nodes):
    for key in nodes.keys():
        cores = 0
        mips = 0
        storage = 0
        ram = 0
        duration = 0
        for task in nodes[key]['tasks']:
            cores = cores + task['task']['cores']
            mips = mips + task['task']['mi']
            ram = ram + task['task']['ram']
            storage = storage + task['task']['storage']
            duration = duration + (task['finish_time'] - task['start_time'])
        task_list_len = len(nodes[key]['tasks'])
        cores = cores / task_list_len
        mips = mips / task_list_len
        storage = storage / task_list_len
        ram = ram / task_list_len
        duration = duration / task_list_len

        nodes[key]['average_task'] = {
            'cores': cores,
            'mips': mips,
            'storage': storage,
            'ram': ram,
            'duration': duration
        }

    # for node in nodes.values():
    #     task = node['average_task']
    #     print(f'Average Task for Node: {node["id"]}\tCores: {task["cores"]}\tMI: {task["mips"]} \tStorage: {round(task["storage"], 3)}\tRAM: {round(task["ram"], 2)}\tDuration: {task["duration"]}')

    return nodes


def overlapping_tasks_by_task(nodes, output_list, input_list):
    # print("\nOverlapping tasks:")
    # for node in nodes.values():
    #     for task in node['tasks']:
    #         print(f'\t{task["task"]["name"]}\tStart Time: {task["start_time"]}\tFinish Time: {task["finish_time"]}')
    #         for item in task['task']['overlapping_tasks']:
    #             print(f'\t\t{item["task"]["name"]}\tStart Time: {item["start_time"]}\tFinish Time: {item["finish_time"]}')
    #         print("")
    return


def highest_concurrent_task_per_node(nodes, output_list, input_list):
    for node in nodes.values():
        task_list_length = len(node['tasks'])
        for i in range(0, task_list_length):
            current_task = node['tasks'][i]
            for x in range(0, task_list_length):
                if i == x:
                    continue
                compare_task = node['tasks'][x]

                if current_task['start_time'] <= compare_task['finish_time'] and compare_task['start_time'] <= current_task['finish_time']:
                    node['tasks'][i]['task']['overlapping_tasks'].append(
                        node['tasks'][x])

    # for key in nodes.keys():
    #     for x in range(0, len(nodes[key]['tasks'])):
    #         task = calculate_highest_parrallel_tasks(nodes[key]['tasks'][x])
    #         nodes[key]['tasks'][x] = task

    return nodes


def calculate_highest_parrallel_tasks(task):
    finished = False

    overlapping_windows = []
    for task_item in task['task']['overlapping_tasks']:
        names = {task['task']['name'], task_item['task']['name']}
        start_time = task['start_time']
        finish_time = task['finish_time']

        if(task_item['start_time'] > start_time):
            start_time = task_item['start_time']
        if(task_item['finish_time'] < finish_time):
            finish_time = task_item['finish_time']

        overlapping_windows.append([names, [start_time, finish_time]])

    while not finished:
        new_windows = []
        for i in range(0, len(overlapping_windows)):
            for x in range(0, len(overlapping_windows)):
                if i == x:
                    continue
                start_time_a = overlapping_windows[i][1][0]
                finish_time_a = overlapping_windows[i][1][1]

                start_time_b = overlapping_windows[x][1][0]
                finish_time_b = overlapping_windows[x][1][1]

                if start_time_a <= finish_time_b and start_time_b <= finish_time_a:
                    start_time = start_time_a
                    finish_time = finish_time_a

                    names = overlapping_windows[i][0].union(
                        overlapping_windows[x][0])

                    if(start_time_b > start_time):
                        start_time = start_time_b
                    if(finish_time_b < finish_time):
                        finish_time = finish_time_b

                    new_windows.append([names, [start_time, finish_time]])
        if len(new_windows) != 0:
            overlapping_windows = new_windows
        else:
            finished = True

    task['overlapping_windows'] = overlapping_windows

    return task


def applications_per_node(input_list, output_list):
    nodes = {}

    for allocation in output_list:
        ky = allocation['vertex']['id']
        nodes[ky] = allocation['vertex']
        nodes[ky]['task_count'] = 0
        nodes[ky]['tasks'] = []
        nodes[ky]['highest_concurrent_task'] = 0
        nodes[ky]['highest_concurrent_task_list'] = []

    for output in output_list:
        ky = output['vertex']['id']
        nodes[ky]['task_count'] = nodes[ky]['task_count'] + 1
        temp_task = output['task']
        temp_task['overlapping_tasks'] = []
        nodes[ky]['tasks'].append({
            'start_time': output['start_time'],
            'finish_time': output['finish_time'],
            'task': temp_task
        })

    # print("")
    # for node in nodes.values():
    #     print(f"Node: {node['id']} \tType: {node['type']} \tTotal tasks processed: {node['task_count']}")
    return nodes


def applications_completed(input_list, output_list):
    output_list_names = list(map(lambda x: x['task']['name'], output_list))

    applications_completed_values = [True] * len(input_list)

    for i in range(0, len(input_list)):
        for task in input_list[i]['tasks']:
            if task['name'] not in output_list_names:
                applications_completed_values[i] = False

    total_application_count = len(applications_completed_values)
    completed_application_count = 0

    for values in applications_completed_values:
        if values:
            completed_application_count = completed_application_count + 1

    # print(f'{completed_application_count}/{total_application_count} applications completed')
    return completed_application_count


def convert_output_to_json(output_lines):
    raw_mappings = []
    task_mapping = []
    for i in range(0, len(output_lines)):
        task_mapping.append(output_lines[i].strip())
        if(output_lines[i].strip() == "===================="):
            raw_mappings.append(task_mapping)
            task_mapping = []
    raw_mappings.append(task_mapping)
    task_mappings = []

    for item in raw_mappings:
        if len(item) == 0:
            continue

        allocation = {}
        task = {}
        vertex = {}

        task['name'] = item[1].split(" ")[1]
        task['ram'] = float(item[2].split(" ")[1])
        task['mi'] = int(item[3].split(" ")[1])
        task['cores'] = int(item[4].split(" ")[1])
        task['data_in'] = int(item[5].split(" ")[2])
        task['data_out'] = int(item[6].split(" ")[2])
        task['storage'] = float(item[7].split(" ")[1])
        task['source'] = int(item[8].split(" ")[3])
        task['offload'] = bool(int(item[9].split(" ")[2]))
        task['id'] = int(item[10].split(" ")[2])

        vertex['cores'] = int(item[13].split(" ")[1])
        vertex['mips'] = int(item[14].split(" ")[1])
        vertex['storage'] = float(item[15].split(" ")[1])
        vertex['ram'] = float(item[16].split(" ")[1])
        vertex['type'] = item[17].split(" ")[1]

        offset = 0
        if vertex['type'] == 'mobile':
            offset = 2
        elif vertex['type'] == 'edge':
            offset = 1
        vertex['id'] = item[19 + offset].split(" ")[1]
        start_time = item[21 + offset].split(" ")
        finish_time = item[22 + offset].split(" ")

        allocation['start_time'] = float(start_time[2])
        allocation['finish_time'] = float(finish_time[2])
        allocation['task'] = task
        allocation['vertex'] = vertex

        task_mappings.append(allocation)
    return task_mappings


def convert_input_to_json(input_lines, number_of_applications):
    input_application_list = []
    for i in range(0, number_of_applications):
        application_data = {}
        application_data["task_count"] = int(input_lines[0])
        application_data["offload_time"] = float(input_lines[1])
        application_data["tasks"] = []
        input_lines = input_lines[2:len(input_lines)]

        for x in range(0, application_data["task_count"]):
            task = {}
            task_raw_data = input_lines[0].split(" ")
            task["parents"] = list(map(lambda x: int(x), list(
                filter(lambda y: y != '\n', input_lines[1].split(" ")))))
            task["name"] = task_raw_data[0]
            task["mi"] = int(task_raw_data[1])
            task["ram"] = float(task_raw_data[2])
            task["data_in"] = int(task_raw_data[3])
            task["data_out"] = int(task_raw_data[4])
            task["storage"] = float(task_raw_data[5])
            task["offload"] = bool(int(task_raw_data[6]))
            task["cores"] = int(task_raw_data[7].replace('\n', ""))
            application_data["tasks"].append(task)
            input_lines = input_lines[2:len(input_lines)]

        input_application_list.append(application_data)

    return input_application_list


def read_file(filename):
    res = ""
    with open(filename, 'r') as f:
        res = f.readlines()
    return res


if __name__ == "__main__":
    output_directory = "./output_dir"
    input_directory = "./input_dir"
    first_level_subdirectory = [f.name for f in os.scandir(output_directory) if f.is_dir()]

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    algorithm_meta_values = {}
    for directory in first_level_subdirectory:
        algorithm_meta_values[directory] = generate_meta_values(f"{output_directory}/{directory}", directory, input_directory)

    generate_graphs(algorithm_meta_values)
