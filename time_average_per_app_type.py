import os
from time import process_time
import numpy as np
import json

input_directory = "./results/input_dir"
output_file = "application_type_properties.json"
mobile_to_edge_bandwidth = 150
mobile_to_edge_latency = 20
mobile_to_cloud_latency = 70
mobile_to_cloud_bandwidth = 36


def main():
    application_types = [
        f.name for f in os.scandir(input_directory) if f.is_dir()]

    application_results = {application_type: applicationResults(
        f"{input_directory}/{application_type}") for application_type in application_types}

    generateResult(application_results)
    return


def generateResult(application_results):
    for app_type in application_results.keys():
        application_results[app_type]["mobile_over_edge"] = application_results[app_type]["process_time_mobile"] / application_results[app_type]["process_time_edge"]
        
        application_results[app_type]["average_edge_input_time"] = application_results[app_type]["input_size"] / mobile_to_edge_bandwidth
        application_results[app_type]["average_edge_output_time"] = application_results[app_type]["output_size"] / mobile_to_edge_bandwidth
        
        application_results[app_type]["average_cloud_input_time"] = application_results[app_type]["input_size"] / mobile_to_cloud_bandwidth
        application_results[app_type]["average_cloud_output_time"] = application_results[app_type]["output_size"] / mobile_to_cloud_bandwidth

        application_results[app_type]["input_chaining_edge_time"] = application_results[app_type]["average_edge_input_time"] + application_results[app_type]["process_time_edge"] + (mobile_to_edge_latency / 1000)
        application_results[app_type]["reactive_edge_time"] = application_results[app_type]["average_edge_output_time"] + application_results[app_type]["process_time_edge"] + application_results[app_type]["average_edge_input_time"] + ((mobile_to_edge_latency * 2) / 1000)
        application_results[app_type]["mobile_over_reactive_edge"] = application_results[app_type]["process_time_mobile"] / application_results[app_type]["reactive_edge_time"]
        application_results[app_type]["mobile_over_input_chaining_edge"] = application_results[app_type]["process_time_mobile"] / application_results[app_type]["input_chaining_edge_time"]

        application_results[app_type]["input_chaining_cloud_time"] = application_results[app_type]["average_cloud_input_time"] + application_results[app_type]["process_time_cloud"] + (mobile_to_cloud_latency / 1000)
        application_results[app_type]["reactive_cloud_time"] = application_results[app_type]["average_cloud_output_time"] + application_results[app_type]["process_time_cloud"] + application_results[app_type]["average_cloud_input_time"] + ((mobile_to_cloud_latency * 2) / 1000)
        application_results[app_type]["mobile_over_reactive_cloud"] = application_results[app_type]["process_time_mobile"] / application_results[app_type]["reactive_cloud_time"]
        application_results[app_type]["mobile_over_input_chaining_cloud"] = application_results[app_type]["process_time_mobile"] / application_results[app_type]["input_chaining_cloud_time"]

    f = open(f"./raw_{output_file}", "w")
    json.dump(application_results, f)
    
    for app_type in application_results.keys():
        application_results[app_type]["mobile_over_reactive_edge"] = application_results[app_type]["mobile_over_reactive_edge"] * 100
        application_results[app_type]["mobile_over_input_chaining_edge"] = application_results[app_type]["mobile_over_input_chaining_edge"] * 100

        application_results[app_type]["mobile_over_reactive_cloud"] = application_results[app_type]["mobile_over_reactive_cloud"] * 100
        application_results[app_type]["mobile_over_input_chaining_cloud"] = application_results[app_type]["mobile_over_input_chaining_cloud"] * 100

        for k in application_results[app_type].keys():
            application_results[app_type][k] = round(application_results[app_type][k], 4)


    j = open(f"./{output_file}", "w")
    json.dump(application_results, j)
    return


def applicationResults(app_dir):
    set_sizes = [f.name for f in os.scandir(app_dir) if f.is_dir()]

    set_results = [setResults(
        f"{app_dir}/{set_size}") for set_size in set_sizes]

    process_time_cloud = []
    process_time_edge = []
    process_time_mobile = []
    input_size = []
    output_size = []
    task_count = 0

    for instance in set_results:
        process_time_cloud.append(instance["process_time_cloud"])
        process_time_edge.append(instance["process_time_edge"])
        process_time_mobile.append(instance["process_time_mobile"])
        input_size.append(instance["input_size"])
        output_size.append(instance["output_size"])
        task_count = instance["task_count"] + task_count

    return {
        "process_time_cloud": np.mean(process_time_cloud),
        "process_time_edge": np.mean(process_time_edge),
        "process_time_mobile": np.mean(process_time_mobile),
        "input_size": np.mean(input_size),
        "output_size": np.mean(output_size),
        "task_count": task_count
    }


def setResults(set_dir):
    instance_results = {}
    instance_names = [f.name for f in os.scandir(set_dir) if f.is_file()]

    instance_results = [instanceResults(
        f"{set_dir}/{instance}") for instance in instance_names]

    process_time_cloud = []
    process_time_edge = []
    process_time_mobile = []
    input_size = []
    output_size = []
    task_count = 0

    for instance in instance_results:
        process_time_cloud.append(instance["process_time_cloud"])
        process_time_edge.append(instance["process_time_edge"])
        process_time_mobile.append(instance["process_time_mobile"])
        input_size.append(instance["input_size"])
        output_size.append(instance["output_size"])
        task_count = instance["task_count"] + task_count

    return {
        "process_time_cloud": np.mean(process_time_cloud),
        "process_time_edge": np.mean(process_time_edge),
        "process_time_mobile": np.mean(process_time_mobile),
        "input_size": np.mean(input_size),
        "output_size": np.mean(output_size),
        "task_count": task_count
    }


def instanceResults(instance_dir):
    f = open(f"{instance_dir}", "r")
    instance_results = [line.strip() for line in f.readlines()]
    f.close()

    application_count = int(instance_results[1])
    instance_results = instance_results[2: len(instance_results)]

    process_time_cloud = []
    process_time_edge = []
    process_time_mobile = []
    input_size = []
    output_size = []

    for i in range(0, application_count):
        for x in range(2, int(instance_results[0]), 2):
            task = instance_results[x].split(" ")
            if(float(task[1]) == 0 or float(task[2]) == 0 or float(task[3]) == 0 or float(task[5]) == 0 or float(task[6]) == 0):
                continue
            process_time_cloud.append(float(task[1]))
            process_time_edge.append(float(task[2]))
            process_time_mobile.append(float(task[3]))
            input_size.append(float(task[5]))
            output_size.append(float(task[6]))
        instance_results = instance_results[(2 + (int(instance_results[0]) * 2)): len(instance_results)]

    return {
        "process_time_cloud": np.mean(process_time_cloud),
        "process_time_edge": np.mean(process_time_edge),
        "process_time_mobile": np.mean(process_time_mobile),
        "input_size": np.mean(input_size),
        "output_size": np.mean(output_size),
        "task_count": len(process_time_cloud)
    }


if __name__ == "__main__":
    main()
