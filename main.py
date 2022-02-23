import os
from re import I
import sys
import json
from os import path
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
DEBUG_MODE= ""

column_width = 0.16

meta_folder = "./results/"
algorithm_folder = f"{meta_folder}output_dir"
graph_folder = f"{meta_folder}graphs"
script_name = "algorithm_output"

edge_node_count = 3
topology = {
    'mobile': {
        "cpu": 1,
        "ram": 8
    },
    "edge": {
        "cpu": 2 * edge_node_count,
        "ram": 64 * edge_node_count},
    "cloud": {
        "cpu": sys.maxsize,
        "ram": sys.maxsize
    }
}


def main():
    global DEBUG_MODE
    application_types = [
        f.name for f in os.scandir(algorithm_folder) if f.is_dir()]

    if not path.isdir(graph_folder):
        os.mkdir(graph_folder)

    for application_type in application_types:
        DEBUG_MODE = application_type
        generate_graphs(application_type)

    return


def generate_graphs(application_type):
    simulator_result = fetchAlgorithmResults(application_type)
    agg_vals = aggregate_values(simulator_result)

    folder_path = f"{graph_folder}/{application_type}"
    if not path.isdir(folder_path):
        os.mkdir(folder_path)

    graph_app_completion_rate(
        agg_vals['app_completion_rate_per_app_size'], folder_path)
    graph_task_completion_rate_by_node_type(
        agg_vals['task_completion_rate_per_app_size'], folder_path)
    graph_communication_v_computation_per_algo(agg_vals['communication_computation'], folder_path)
    return


def graph_communication_v_computation_per_algo(app_task_rate, folder_path):
    fig, ax = plt.subplots()
    ntypes = sorted(topology.keys())

    transformed_vals = {"Communication": {}, "Computation": {}}

    for algorithm, meta_values in app_task_rate.items():
        comm_mean = np.mean(app_task_rate[algorithm]["comm"])
        comp_mean = np.mean(app_task_rate[algorithm]["comp"])
        total = comm_mean + comp_mean
        transformed_vals["Communication"][algorithm] = comm_mean / total
        transformed_vals["Computation"][algorithm] = comp_mean / total

    x_pos = []
    y_values = []
    width_val = column_width
    counter = 0
    for times_type, vol in transformed_vals.items():
        # if len(x_pos) == 0:
        x_pos = np.arange(len(app_task_rate.keys()))
        # else:
        #     x_pos = [i + width_val for i in x_pos]

        if DEBUG_MODE != 'generic':
            print()
        
        x_vals = list(vol.values())

        if counter == 0:
            y_values = x_vals
            ax.bar(x_pos, x_vals, width=width_val, label=times_type)
            
        elif counter == 1:
            ax.bar(x_pos, x_vals, bottom=y_values, width=width_val, label=times_type)
            y_values = [y_values[i] + x_vals[i] for i in range(0, len(y_values))]
        else:
            ax.bar(x_pos, x_vals, bottom=y_values, width=width_val, label=times_type)
            y_values = [y_values[i] + x_vals[i] for i in range(0, len(y_values))]
        counter = counter + 1

    ax.set_ylabel('Communication and Computational Time Ratio')

    legends = list(transformed_vals.keys())

    plt.ylim([0, 1.3])
    plt.legend(legends, prop={'size': 6}, loc='upper center', ncol=4)
    ax.set_xticks(np.arange(len(x_pos)))
    ax.set_xticklabels(list(app_task_rate.keys()))
    ax.set_title(
        f'Communication and Computational Time Ratio per Algorithm')
    ax.yaxis.grid(True)
    plt.savefig(
        f"{folder_path}/communication_computation_time_ratio.pdf")
    plt.close()
    return


def graph_task_completion_rate_by_node_type(app_task_rate, folder_path):
    task_node_meta_value = {ky: {} for ky in app_task_rate.keys()}

    for algo, vals in app_task_rate.items():
        sorted_vals = dict(sorted(vals.items(), key=lambda x: int(x[0])))
        for ky, vl in sorted_vals.items():
            task_node_meta_value[algo][ky] = {
                node: [] for node in topology.keys()}
            sorted_vls = dict(sorted(vl.items(), key=lambda x: int(x[0])))
            for k, v in sorted_vls.items():
                for node_key, node_item in v.items():
                    task_node_meta_value[algo][ky][node_key].append(node_item)

    fig, ax = plt.subplots()
    ntypes = sorted(topology.keys())

    indices = []
    x_pos = []
    width_val = column_width
    for algo, vol in task_node_meta_value.items():
        val = {k: v for k, v in vol.items() if int(k) == 1 or int(k)
               == len(v) or int(k) % 2 == 0}
        indices = list(val.keys())
        if len(x_pos) == 0:
            x_pos = np.arange(len(val))
        else:
            x_pos = [i + width_val for i in x_pos]
        y_values = []
        counter = 0
        algo_name = algo.replace("_algorithm", "")
        algo_name = algo_name.replace("_", " ")
        for nt in ntypes:
            new_values = [np.mean(item[nt]) for ky, item in val.items()]
            if counter == 0:
                y_values = new_values
                ax.bar(x_pos, new_values, width=width_val,
                       label=f"{algo_name} {nt}")
            elif counter == 1:
                ax.bar(x_pos, new_values, bottom=y_values,
                       width=width_val, label=f"{algo_name} {nt}")
                y_values = [y_values[i] + new_values[i]
                            for i in range(0, len(y_values))]
            else:
                ax.bar(x_pos, new_values, bottom=y_values,
                       width=width_val, label=f"{algo_name} {nt}")
                y_values = [y_values[i] + new_values[i]
                            for i in range(0, len(y_values))]
            counter = counter + 1

    ax.set_ylabel('Tasks Completed per Node Type')

    legends = []
    for algo in task_node_meta_value.keys():
        algo_name = algo.replace("_algorithm", "")
        algo_name = algo_name.replace("_", " ")
        for typ in ntypes:
            legends.append(f"{algo_name} {typ}")
    plt.ylim([0, 1.3])
    plt.legend(legends, prop={'size': 6}, loc='upper center', ncol=4)
    ax.set_xticks(np.arange(len(x_pos)))
    ax.set_xticklabels([i for i in indices])
    ax.set_title(
        f'Tasks Completed per Node Type in batch of {indices[len(indices) - 1]} Applications')
    ax.yaxis.grid(True)
    plt.savefig(
        f"{folder_path}/tasks_completed_per_node_type_set_size_{indices[len(indices) - 1]}.pdf")
    plt.close()
    return


def graph_app_completion_rate(app_comp_rate, folder_path):
    width_val = column_width
    fig, ax = plt.subplots()

    x_pos = []
    for algorithm, meta_values in app_comp_rate.items():
        sorted_vals = {k: v for k, v in dict(
            sorted(meta_values.items(), key=lambda x: int(x[0]))).items() if (int(k) == 1 or int(k) == len(meta_values) or int(k) % 2 == 0)}
        application_completion_list = [item for item in sorted_vals.values()]
        application_completion_mean = [
            np.mean(item) for item in application_completion_list]
        application_completion_std = [
            np.std(item) for item in application_completion_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(application_completion_mean))
        else:
            x_pos = [i + width_val for i in x_pos]

        ax.bar(x_pos, application_completion_mean, yerr=application_completion_std, capsize=0.20, width=width_val,
               label=algorithm.replace('_', ' ').capitalize())
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [i for i in sorted_vals.keys()])

    ax.set_ylabel('Percentage of Applications Completed')
    ax.set_title(f'Mean Application Completion')
    ax.yaxis.grid(True)
    plt.ylim([0, 1.2])
    plt.legend([key.replace('_', ' ').capitalize()
               for key in app_comp_rate.keys()], loc="upper center", ncol=3)
    plt.savefig(f"{folder_path}/mean_application_completion.pdf")
    plt.close()
    return


def aggregate_values(simulator_result):
    res = {}
    res['communication_computation'] = {}
    res["app_completion_rate_per_app_size"] = {
        algorithm: {set: [instance['completed_application_count'] / instance['application_count'] for instance in set_data.values()] for set, set_data in data.items()} for algorithm, data in simulator_result.items()}
    res['task_completion_rate_per_app_size'] = {}
    for algorithm, data in simulator_result.items():
        res['task_completion_rate_per_app_size'][algorithm] = {}
        res['communication_computation'][algorithm] = {"comm": [], "comp": []}
        for set, set_data in data.items():
            res['task_completion_rate_per_app_size'][algorithm][set] = {}
            for instance, instance_data in set_data.items():
                res['task_completion_rate_per_app_size'][algorithm][set][instance] = {
                    node: 0 for node in topology.keys()}
                for task in instance_data['completed_tasks']:
                    comm_comp = generateCommunicationComputationValue(task)
                    res['communication_computation'][algorithm]["comm"].append( comm_comp[0] )
                    res['communication_computation'][algorithm]["comp"].append( comm_comp[1] )
                    # res['task_completion_rate_per_app_size'][algorithm][set][instance]["waiting_time"][task['task_name']] = calculateWaitingTime(task, instance_data['completed_tasks'])
                    res['task_completion_rate_per_app_size'][algorithm][set][instance][task['chosen_node_type']
                                                                                       ] = res['task_completion_rate_per_app_size'][algorithm][set][instance][task['chosen_node_type']
                                                                                                                                                              ] + 1

                for node in topology.keys():
                    res['task_completion_rate_per_app_size'][algorithm][set][instance][node] = res[
                        'task_completion_rate_per_app_size'][algorithm][set][instance][node] / instance_data['total_task_count']
    return res

def generateCommunicationComputationValue(task):
    i_finish = task['input_upload_finish_time']
    i_start = task['input_upload_start_time'] if task['input_upload_start_time'] <= task['input_upload_finish_time'] else task['input_upload_finish_time']

    o_finish = task['output_upload_finish_time']
    o_start = task['output_upload_start_time'] if task['output_upload_start_time'] <= task['output_upload_finish_time'] else task['output_upload_finish_time']
    
    comm = ((i_finish - i_start) + (o_finish - o_start))
    comp = (task['process_finish_time'] - task['process_start_time'])

    return (comm, comp)

def calculateWaitingTime(task, task_list):
    max_parent_finish_time = -1

    for node in task_list:
        if node['task_name'] in task['parents']:
            finish_time = node['output_upload_finish_time'] if node['output_upload_finish_time'] != 0 else node['process_finish_time']
            if max_parent_finish_time < finish_time:
                max_parent_finish_time = finish_time
        if node['task_name'] == task['task_name']:
            break

    return


def fetchAlgorithmResults(application_type):
    res = {}
    results_directory = f"{algorithm_folder}/{application_type}"

    algorithm_types = [f.name for f in os.scandir(
        results_directory) if f.is_dir()]

    for algorithm_type in algorithm_types:
        res[algorithm_type] = fetchSetData(
            f"{results_directory}/{algorithm_type}")

    return res


def fetchSetData(set_directory):
    res = {}
    set_info = [f.name for f in os.scandir(set_directory) if f.is_dir()]

    for set in set_info:
        res[set] = fetchInstanceData(f"{set_directory}/{set}", set)
    return res


def fetchInstanceData(instance_directory, set):
    res = {}

    instance_info = [f.name for f in os.scandir(
        instance_directory) if f.is_dir()]

    for instance in instance_info:
        f = open(
            f'{instance_directory}/{instance}/{script_name}_{set}_{instance}.json', 'r')
        res[instance] = json.load(f)

    return res


if __name__ == "__main__":
    main()
