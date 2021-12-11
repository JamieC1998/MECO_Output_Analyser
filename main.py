import json
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
import sys

output_folder = "./outputs"
lower_bound_dir = "./lower_bound_dir"
node_types = ["cloud", "edge", "mobile"]

topology = [{"cpu": 4, "ram": 8, "type": 'mobile'},
            {"cpu": 16, "ram": 64, "type": "edge"},
            {"cpu": sys.maxsize, "ram": sys.maxsize, "type": "cloud"},
            {"cpu": 16, "ram": 64, "type": 'edge'},
            {"cpu": 16, "ram": 64, "type": 'edge'}]

algorithm_debug_name = ""
debug_app_set_size = 0
debug_instance_count = 0


def generate_meta_values(input_dir, algorithm, algorithm_top, lower_bound_values):
    global debug_app_set_size
    global debug_instance_count

    first_level = [
        f.name for f in os.scandir(input_dir) if f.is_dir()]
    first_level.sort(key=lambda x: int(x))

    root_results = {}
    for directory in first_level:
        debug_app_set_size = directory
        second_level_directory_path = f"{input_dir}/{directory}"
        second_level_directories = [f.name for f in os.scandir(
            second_level_directory_path) if f.is_dir()]
        second_level_directories.sort(key=lambda x: int(x))

        group_results = {}

        for subdirectory in second_level_directories:
            debug_instance_count = subdirectory
            subdirectory_path = f"{second_level_directory_path}/{subdirectory}"
            simulator_results = f"{subdirectory_path}/algorithm_output_{directory}_{subdirectory}"
            application_topology = f"{algorithm_top}/{directory}/application_topology_batch_{directory}_{subdirectory}"
            result = generate_result(
                application_topology, simulator_results, lower_bound_values[directory][subdirectory])
            group_results[subdirectory] = result
        root_results[directory] = group_results

    return generate_meta_analysis(root_results, input_dir, algorithm)


def read_lower_bound_vals(l_b_dir):
    res = {}

    first_level_dir = [f.name for f in os.scandir(
        l_b_dir) if f.is_dir() and not f.name.startswith(".")]

    for fst_dir in first_level_dir:
        res[fst_dir] = {}
        nested_folder = f"{l_b_dir}/{fst_dir}"
        lb_vals = sorted([f.name for f in os.scandir(nested_folder) if f.is_file(
        ) and not f.name.startswith(".")], key=lambda x: int(x.split("_")[4]))
        for i, item in enumerate(lb_vals):
            with open(f"{nested_folder}/{item}", 'r') as f:
                res[fst_dir][str(
                    i + 1)] = list(map(lambda x: float(x.strip("\n")), f.readlines()))
    return res


def generate_graphs(algorithm_agg_values):
    graph_time_taken(algorithm_agg_values)
    graph_task_completion(algorithm_agg_values)
    graph_applications_completion(algorithm_agg_values)
    graph_application_completion_times_per_app_size(algorithm_agg_values)
    graph_tasks_per_node(algorithm_agg_values)
    graph_cpu_usage_per_node_type(algorithm_agg_values)
    graph_ram_usage_per_node_type(algorithm_agg_values)
    graph_max_ram_per_node_type(algorithm_agg_values)
    graph_max_cpu_per_node_type(algorithm_agg_values)
    graph_aggregated_normalised_completion_times_by_app(algorithm_agg_values)
    graph_stacked_tasks_completed_per_node_type(algorithm_agg_values)
    return


def graph_stacked_tasks_completed_per_node_type(algorithm_agg_values):
    task_node_meta_value = {ky: {} for ky in algorithm_agg_values.keys()}

    for algo, vals in algorithm_agg_values.items():
        for ky, vl in vals.items():
            task_node_meta_value[algo][ky] = vl['tasks_completed_per_node_type']
    
    fig, ax = plt.subplots()
    ntypes = sorted(node_types, key=lambda x: x == "mobile", reverse=True)

    x_pos = []
    width_val = 0.25
    for algo, val in task_node_meta_value.items():
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
                ax.bar(x_pos, new_values, width=width_val, label=f"{algo_name} {nt}") 
            elif counter == 1:
                ax.bar(x_pos, new_values, bottom=y_values,width=width_val, label=f"{algo_name} {nt}")
                y_values = [ y_values[i] + new_values[i] for i in range(0, len(y_values))]
            else:
                ax.bar(x_pos, new_values, bottom=y_values,width=width_val, label=f"{algo_name} {nt}")
                y_values = [ y_values[i] + new_values[i] for i in range(0, len(y_values))]
            counter = counter + 1

    ax.set_ylabel('Tasks Completed per Node Type')

    legends = []
    for algo in task_node_meta_value.keys():
        algo_name = algo.replace("_algorithm", "")
        algo_name = algo_name.replace("_", " ")
        for typ in ntypes:
            legends.append(f"{algo_name} {typ}")
    plt.legend(legends, prop={'size': 6})
    ax.set_xticks(np.arange(len(x_pos)))
    ax.set_xticklabels([i for i in range(1, len(x_pos) + 1)])
    ax.set_title(
        f'Tasks Completed per Node Type in batch of {len(x_pos)} Applications')
    ax.yaxis.grid(True)
    plt.savefig(f"{output_folder}/tasks_completed_per_node_type_set_size_{len(x_pos)}.pdf")
    plt.close()
    return


def graph_aggregated_normalised_completion_times_by_app(algorithm_agg_values):
    
    normalised_app_completion_folder = f"{output_folder}/normalised_app_completion_by_set_size"
    
    if not os.path.isdir(normalised_app_completion_folder):
        os.mkdir(normalised_app_completion_folder)

    res = {}
    for algorithm, algorithm_data in algorithm_agg_values.items():
        for app_size_num, app_size_val in algorithm_data.items():
            if app_size_num not in res:
                res[app_size_num] = {}
            
            res[app_size_num][algorithm] = app_size_val['aggregated_normalised_duration_values_by_app']
            continue

    width_val = 0.25

    for app_size, app_set_data in res.items():
        fig, ax = plt.subplots()
        x_pos = []
        for algorithm, algorithm_results in app_set_data.items():
            mean_vals = []
            std_vals = []
            for algo_val in algorithm_results.values():
                mean_vals.append(np.mean(algo_val))
                std_vals.append(np.std(algo_val))
            
            if len(x_pos) == 0:
                x_pos = np.arange(len(mean_vals))
            
            else:
                x_pos = [i + width_val for i in x_pos]
            
            ax.bar(x_pos, mean_vals, yerr=std_vals, capsize=0.20, width=width_val, label=algorithm) 
            ax.set_xticks(x_pos)
            ax.set_xticklabels([i for i in range(1, len(mean_vals) + 1)])
            continue

        ax.set_ylabel('Normalised App Duration')
        ax.set_title(
            f'Individual Normalised App Completion Time for batch of {len(x_pos)} Applications')
        ax.yaxis.grid(True)
        plt.legend(app_set_data.keys(), loc=1)
        plt.savefig(f"{normalised_app_completion_folder}/normalised_app_completion_set_size_{app_size}.pdf")
        plt.close()

    agg_app_size_vals = {}

    for app_size_key, app_set_value in res.items():
        for algorithm_name, algorithm_results in app_set_value.items():
            if algorithm_name not in agg_app_size_vals:
                agg_app_size_vals[algorithm_name] = {}
            agg_app_size_vals[algorithm_name][app_size_key] = [np.mean(res_vals) for res_vals in algorithm_results.values()]

    fig, ax = plt.subplots()
    x_pos = []
    for algorithm, algorithm_results in agg_app_size_vals.items():
        mean_vals = []
        std_vals = []

        for results in algorithm_results.values():
            mean_vals.append(np.mean(results))
            std_vals.append(np.std(results))
        
        if len(x_pos) == 0:
            x_pos = np.arange(len(mean_vals))
            
        else:
            x_pos = [i + width_val for i in x_pos]
            
        ax.bar(x_pos, mean_vals, yerr=std_vals, capsize=0.20, width=width_val, label=algorithm) 
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(mean_vals) + 1)])
        

    ax.set_ylabel('Normalised App Duration')
    ax.set_title(
        f'Mean Normalised App Completion Time across {len(x_pos)} Applications')
    ax.yaxis.grid(True)
    plt.legend(agg_app_size_vals.keys(), loc=1)
    plt.savefig(f"{output_folder}/normalised_app_completion_{len(x_pos)}.pdf")
    plt.close()

    return


def graph_max_cpu_per_node_type(algorithm_agg_values):
    max_ram_per_node_type_folder = f"{output_folder}/max_cpu_usage_per_node_type"
    if not os.path.isdir(max_ram_per_node_type_folder):
        os.mkdir(max_ram_per_node_type_folder)

    max_ram_per_n_t = {ky: {nt: [] for nt in node_types}
                       for ky in algorithm_agg_values.keys()}

    for algorithm_name, algorithm_agg_value in algorithm_agg_values.items():
        for idx, instance_val in algorithm_agg_value.items():
            for node_type_key, node_type_val in instance_val["max_cpu_per_node_type"].items():
                max_ram_per_n_t[algorithm_name][node_type_key].append(
                    {"mean": np.mean(node_type_val), "std": np.std(node_type_val)})

    for algo_name, agg_vals in max_ram_per_n_t.items():
        width_val = 0.25
        fig, ax = plt.subplots()
        x_pos = []

        for node_type_name, node_type_vals in agg_vals.items():
            ram_mean_values = [item["mean"] for item in node_type_vals]
            ram_std_values = [item["std"] for item in node_type_vals]

            if len(x_pos) == 0:
                x_pos = np.arange(len(ram_mean_values))

            else:
                x_pos = [i + width_val for i in x_pos]

            ax.bar(x_pos, ram_mean_values, yerr=ram_std_values,
                   capsize=0.20, width=width_val, label=node_type_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([i for i in range(1, len(ram_mean_values) + 1)])
        ax.set_ylabel('CPU Usage')
        ax.set_title(
            f'Mean CPU Usage per Node Type with {len(ram_mean_values)} Applications')
        ax.yaxis.grid(True)
        plt.legend([name.capitalize() for name in agg_vals.keys()], loc=1)
        plt.savefig(
            f"{max_ram_per_node_type_folder}/max_cpu_use_per_node_type_{algo_name}.pdf")
        plt.close()

    return


def graph_max_ram_per_node_type(algorithm_agg_values):
    max_ram_per_node_type_folder = f"{output_folder}/max_ram_usage_per_node_type"
    if not os.path.isdir(max_ram_per_node_type_folder):
        os.mkdir(max_ram_per_node_type_folder)

    max_ram_per_n_t = {ky: {nt: [] for nt in node_types}
                       for ky in algorithm_agg_values.keys()}

    for algorithm_name, algorithm_agg_value in algorithm_agg_values.items():
        for idx, instance_val in algorithm_agg_value.items():
            for node_type_key, node_type_val in instance_val["max_ram_per_node_type"].items():
                max_ram_per_n_t[algorithm_name][node_type_key].append(
                    {"mean": np.mean(node_type_val), "std": np.std(node_type_val)})

    for algo_name, agg_vals in max_ram_per_n_t.items():
        width_val = 0.25
        fig, ax = plt.subplots()
        x_pos = []

        for node_type_name, node_type_vals in agg_vals.items():
            ram_mean_values = [item["mean"] for item in node_type_vals]
            ram_std_values = [item["std"] for item in node_type_vals]

            if len(x_pos) == 0:
                x_pos = np.arange(len(ram_mean_values))

            else:
                x_pos = [i + width_val for i in x_pos]

            ax.bar(x_pos, ram_mean_values, yerr=ram_std_values,
                   capsize=0.20, width=width_val, label=node_type_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([i for i in range(1, len(ram_mean_values) + 1)])
        ax.set_ylabel('RAM Usage')
        ax.set_title(
            f'Mean RAM Usage per Node Type with {len(ram_mean_values)} Applications')
        ax.yaxis.grid(True)
        plt.legend([name.capitalize() for name in agg_vals.keys()], loc=1)
        plt.savefig(
            f"{max_ram_per_node_type_folder}/max_ram_use_per_node_type_{algo_name}.pdf")
        plt.close()

    return


def graph_ram_usage_per_node_type(algorithm_aggr_values):
    ram_per_node_type_folder = f"{output_folder}/ram_per_node_type"
    if not os.path.isdir(ram_per_node_type_folder):
        os.mkdir(ram_per_node_type_folder)

    for algo, val in algorithm_aggr_values.items():
        ram_per_node_type_algo_folder = f"{ram_per_node_type_folder}/{algo}"
        if not os.path.isdir(ram_per_node_type_algo_folder):
            os.mkdir(ram_per_node_type_algo_folder)

        for app_batch, items in val.items():
            fig, ax = plt.subplots()

            for n_type, n_type_list in items['ram_use_per_node_type'].items():
                y_axis_mean = [np.mean(percents)
                               for percents in n_type_list.values()]
                y_axis_std = [np.std(percents)
                              for percents in n_type_list.values()]
                x_axis = np.arange(len(y_axis_mean))

                ax.errorbar(x_axis, y_axis_mean, yerr=y_axis_std, fmt='-o')

            ax.set_ylabel('Resource Use')
            ax.set_title(
                f'Weighted Sum of RAM Use {algo.replace("_", " ").capitalize()} for {app_batch} Applications')
            ax.grid(True)
            plt.legend(node_types)
            plt.savefig(
                f"{ram_per_node_type_algo_folder}/ram_use_{algo}_{app_batch}.pdf")
            plt.close()

    return


def graph_cpu_usage_per_node_type(algorithm_meta_values):
    cpu_per_node_type_folder = f"{output_folder}/cpu_per_node_type"
    if not os.path.isdir(cpu_per_node_type_folder):
        os.mkdir(cpu_per_node_type_folder)

    for algo, val in algorithm_meta_values.items():
        cpu_per_node_type_algo_folder = f"{cpu_per_node_type_folder}/{algo}"
        if not os.path.isdir(cpu_per_node_type_algo_folder):
            os.mkdir(cpu_per_node_type_algo_folder)

        for app_batch, items in val.items():
            fig, ax = plt.subplots()

            for n_type, n_type_list in items['cpu_use_per_node_type'].items():
                y_axis_mean = [np.mean(percents)
                               for percents in n_type_list.values()]
                y_axis_std = [np.std(percents)
                              for percents in n_type_list.values()]
                x_axis = np.arange(len(y_axis_mean))

                ax.errorbar(x_axis, y_axis_mean, yerr=y_axis_std, fmt='-o')

            ax.set_ylabel('Resource Use')
            ax.set_title(
                f'Weighted Sum of CPU Use {algo.replace("_", " ").capitalize()} for {app_batch} Applications')
            ax.grid(True)
            plt.legend(node_types)
            plt.savefig(
                f"{cpu_per_node_type_algo_folder}/cpu_use_{algo}_{app_batch}.pdf")
            plt.close()

    return


def graph_tasks_per_node(algo_meta_vals):
    tasks_per_node_folder = f"{output_folder}/tasks_per_node"
    if not os.path.isdir(tasks_per_node_folder):
        os.mkdir(tasks_per_node_folder)

    tasks_per_node = {}

    for key, value in algo_meta_vals.items():
        for ky, val in value.items():
            if ky not in tasks_per_node:
                tasks_per_node[ky] = {}
            if key not in tasks_per_node[ky]:
                tasks_per_node[ky][key] = {}
            for item in val['tasks_per_node']:
                for k, v in item.items():
                    if k not in tasks_per_node[ky][key]:
                        tasks_per_node[ky][key][k] = [v]
                    else:
                        tasks_per_node[ky][key][k].append(v)

    for key, value in tasks_per_node.items():
        width_val = 0.25
        fig, ax = plt.subplots()
        x_pos = []

        for ky, val in value.items():
            node_task_count_means = [np.mean(item) for item in val.values()]
            node_task_count_std = [np.mean(item) for item in val.values()]

            if len(x_pos) == 0:
                x_pos = np.arange(len(node_task_count_means))
            else:
                x_pos = [i + width_val for i in x_pos]

            ax.bar(x_pos, node_task_count_means, yerr=node_task_count_std,
                   capsize=0.20, width=width_val, label=ky)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                ['Cloud A', 'Edge', 'Mobile', 'Cloud B', 'Cloud C'])

        ax.set_ylabel('Tasks Completed')
        ax.set_title(f'Mean Tasks Completed per Node with {key} Applications')
        ax.yaxis.grid(True)
        plt.legend([name.replace('_', ' ').capitalize()
                   for name in value.keys()], loc=1)
        plt.savefig(
            f"{tasks_per_node_folder}/tasks_per_node_app_size_{key}.pdf")
        plt.close()
    return


def graph_application_completion_times_per_app_size(algorithm_meta_values):
    application_completion_time_folder = f"{output_folder}/application_completion_times"
    if not os.path.isdir(application_completion_time_folder):
        os.mkdir(application_completion_time_folder)

    application_completion_times = {}
    for key, value in algorithm_meta_values.items():
        for ky, val in value.items():
            for k, v in val['application_finish_times'].items():
                if ky not in application_completion_times:
                    application_completion_times[ky] = {}

                if key not in application_completion_times[ky]:
                    application_completion_times[ky][key] = {}
                application_completion_times[ky][key][k] = [
                    item[1] - item[0] for item in v]
                continue

    for key, value in application_completion_times.items():
        width_val = 0.25
        fig, ax = plt.subplots()
        x_pos = []

        for ky, val in value.items():
            application_completion_time_means = [
                np.mean(item) for item in val.values()]
            application_completion_std = [
                np.std(item) for item in val.values()]

            if len(x_pos) == 0:
                x_pos = np.arange(len(application_completion_time_means))
            else:
                x_pos = [i + width_val for i in x_pos]

            ax.bar(x_pos, application_completion_time_means,
                   yerr=application_completion_std, capsize=0.20, width=width_val, label=ky)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [i for i in range(1, len(application_completion_std) + 1)])

        ax.set_ylabel('Application Duration Times')
        ax.set_title(
            f'Mean Application Duration Times with {key} Applications')
        ax.yaxis.grid(True)
        plt.legend([name.replace('_', ' ').capitalize()
                   for name in value.keys()], loc=4)
        plt.savefig(
            f"{application_completion_time_folder}/application_duration_times_size_{key}.pdf")
        plt.close()

        # for algorithm, completion_values in value.values():

    # application_completion_times = {key: value for key, value in algorithm_meta_values.items()}
    return


def graph_task_completion(algorithm_meta_val):
    # Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()

    x_pos = []
    for algorithm, meta_values in algorithm_meta_val.items():
        task_completion_list = [np.array(
            [instance['task_completed'][0] / instance['task_completed'][1] for instance in items['raw_data'].values()])
            for items in meta_values.values()]
        task_completion_mean = [np.mean(item) for item in task_completion_list]
        task_completion_std = [np.std(item) for item in task_completion_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(task_completion_mean))
        else:
            x_pos = [i + width_val for i in x_pos]

        ax.bar(x_pos, task_completion_mean, yerr=task_completion_std,
               capsize=0.20, width=width_val, label=algorithm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(task_completion_std) + 1)])

    ax.set_ylabel('Percentage of Tasks Completed')
    ax.set_title(f'Mean Task Completion')
    ax.yaxis.grid(True)
    plt.legend([key.replace('_', ' ').capitalize()
               for key in algorithm_meta_val.keys()], loc=4)
    plt.savefig(f"{output_folder}/mean_task_completion.pdf")
    plt.close()


def graph_applications_completion(algorithm_meta_val):
    # Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()

    x_pos = []
    for algorithm, meta_values in algorithm_meta_val.items():
        application_completion_list = [np.array(
            [instance['completed_application_ratio'][0] / instance['completed_application_ratio'][1] for instance in
             items['raw_data'].values()]) for items in meta_values.values()]
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
            [i for i in range(1, len(application_completion_std) + 1)])

    ax.set_ylabel('Percentage of Applications Completed')
    ax.set_title(f'Mean Application Completion')
    ax.yaxis.grid(True)
    plt.legend([key.replace('_', ' ').capitalize()
               for key in algorithm_meta_val.keys()], loc=1)
    plt.savefig(f"{output_folder}/mean_application_completion.pdf")
    plt.close()


def graph_time_taken(algorithm_meta_val):
    # Getting a list of time values
    width_val = 0.25
    fig, ax = plt.subplots()

    x_pos = []
    for algorithm, meta_values in algorithm_meta_val.items():
        time_values_list = [
            np.array([instance['time_ratio'][0] / instance['time_ratio'][1]
                     for instance in items['raw_data'].values()])
            for items in meta_values.values()]
        time_values_mean = [np.mean(item) for item in time_values_list]
        time_values_std = [np.std(item) for item in time_values_list]

        if len(x_pos) == 0:
            x_pos = np.arange(len(time_values_mean))
        else:
            x_pos = [i + width_val for i in x_pos]

        ax.bar(x_pos, time_values_mean, yerr=time_values_std,
               capsize=0.20, width=width_val, label=algorithm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([i for i in range(1, len(time_values_std) + 1)])

    ax.set_ylabel('Time Taken as Percentage')
    ax.set_title(f'Mean Time Completion')
    ax.yaxis.grid(True)
    plt.legend([key.replace('_', ' ').capitalize()
               for key in algorithm_meta_val.keys()], loc=4)
    plt.savefig(f"{output_folder}/mean_time_completion.pdf")
    plt.close()


def generate_meta_analysis(root_results, input_dir, algorithm):
    meta_values = {}
    for key, value in root_results.items():
        time_percentage = 0
        completed_application_rate = 0
        tasks_completion_rate = 0
        application_finish_time = {}
        tasks_per_node = []

        max_cpu_per_node_type = {
            ky: [] for ky in node_types
        }

        max_ram_per_node_type = {
            ky: [] for ky in node_types
        }

        ram_use_per_node_type = {
            res_typ: {
                i: [] for i in range(0, int(value['1']['time_ratio'][1]))
            } for res_typ in node_types
        }

        cpu_use_per_node_type = {
            res_typ: {
                i: [] for i in range(0, int(value['1']['time_ratio'][1]))
            } for res_typ in node_types
        }
        
        tasks_completed_per_node_type = {
            ky: [] for ky in node_types
        }
        tasks_completed_per_node_type['total'] = []

        app_duration_lower_bound_norm = {

        }

        meta_values[key] = {}
        meta_values[key]['raw_data'] = []

        for idx, instance in value.items():
            time_percentage = time_percentage + \
                (instance['time_ratio'][0] / instance['time_ratio'][1])
            completed_application_rate = completed_application_rate + \
                (instance['completed_application_ratio'][0] /
                 instance['completed_application_ratio'][1])
            tasks_completion_rate = tasks_completion_rate + \
                (instance['task_completed'][0] / instance['task_completed'][1])

            for ky, val in instance['tasks_completed_per_node_type'].items():
                tasks_completed_per_node_type[ky].append(val)

            for ky, val in instance['finish_times_per_app'].items():
                if ky not in application_finish_time:
                    application_finish_time[ky] = [val]
                else:
                    application_finish_time[ky].append(val)

            for n_type, perc in instance['ram_use_per_node_type'].items():
                for index, perc_val in enumerate(perc):
                    ram_use_per_node_type[n_type][index].append(perc_val)

            for typ in node_types:
                max_cpu_per_node_type[typ].append(
                    instance["max_cpu_per_type"][typ])
                max_ram_per_node_type[typ].append(
                    instance["max_ram_per_type"][typ])

            for n_type, perc in instance['cpu_use_per_node_type'].items():
                for index, perc_val in enumerate(perc):
                    cpu_use_per_node_type[n_type][index].append(perc_val)

            tasks_per_node.append(instance['tasks_per_node'])

            for app_num, normalised_val in instance['normalised_app_time'].items():
                if app_num not in app_duration_lower_bound_norm:
                    app_duration_lower_bound_norm[app_num] = [normalised_val]
                app_duration_lower_bound_norm[app_num].append(normalised_val)

            meta_values[key]['raw_data'].append(instance)

        instance_count = len(value.keys())
        time_percentage = time_percentage / instance_count
        completed_application_rate = completed_application_rate / instance_count
        tasks_completion_rate = tasks_completion_rate / instance_count


        meta_values[key] = {
            'time_percentage': time_percentage,
            'tasks_completion_rate': tasks_completion_rate,
            'completed_application_rate': completed_application_rate,
            'application_finish_times': application_finish_time,
            'tasks_per_node': tasks_per_node,
            'ram_use_per_node_type': ram_use_per_node_type,
            'cpu_use_per_node_type': cpu_use_per_node_type,
            'max_cpu_per_node_type': max_cpu_per_node_type,
            'max_ram_per_node_type': max_ram_per_node_type,
            'aggregated_normalised_duration_values_by_app': app_duration_lower_bound_norm,
            'tasks_completed_per_node_type': tasks_completed_per_node_type,
            'raw_data': value
        }

    output_str = ""
    for key, items in meta_values.items():
        output_str = output_str + \
            f"{key}\nTime Percentage: {items['time_percentage']}\nAverage Task Completion: {items['tasks_completion_rate']}\nAverage Application Completion Percentage: {items['completed_application_rate']}\n\n"

    with open(f"{input_dir}/{algorithm}_output_file", "w") as f:
        f.write(output_str)

    with open(f"{input_dir}/{algorithm}_output_json.json", "w") as f:
        f.write(json.dumps(meta_values, indent=4))
    return meta_values


def generate_result(input_file, output_file, lower_bound_values):
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
                                     output_application_list, lower_bound_values)
    return results_dict


def generate_analysis(total_time, input_list, output_list, lower_bound_values):
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

    results_dict['finish_times_per_app'] = finish_time_per_application(
        input_list, output_list)

    nodes = applications_per_node(input_list, output_list)

    nodes = highest_concurrent_task_per_node(nodes, output_list, input_list)
    overlapping_tasks_by_task(nodes, output_list, input_list)
    nodes = average_task_per_node(nodes)
    nodes = longest_task_per_node(nodes)
    nodes = shortest_task_per_node(nodes)
    longest_task_per_application(input_list, output_list)
    shortest_task_per_application(input_list, output_list)

    results_dict['tasks_per_node'] = {i: 0 for i in range(0, 5)}
    results_dict['tasks_per_node'] = {**results_dict['tasks_per_node'], **{
        int(key): len(value['tasks']) for key, value in nodes.items()}}

    results_dict['tasks_completed_per_node_type'] = tasks_completed_per_node_type(nodes, total_task_count)
    resources_per_node_type = resource_use_per_node_type(nodes, total_time)

    mx_resource_use_per_node_type = max_resource_use_per_node_type(
        resources_per_node_type, total_time)

    results_dict["normalised_app_time"] = normalised_app_completion_time_to_lower_bound(
        results_dict['finish_times_per_app'], lower_bound_values)

    results_dict['ram_use_per_node_type'] = resources_per_node_type[0]
    results_dict['cpu_use_per_node_type'] = resources_per_node_type[1]
    results_dict['max_cpu_per_type'] = mx_resource_use_per_node_type['max_cpu_per_type']
    results_dict['max_ram_per_type'] = mx_resource_use_per_node_type['max_ram_per_type']

    return results_dict


def tasks_completed_per_node_type(nodes, total_task_count):
    tasks_per_type = {ky: 0 for ky in node_types}

    for node in nodes.values():
        tasks_per_type[node["type"]] = tasks_per_type[node["type"]] + len(node['tasks'])
    
    res = {}
    for ky, val in tasks_per_type.items():
        res[ky] = val / total_task_count

    total = 0
    for val in res.values():
        total = total + val
    res['total'] = total
    return res


def normalised_app_completion_time_to_lower_bound(duration_times_per_app, lower_bound_values):
    res = {index + 1: ((duration_times_per_app[index + 1][1] - duration_times_per_app[index + 1][0]) / value) for index, value in enumerate(lower_bound_values)}

    max_val = max(res.values())

    res = {k: v if v >= 0 else max_val for k, v in res.items()}

    return res


def max_resource_use_per_node_type(node_type_resources, total_time):
    cpu_max_per_type = {ky: -1 for ky in node_type_resources[0].keys()}
    ram_max_per_type = {ky: -1 for ky in node_type_resources[0].keys()}

    for i in range(0, int(total_time)):
        for ky in cpu_max_per_type.keys():
            if node_type_resources[1][ky][i] > cpu_max_per_type[ky]:
                cpu_max_per_type[ky] = node_type_resources[1][ky][i]

            if node_type_resources[0][ky][i] > ram_max_per_type[ky]:
                ram_max_per_type[ky] = node_type_resources[0][ky][i]

    return {
        "max_cpu_per_type": cpu_max_per_type,
        "max_ram_per_type": ram_max_per_type
    }


def resource_use_per_node_type(nodes, total_time):
    resource_types = {res_typ: sorted([task for node in nodes.values(
    ) if node['type'] == res_typ for task in node['tasks']], key=lambda x: x['finish_time']) for res_typ in node_types}

    resource_types_summed_ram = {res_typ: [] for res_typ in node_types}
    resource_types_summed_cores = {res_typ: [] for res_typ in node_types}

    node_type_total_resources = {
        f"{node['type']}": {
            "cpu": 0,
            "ram": 0
        } for node in topology}

    for node in topology:
        node_type_total_resources[node['type']
                                  ]['cpu'] = node_type_total_resources[node['type']]['cpu'] + node['cpu']
        node_type_total_resources[node['type']
                                  ]['ram'] = node_type_total_resources[node['type']]['ram'] + node['ram']

    for i in range(0, int(total_time)):
        for n_type, task_list in resource_types.items():
            cpu = 0
            ram = 0
            for task in task_list:
                if task['start_time'] <= i + 1 and i <= task['finish_time']:
                    duration = 1

                    if task['start_time'] >= i:
                        duration = duration - (task['start_time'] - i)
                    if task['finish_time'] < i + 1:
                        duration = duration - ((i + 1) - task['finish_time'])

                    cpu = cpu + (task['task']['cores'] * duration)
                    ram = ram + (task['task']['ram'] * duration)

            resource_types_summed_cores[n_type].append(
                cpu / node_type_total_resources[n_type]['cpu'])
            resource_types_summed_ram[n_type].append(
                ram / node_type_total_resources[n_type]['ram'])

    return [resource_types_summed_ram, resource_types_summed_cores]


def finish_time(output_list):
    last_finish_time = 0
    for allocation in output_list:
        if allocation['finish_time'] > last_finish_time:
            last_finish_time = allocation['finish_time']
    return last_finish_time


def finish_time_per_application(input_list, output_list):
    finish_times = {}
    for i in range(0, len(input_list)):
        fin_time = 0
        for task_a in input_list[i]['tasks']:
            for allocation in output_list:
                if task_a['name'] == allocation['task']['name']:
                    if allocation['finish_time'] > fin_time:
                        fin_time = allocation['finish_time']
        finish_times[i + 1] = [input_list[i]['offload_time'], fin_time]

    # print("")
    # for i in range(0, len(finish_times)):
    #     print(f"Application {i} finish time: {finish_times[i]}")
    return finish_times


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

                if current_task['start_time'] <= compare_task['finish_time'] and compare_task['start_time'] <= \
                        current_task['finish_time']:
                    node['tasks'][i]['task']['overlapping_tasks'].append(
                        node['tasks'][x])

    # for key in nodes.keys():
    #     for x in range(0, len(nodes[key]['tasks'])):
    #         task = calculate_highest_parrallel_tasks(nodes[key]['tasks'][x])
    #         nodes[key]['tasks'][x] = task

    return nodes


def calculate_highest_parallel_tasks(task):
    finished = False

    overlapping_windows = []
    for task_item in task['task']['overlapping_tasks']:
        names = {task['task']['name'], task_item['task']['name']}
        start_time = task['start_time']
        fin_time = task['finish_time']

        if task_item['start_time'] > start_time:
            start_time = task_item['start_time']
        if task_item['finish_time'] < fin_time:
            fin_time = task_item['finish_time']

        overlapping_windows.append([names, [start_time, fin_time]])

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
                    fin_time = finish_time_a

                    names = overlapping_windows[i][0].union(
                        overlapping_windows[x][0])

                    if start_time_b > start_time:
                        start_time = start_time_b
                    if finish_time_b < fin_time:
                        fin_time = finish_time_b

                    new_windows.append([names, [start_time, fin_time]])
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
        if (output_lines[i].strip() == "===================="):
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

    first_level_subdirectory = [
        f.name for f in os.scandir(output_directory) if f.is_dir()]

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    algorithm_meta_values = {}
    lower_bound_values = read_lower_bound_vals(lower_bound_dir)

    for directory in first_level_subdirectory:
        if directory.startswith('.'):   # to ignore any hidden file
            continue
        algorithm_debug_name = directory
        algorithm_meta_values[directory] = generate_meta_values(f"{output_directory}/{directory}", directory,
                                                                input_directory, lower_bound_values)

    generate_graphs(algorithm_meta_values)
