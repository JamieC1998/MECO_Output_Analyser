import os
from re import I
import sys
import json
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.core.fromnumeric import std
DEBUG_MODE= ""

column_width = 0.16

meta_folder = "./results/"
algorithm_folder = f"{meta_folder}output_dir"
graph_folder = f"{meta_folder}graphs"
script_name = "algorithm_output"

algorithms = ['reactive_basic', 'reactive_mobile', 'preallocation', 'proactive', 'partition']
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
devices = list(topology.keys())


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


def graph_communication_v_computation_comparison(comms_comp_ratios, folder_path):
    """
    Parameters
    ==========
    This requires both the 10 and 30% results to be available. Recommendation is
    to pickle them separately and reload them before this is called, as processing
    both results takes lots of memory.

    comms_comp_ratios: dictionary {10: {'generic': agg_vals['communication_computation'],
                                        'dnn': agg_vals_dnn['communication_computation']},
                                   30: <similar>}

    folder_path: ...
    """
    fig, axs = plt.subplots(2,1, sharex=True)
    comms_comp_ratios_10 = comms_comp_ratios[10]
    comms_comp_ratios_30 = comms_comp_ratios[30]
    for app_idx, app_type in enumerate(['generic', 'dnn']):
        for idx, alg in enumerate(algs):
            comm_mean = np.mean(comms_comp_ratios_10[app_type][alg]['comm'])
            comp_mean = np.mean(comms_comp_ratios_10[app_type][alg]['comp'])
            comm_ratio = comm_mean/(comm_mean+comp_mean)
            comp_ratio = comp_mean/(comm_mean+comp_mean)
            axs[app_idx].bar(1+idx, comm_ratio, bottom=0, width=0.4, color='tab:blue')
            axs[app_idx].bar(1+idx, comp_ratio, bottom=comm_ratio, width=0.4, color='tab:orange')
            comm_mean = np.mean(comms_comp_ratios_30[app_type][alg]['comm'])
            comp_mean = np.mean(comms_comp_ratios_30[app_type][alg]['comp'])
            comm_ratio = comm_mean/(comm_mean+comp_mean)
            comp_ratio = comp_mean/(comm_mean+comp_mean)
            axs[app_idx].bar(1+idx+0.45, comm_ratio, bottom=0, width=0.4,
                             color='tab:blue', hatch='//')
            axs[app_idx].bar(1+idx+0.45, comp_ratio, bottom=comm_ratio, width=0.4,
                             color='tab:orange', hatch='//')
    axs[0].set_ylabel('Generic')
    axs[1].set_ylabel('DNN')
    # Set xitcks on top plot so the labels from top don't overlap with bottom
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # Set xticks as algorithm names
    axs[1].set_xticks([1.2+x for x in range(5)])
    axs[1].set_xticklabels(['reactive\nbasic', 'reactive\nmobile',
                            'preallocation', 'proactive', 'partition'])
    # Patches for the legend
    patches = [mpatches.Patch(color='tab:orange', label='computation'),
               mpatches.Patch(color='tab:blue', label='communication'),
               mpatches.Patch(fill=False, hatch='//',edgecolor='black', label='30% non-offloadable'),
               mpatches.Patch(fill=False, edgecolor='black', label='10% non-offloadable')]
    # Display the legend only on the bottom plot
    axs[1].legend(handles=patches, loc='lower center', ncol=2)
    plt.tight_layout()  # Maximise plots in figure
    fig.subplots_adjust(hspace=0)   # Remove the space between top and bottom
    plt.savefig(f"{folder_path}/comms_vs_comp_comparison.pdf", dpi=150)

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

def graph_task_completion_rate_by_node_type_hatched(app_task_rate, folder_path):
    """
    Hatched plot with reduced legend for the task completion and node type usage.
    """
    x_coords = range(2,21,2) # Number of applications considered
    fills = {'mobile': 'xxxx', 'edge': '..', 'cloud': '****'}   # Hatches for each device type
    colors = {'reactive_basic': 'tab:blue',     # Colors for the algorithms
              'reactive_mobile': 'tab:orange',
              'preallocation': 'tab:green',
              'proactive': 'tab:red',
              'partition': 'tab:purple'}

    for idx, alg in enumerate(algorithms):
        bottom = np.array([np.float64(0) for x in x_coords])
        for dev in devices:
            # Get the values for this type of device
            vals = np.array([
                np.mean([app_task_rate[alg][str(x)][_iter][dev]
                    for _iter in app_task_rate[alg][str(x)].keys()])
                for x in x_coords])
            # Plot a bar for each application set size
            plt.bar([x+idx*0.3 for x in x_coords], vals,
                    bottom=bottom, width=0.3,
                    edgecolor='black', color=colors[alg], hatch=fills[dev])
            # Update the bottom of the bars
            bottom += vals
    plt.xticks(x_coords, x_coords)

    # Prepare the legend with manual patches for the colors (algs) and hatches (devices)
    leg_algs = [mpatches.Patch(fill=False, hatch=h, label=d) for d, h in fills.items()]
    leg_devs = [mpatches.Patch(color=c, label=a) for a,c in colors.items()]
    # The location of the legend and the ylim of the graph should be tweaked
    # based on the graph.
    plt.ylim([0,1.2])
    plt.legend(handles=leg_fills+patches, ncol=3, fontsize='small', loc='upper right')
    plt.grid(True, axis='y')
    # modified the following so that set size is always 20
    plt.savefig(
        f"{folder_path}/tasks_completed_per_node_type_set_size_20.pdf")
    plt.close()
    return

def graph_app_completion_rate_generic_dnn(folder_path):
    """
    Two-in-one app completion rate results plot, combines generic and dnn.
    Requires obtaining both results. Below it is done in the function but
    could be passed as parameters.
    """
    # The following required to obtain the results for both generic and DNN
    sim_results = fetchAlgorithmResults('generic')
    sim_results_dnn = fetchAlgorithmResults('dnn')
    agg_vals = aggregate_values(sim_results)
    agg_vals_dnn = aggregate_values(sim_results_dnn)

    x_coords = range(2,21,2)
    fig, axs = plt.subplots(2,1, sharex=True)
    for app_idx, app_type in enumerate([agg_vals, agg_vals_dnn]):
        app_comp = app_type['app_completion_rate_per_app_size']
        for idx, alg in enumerate(algs):
            axs[app_idx].bar([x+idx*0.3 for x in x_coords],
                             [np.mean(app_comp[alg][str(x)]) for x in x_coords],
                             width=0.3)
    # Tick labels for x axis
    axs[1].set_xticks([x+0.5 for x in x_coords])
    axs[1].set_xticklabels(x_coords)
    # Tick labels for y axis of top plot
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # Axis labels
    axs[1].set_xlabel('# applications')
    axs[0].set_ylabel('Generic')
    axs[1].set_ylabel('DNN')
    # Plot legend
    axs[0].legend(algs, ncol=3, fontsize='small')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

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
