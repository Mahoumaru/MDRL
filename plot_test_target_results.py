import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="whitegrid")
sns.set_palette("tab10")

import argparse
import subprocess as sp
import fnmatch
import yaml
import os

import operator
from pytorch_rl_collection.utils import make_Dirs, interquartile_mean
from scipy.stats import iqr
from scipy.stats import t as StudentT

################################################################################
BASE_DIR = "./results/test_target_results/"
aggregation_function = np.mean#interquartile_mean#
range_function = np.std#iqr#
PER_SEED = False#True#

################################################################################
def fetch_args():
    parser = argparse.ArgumentParser(description='PyTorch RL Collection Launcher Arguments')
    parser.add_argument('--env_name', default='Pendulum-v1', type=str, help='Open-ai gym environment (default: Pendulum-v1)')
    parser.add_argument('--algos', default=["sac"], nargs='+', type=str, help='The algos to run. (default: ["sac"])')
    parser.add_argument('--actor_seeds', default=[1], nargs='+', type=int, help='The Actor seeds to run. (default: [1])')
    parser.add_argument('--use_ut', action='store_true', help='Set agent to use the Unscented Transform in the actor update (default: False)')
    parser.add_argument('--explicit_scal', action='store_true', help='Whether to use explicit scalarization for the critic (default: False)')
    parser.add_argument('--ccs', action='store_true', help='Run with CCS target (default: False)')
    parser.add_argument('--use_osi', action='store_true', help='Load test with osi data (default: False)')
    parser.add_argument('--continuous', action='store_true', help='Continuous varpi. (default: False)')
    parser.add_argument('--multi_dim', action='store_true', help='Multi-dimensional parameter space. (default: False)')
    parser.add_argument('--her', action='store_true', help='Hindsight Experience Replay results. (default: False)')
    return parser.parse_args()

################################################################################
### See https://rosettacode.org/wiki/Welch%27s_t-test#Using_NumPy_&_SciPy
def welch_ttest(x1=None, x2=None, m1=None, m2=None, s1=None, s2=None, N1=None, N2=None):
    if m1 is None:
        m1 = np.mean(x1)
    if m2 is None:
        m2 = np.mean(x2)
    if s1 is None:
        v1 = np.var(x1, ddof=1)
    else:
        v1 = s1**2
    if s2 is None:
        v2 = np.var(x2, ddof=1)
    else:
        v2 = s2**2
    if x1 is None:
        n1 = N1
    else:
        n1 = x1.size
    if x2 is None:
        n2 = N2
    else:
        n2 = x2.size
    ######
    t = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
    p = 2 * StudentT.cdf(-abs(t), df)
    return {"statistic": t, "dof": df, "pvalue": p}

################################################################################
def plot_confidence_interval(x, mean, std, color='#2187bb', horizontal_line_width=0.25):
    confidence_interval = z * std / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color=color)

################################################################################
def plot_stdev(xval, mean, std, color='#2187bb', horizontal_line_width=0.25):
    left = xval - horizontal_line_width / 2
    top = mean - std
    right = xval + horizontal_line_width / 2
    bottom = mean + std
    for i, x in enumerate(xval):
        plt.plot([x, x], [top[i], bottom[i]], color=color)
        plt.plot([left[i], right[i]], [top[i], top[i]], color=color)
        plt.plot([left[i], right[i]], [bottom[i], bottom[i]], color=color)
    plt.plot(xval, mean, 'o', color=color)

################################################################################
#print(data)
def convert_string_to_numpy(serie):
    result = np.fromstring(
        serie.replace('\n','')
         .replace('[','')
         .replace(']','')
         .replace('  ',' '), sep=' ')
    return result

################################################################################
# See: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-or-more-groups-given-known-group-varianc
def compute_mean_from_two_groups(p1, p2):
    (m1, n1), (m2, n2) = p1, p2
    return (n1 * m1 + n2 * m2) / (n1 + n2)

################################################################################
def compute_std_from_two_groups(p1, p2, m12):
    (s1, m1, n1), (s2, m2, n2) = p1, p2
    return np.sqrt(np.max((np.zeros_like(m12), (n1 * (s1**2 + m1**2) + n2 * (s2**2 + m2**2)) / (n1 + n2) - m12**2), axis=0))

################################################################################
def keep_three_best_values(new_elt, best_vals, prev_best_vals, relate=operator.gt):
    assert len(best_vals) == 3, "Length of best_vals = {} is not 3".format(best_vals)
    assert len(prev_best_vals) == 2, "Length of prev_best_vals = {} is not 2".format(prev_best_vals)
    if relate(new_elt, best_vals[0]):
        prev_best_vals[0] = best_vals[0]
        best_vals[0] = new_elt
        if relate(prev_best_vals[0], best_vals[1]):
            prev_best_vals[1] = best_vals[1]
            best_vals[1] = prev_best_vals[0]
            if relate(prev_best_vals[1], best_vals[2]):
                best_vals[2] = prev_best_vals[1]
        elif relate(prev_best_vals[0], best_vals[2]):
            best_vals[2] = prev_best_vals[0]
    elif relate(new_elt, best_vals[1]):
        prev_best_vals[1] = best_vals[1]
        best_vals[1] = new_elt
        if relate(prev_best_vals[1], best_vals[2]):
            best_vals[2] = prev_best_vals[1]
    elif relate(new_elt, best_vals[2]):
        best_vals[2] = new_elt
    #####
    return best_vals, prev_best_vals

################################################################################
def get_latex_table_lines_ccs(env_name, selected_data, Agent_Seeds, My_Algo_Names, Env_Params, continuous, n_regions, Regions, keys=["Env. Returns", "CCS Return"], keep_best_model=False):
    ##############
    if continuous:
        path = BASE_DIR + "{}_test_cont_her_params.pdf".format(env_name)
    else:
        path = BASE_DIR + "{}_test_params.pdf".format(env_name)
    param_scatterplot_exist = True#os.path.isfile(path) or os.path.islink(path)
    if not param_scatterplot_exist:
        fig, axes = plt.subplots(1, len(Env_Params), figsize=(15, 10), sharey=True)
    ##############
    latex_table_lines = {"Head": "Parameters & Algos & $V_{i}$ & $S$"}
    ##############
    values_dict = {key: ([], []) for key in My_Algo_Names}
    ##############
    for i, param in enumerate(Env_Params):
        if len(list(selected_data.loc[selected_data.Env_Params.isin([param])][keys[1]])) == 0:#len(best_vals) == 0:
            continue
        if not param_scatterplot_exist and (n_regions==5):
            np_param = np.round(convert_string_to_numpy(param), 3).reshape(-1, 2)
            if continuous:
                pass
            else:
                #print(np_param)
                if isinstance(axes, tuple):
                    ax = axes[i]
                else:
                    ax = axes
                for p in np_param:
                    #print(i, '$P_{}$'.format(i+1))
                    ax.title.set_text('$P_{}$'.format(i+1))
                    for reg in Regions:
                        ax.vlines(x=reg[0], ymin=0., ymax=reg[1], colors='k', alpha=0.5)
                        ax.hlines(y=reg[1], xmin=0., xmax=reg[0], colors='k', alpha=0.5)
                    ax.scatter(p[0], p[1])
                    ax.set_xlim(left=0.5, right=2.)
                    ax.set_ylim(bottom=0.5, top=2.)
        #########################
        #########################
        latex_table_lines["P{}".format(i+1)] = "\\multirow{{{}}}{{*}}{{$P_{{{}}}$}}".format(len(My_Algo_Names), i+1)
        best_vals = [-np.inf if "Return" in keys[0] else np.inf]*3
        prev_best_vals = [-np.inf if "Return" in keys[0] else np.inf]*2
        colors = ["red", "mydarkgreen", "blue"]
        #colors = ["gold(metallic)", "silver", "copper"]
        for k, algo in enumerate(My_Algo_Names):
            total_ret_means = []
            total_ret_ccs_means = []
            best_seed = 1
            total_ret_means_per_seed = []
            total_ret_stds_per_seed = []
            total_ret_ccs_means_per_seed = []
            total_ret_ccs_stds_per_seed = []
            for j, selected_agent_seed in enumerate(Agent_Seeds):
                df = selected_data.loc[selected_data.Algo.isin([algo]) & selected_data.Env_Params.isin([param]) & selected_data.Actor_Seed.isin([selected_agent_seed])]
                ##### MEAN
                try:
                    ret_means_pandas_array = df[keys[0]].array
                    ret_means = []
                    for elt in ret_means_pandas_array:
                        ret_means.append(convert_string_to_numpy(elt))
                    ret_means = np.array(ret_means)
                    ret_ccs_means = np.array(df[keys[1]].array)#[0]
                except IndexError:
                    continue
                ####
                #print(algo, selected_agent_seed, type(ret_means), type(ret_ccs_means))#, len(ret_means), len(ret_ccs_means))
                #print(ret_means.shape)
                #print(ret_ccs_means.shape)
                total_ret_means += list(ret_means)
                total_ret_ccs_means += list(ret_ccs_means)
                ####
                total_ret_means_per_seed.append(aggregation_function(ret_means, axis=0))
                total_ret_stds_per_seed.append(range_function(ret_means, axis=0))
                total_ret_ccs_means_per_seed.append(aggregation_function(ret_ccs_means))
                total_ret_ccs_stds_per_seed.append(range_function(ret_ccs_means))
            #######
            #print(np.array(total_ret_means).shape, np.array(total_ret_ccs_means).shape)
            if keep_best_model:
                assert len(total_ret_means_per_seed) == len(Agent_Seeds), "{} vs {}".format(len(total_ret_means_per_seed), len(Agent_Seeds))
                assert len(total_ret_ccs_means_per_seed) == len(Agent_Seeds), "{} vs {}".format(len(total_ret_ccs_means_per_seed), len(Agent_Seeds))
                best_ret_idx = np.argmax(total_ret_ccs_means_per_seed, axis=0) if "Return" in keys[0] else np.argmin(total_ret_ccs_means_per_seed, axis=0)
                best_seed = Agent_Seeds[best_ret_idx]
                ####
                total_ret_means = total_ret_means_per_seed[best_ret_idx]
                total_ret_stds = total_ret_stds_per_seed[best_ret_idx]
                total_ret_ccs_means = total_ret_ccs_means_per_seed[best_ret_idx]
                total_ret_ccs_stds = total_ret_ccs_stds_per_seed[best_ret_idx]
            else:
                if PER_SEED:
                    total_ret_stds = range_function(total_ret_means_per_seed, axis=0)#
                    total_ret_means = aggregation_function(total_ret_means_per_seed, axis=0)#
                    total_ret_ccs_stds = range_function(total_ret_ccs_means_per_seed)#
                    total_ret_ccs_means = aggregation_function(total_ret_ccs_means_per_seed)#
                else:
                    total_ret_stds = range_function(total_ret_means, axis=0)#(total_ret_means_per_seed, axis=0)#
                    total_ret_means = aggregation_function(total_ret_means, axis=0)#(total_ret_means_per_seed, axis=0)#
                    total_ret_ccs_stds = range_function(total_ret_ccs_means)#(total_ret_ccs_means_per_seed)#
                    total_ret_ccs_means = aggregation_function(total_ret_ccs_means)#(total_ret_ccs_means_per_seed)#
            #######
            ## keep only the min and max of the uncertainy's environments scores
            min_idx, max_idx = np.argmin(total_ret_means), np.argmax(total_ret_means)
            total_ret_means, total_ret_stds = np.round(total_ret_means, 4), np.round(total_ret_stds, 4)
            total_ret_means = [total_ret_means[min_idx], total_ret_means[max_idx]]
            total_ret_stds = [total_ret_stds[min_idx], total_ret_stds[max_idx]]
            #######
            total_ret_ccs_means = np.round(total_ret_ccs_means, 4)
            total_ret_ccs_stds = np.round(total_ret_ccs_stds, 4)
            #######
            #######
            if "Return" in keys[0]:
                best_vals, prev_best_vals = keep_three_best_values(total_ret_ccs_means, best_vals, prev_best_vals, relate=operator.gt)
            else:
                best_vals, prev_best_vals = keep_three_best_values(total_ret_ccs_means, best_vals, prev_best_vals, relate=operator.lt)
            #######
            if keep_best_model:
                latex_table_lines["P{}".format(i+1)] += " & {} & ${}$ & ${}$ \\\\".format(algo, (total_ret_means, best_seed), total_ret_ccs_means)
            else:
                latex_table_lines["P{}".format(i+1)] += " & {} & ${}$ & ${}$ \\\\".format(algo, total_ret_means, total_ret_ccs_means)
            #min_total_ret_means = min(total_ret_means)
            #latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("{}".format(min_total_ret_means), "\\textbf{{{}}}".format(min_total_ret_means))
            latex_table_lines["P{}".format(i+1)] += " & & $({} {})$ & $({} {})$ \\\\".format(u"\u00B1", total_ret_stds, u"\u00B1", total_ret_ccs_stds)
            #####
            values_dict[algo][0].append(total_ret_ccs_means)
            values_dict[algo][1].append(total_ret_ccs_stds)
        #####
        #latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("${}$".format(best_val), "$\\textbf{{{}}}$".format(best_val))
        #print(best_vals)
        if len(My_Algo_Names) > 3:
            for n, elt in enumerate(best_vals):
                latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("${}$".format(elt), "$\\textcolor{{{}}}{{\\textbf{{{}}}}}$".format(colors[n], elt))
        elif len(My_Algo_Names) > 2:
            for n, elt in enumerate(best_vals[:2]):
                latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("${}$".format(elt), "$\\textcolor{{{}}}{{\\textbf{{{}}}}}$".format(colors[n], elt))
        else:
            latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("${}$".format(best_vals[0]), "$\\textcolor{{{}}}{{\\textbf{{{}}}}}$".format(colors[0], best_vals[0]))
        latex_table_lines["P{}".format(i+1)] += "\n\hline"
    #######
    if not param_scatterplot_exist:
        plt.savefig(path)
    #print(latex_table_lines)
    #print(jklkl)
    #####
    return latex_table_lines, values_dict

################################################################################
def get_latex_table_lines(selected_data, Agent_Seeds, Varpis, My_Algo_Names, Env_Params, n_regions, VARPI_REPR):
    latex_table_lines = {"Head": "Parameters coefficients & Algos & $Score$"}
    for i, param in enumerate(Env_Params):
        #print("++ ", selected_data)#.loc[selected_data.Env_Params.isin([param])])
        #print(klkl)
        #######
        if len(list(selected_data.loc[selected_data.Env_Params.isin([param])]["Env. Returns Mean"])) == 0:
            continue
        latex_table_lines["P{}".format(i+1)] = "\\multirow{{{}}}{{*}}{{{}}}".format(len(My_Algo_Names), param.replace("_", "\_"))
        ##########
        best_val = -np.inf
        for k, algo in enumerate(My_Algo_Names):
            if "MD" in algo:
                algo_varpi = [v for v in Varpis if v != str([0.]*n_regions)]
            else:
                algo_varpi = [str([0.]*n_regions)]
            #########
            for selected_varpi in algo_varpi:
                total_ret_means = 0.
                total_ret_stds = 0.
                prev_n_runs = 0
                n_runs_per_seed = 10#50
                #print("++ ", selected_data.loc[selected_data.Algo.isin([algo]) & selected_data.Env_Params.isin([param])])
                for j, selected_agent_seed in enumerate(Agent_Seeds):
                    df = selected_data.loc[selected_data.Algo.isin([algo]) & selected_data.Env_Params.isin([param])
                                         & selected_data.Actor_Seed.isin([selected_agent_seed])
                                         & selected_data.Varpi.isin([selected_varpi])
                    ]
                    #print(i+1, algo, df["Env. Returns Mean"].array, selected_varpi)
                    #####
                    prev_total_ret_means = np.copy(total_ret_means)
                    ##### MEAN
                    ret_means = df["Env. Returns Mean"].array
                    if len(ret_means) == 0:
                        print(df, ret_means, selected_varpi)
                        continue
                    ret_means = ret_means[0]
                    total_ret_means = compute_mean_from_two_groups((total_ret_means, prev_n_runs), (ret_means, n_runs_per_seed))
                    #### STD
                    ret_stds = df[keys[1]].array[0]
                    total_ret_stds = compute_std_from_two_groups((total_ret_stds, prev_total_ret_means, prev_n_runs), (ret_stds, ret_means, n_runs_per_seed), total_ret_means)
                    ####
                    prev_n_runs += n_runs_per_seed
                ######
                total_ret_means = np.round(total_ret_means, 2)
                total_ret_stds = np.round(total_ret_stds, 2)
                if "MD" in algo and not continuous:
                    malgo = algo + " " + VARPI_REPR[selected_varpi]
                elif "MD" in algo and continuous:
                    malgo = algo + " " + selected_varpi.replace(" ", ",")
                else:
                    malgo = algo
                if total_ret_means > best_val:
                    best_val = total_ret_means
                latex_table_lines["P{}".format(i+1)] += " & {} & ${}$ \\\\".format(malgo, total_ret_means)
                latex_table_lines["P{}".format(i+1)] += " & & $({} {})$ \\\\".format(u"\u00B1", total_ret_stds)
        #####
        latex_table_lines["P{}".format(i+1)] = latex_table_lines["P{}".format(i+1)].replace("${}$".format(best_val), "$\\textbf{{{}}}$".format(best_val))
        latex_table_lines["P{}".format(i+1)] += "\n\hline"

    #######
    #print(latex_table_lines)
    return latex_table_lines

################################################################################
def get_latex_table_lines_uevol(env_name, selected_data, colors, Agent_Seeds, My_Algo_Names, Env_Params, n_episodes_per_run=10):
    latex_table_lines = {"Head": "Parameters coefficients & Algos & $Score$"}
    for i, param in enumerate(Env_Params):
        #######
        if len(list(selected_data.loc[selected_data.Env_Params.isin([param])]["Env. Returns Mean"])) == 0:
            continue
        latex_table_lines["P{}".format(i+1)] = "\\multirow{{{}}}{{*}}{{{}}}".format(len(My_Algo_Names), param.replace("_", "\_"))
        ##########
        dico = {}
        xval=np.arange(1, n_episodes_per_run+1)
        horizontal_line_width=0.25
        #plt.switch_backend('TkAgg')
        ##########
        fig, ax = plt.subplots(figsize=(9, 7))
        ##########
        for k, algo in enumerate(My_Algo_Names):
            #########
            total_ret_means = np.zeros(n_episodes_per_run)
            total_ret_stds = np.zeros(n_episodes_per_run)
            prev_n_runs = 0
            n_runs_per_seed = 1 if "DClawTurnFixed" in env_name else 50
            for j, selected_agent_seed in enumerate(Agent_Seeds):
                df = selected_data.loc[selected_data.Algo.isin([algo]) & selected_data.Env_Params.isin([param])
                                     & selected_data.Actor_Seed.isin([selected_agent_seed])
                ]
                #print(i+1, algo, df["Env. Returns Mean"].array)
                #####
                prev_total_ret_means = np.copy(total_ret_means)
                ##### MEAN
                ret_means = df["Env. Returns Mean"].array
                if len(ret_means) == 0:
                    #print(df, ret_means)
                    continue
                ret_means = convert_string_to_numpy(ret_means[0])
                total_ret_means = compute_mean_from_two_groups((total_ret_means, prev_n_runs), (ret_means, n_runs_per_seed))
                #### STD
                ret_stds = convert_string_to_numpy(df["Env. Returns Std"].array[0])
                total_ret_stds = compute_std_from_two_groups((total_ret_stds, prev_total_ret_means, prev_n_runs), (ret_stds, ret_means, n_runs_per_seed), total_ret_means)
                ####
                prev_n_runs += n_runs_per_seed
            ######
            total_ret_means = np.round(total_ret_means, 2)
            total_ret_stds = np.round(total_ret_stds, 2)
            ######
            #dico[algo] = (total_ret_means, total_ret_stds)
            #plot_stdev(xval=np.arange(1, n_episodes_per_run+1), mean=total_ret_means, std=0.*total_ret_stds, color=colors[k], horizontal_line_width=0.25)
            left = xval - horizontal_line_width / 2
            top = total_ret_means - total_ret_stds
            right = xval + horizontal_line_width / 2
            bottom = total_ret_means + total_ret_stds
            #####
            ax.plot(np.array(xval, dtype=np.int32), total_ret_means, 'o-', color=colors[k], label=algo)
            #plt.errorbar(np.array(xval, dtype=np.int32), total_ret_means, yerr=total_ret_stds, marker='o', capsize=3, capthick=1.5, color=colors[k], label=algo)
            ######
            latex_table_lines["P{}".format(i+1)] += " & {} & ${}$ \\\\".format(algo, total_ret_means)
            latex_table_lines["P{}".format(i+1)] += " & & $({} {})$ \\\\".format(u"\u00B1", total_ret_stds)
        #####
        latex_table_lines["P{}".format(i+1)] += "\n\hline"
        #####
        #plt.yscale("symlog")
        ax.set_xlabel("Episodes", fontsize=18)
        ax.set_ylabel("Scores", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.legend(fontsize=18)
        plt.tight_layout()
        #plt.show()
        plt.savefig(BASE_DIR + "{}_perf_osi_ParamsParam.pdf".format(env_name))
        #print(jklk)
        break
    #######
    #print(latex_table_lines)
    return latex_table_lines

################################################################################
def get_date_suffix(algo_name, env_name, multi_dim, her_flag):
    if her_flag: ## HER
        if "Hopper" in env_name:
            SUFFIX = {
                "drsac": "_20240620" if multi_dim else "_20240517",
                "cmdsac": "_20240710" if multi_dim else "_20240802",
                "emdsac": "_20240710" if multi_dim else "_20240802",
                "umdsac": "_20240710" if multi_dim else "_20240802",
                "sirsa": "_20240710" if multi_dim else "_20240622",#"_20240612_2000ep_true_truesirsa",
                "umd_sirsa": "_20240710" if multi_dim else "_20240802",
                "umdsac-v1": "_20240710" if multi_dim else "_20240802",
            }
        elif "Ant" in env_name:
            SUFFIX = {
                "drsac": "_20240719" if multi_dim else "_20240802",
                "cmdsac": "_20240710" if multi_dim else "_20240802",
                "emdsac": "_20240710" if multi_dim else "_20240802",
                "umdsac": "_20240710" if multi_dim else "_20240802",#"_20240602",#
                "sirsa": "_20240710" if multi_dim else "_20240802",#"_20240602_2000ep_truesirsa",#
                "umd_sirsa": "_20240710" if multi_dim else "_20240802",
                "umdsac-v1": "_20240710" if multi_dim else "_20240802",
            }
        elif "Walker2d" in env_name:
            SUFFIX = {
                "drsac": "_20240620" if multi_dim else "_20240517",
                "cmdsac": "_20240710" if multi_dim else "_20240802",
                "emdsac": "_20240710" if multi_dim else "_20240802",
                "umdsac": "_20240710" if multi_dim else "_20240802",
                "sirsa": "_20240710" if multi_dim else "_20240620",#"_20240611_2000ep_true_truesirsa",
                "umd_sirsa": "_20240710" if multi_dim else "_20240802",
                "umdsac-v1": "_20240710" if multi_dim else "_20240802",
            }
        elif "DClawTurnFixed" in env_name:
            SUFFIX = {
                "drsac": "_20240625" if multi_dim else "_20240625",
                "cmdsac": "_20240710" if multi_dim else "_20240802",
                "emdsac": "_20240710" if multi_dim else "_20240802",
                "umdsac": "_20240710" if multi_dim else "_20240802",
                "sirsa": "_20240710" if multi_dim else "_20240625",
                "umd_sirsa": "_20240710" if multi_dim else "_20240802",
                "umdsac-v1": "_20240710" if multi_dim else "_20240802",
            }
        else:
            raise NotImplementedError
    else: ## Not HER
        if "Hopper" in env_name:
            SUFFIX = {
                "drsac": "_20240620" if multi_dim else "_20240517",
                "cmdsac": "_20240729" if multi_dim else "_20240612",
                "emdsac": "_20240729" if multi_dim else "_20240613",
                "umdsac": "_20240729" if multi_dim else "_20240612",
                "sirsa": "_20240710" if multi_dim else "_20240612_2000ep_true_truesirsa",
                "umd_sirsa": "_20240729",
                "umdsac-v1": "_20240729",
            }
        elif "Ant" in env_name:
            SUFFIX = {
                "drsac": "_20240719",
                "cmdsac": "_20240729",
                "emdsac": "_20240729",
                "umdsac": "_20240729",#"_20240602",#
                "sirsa": "_20240710",#"_20240602_2000ep_truesirsa",#
                "umd_sirsa": "_20240729",
                "umdsac-v1": "_20240729",
            }
        elif "Walker2d" in env_name:
            SUFFIX = {
                "drsac": "_20240620" if multi_dim else "_20240517",
                "cmdsac": "__20240729" if multi_dim else "_20240611",
                "emdsac": "_20240729" if multi_dim else "_20240612",
                "umdsac": "_20240729" if multi_dim else "_20240611",
                "sirsa": "_20240710" if multi_dim else "_20240611_2000ep_true_truesirsa",
                "umd_sirsa": "__20240729",
                "umdsac-v1": "_20240729",
            }
        elif "DClawTurnFixed" in env_name:
            SUFFIX = {
                "drsac": "_20240625" if multi_dim else "_20240625",
                "cmdsac": "__20240729" if multi_dim else "_20240624",
                "emdsac": "_20240729" if multi_dim else "_20240624",
                "umdsac": "_20240729" if multi_dim else "_20240624",
                "sirsa": "_20240710" if multi_dim else "_20240625",
                "umd_sirsa": "__20240729",
                "umdsac-v1": "_20240729",
            }
        else:
            raise NotImplementedError
    return SUFFIX[algo_name]

################################################################################
def get_data(files, ccs, n_regions, env_name):
    assert len(files) > 0, "{}".format(filename)
    ####
    data = None
    for filename in files:
        if data is None:
            data = pd.read_csv(filename)
            if "_no_solver" in filename:
                data["Algo"] = data["Algo"].str.replace('umdsac', 'umdsac-v1')
        else:
            df = pd.read_csv(filename)
            if "_no_solver" in filename:
                df["Algo"] = df["Algo"].str.replace('umdsac', 'umdsac-v1')
            data = pd.concat([data, df], ignore_index=True, sort=True)

    data.rename(columns={'Env. Params': 'Env_Params', 'Actor seed': 'Actor_Seed'}, inplace=True)
    #data = data.loc[data["Actor_Seed"].isin([1])]
    ########
    meths = list(data["Algo"].drop_duplicates())
    print("++ Original methods names: ", meths)
    new_meths = []
    for m in meths:
        m = m.upper()
        if "MD" in m or "SDR" in m:
            m = m.replace("ESMDSAC", "esMDSAC")
            m = m.replace("SDRSAC", "sDRSAC") + " (Ours)"
            m = m.replace("SMDSAC", "uMDSAC")
            m = m.replace("CMDSAC", "cMDSAC")
            m = m.replace("CMD", "cMD")
            m = m.replace("EMD", "eMD")
            m = m.replace("SMD_SIRSA", "uMD_SIRSA")
            m = m.replace("EMDSAC", "eMDSAC").replace("_", "-").replace("-V", "-v")
        new_meths.append(m)
    print("++ Renamed methods names: ", new_meths)
    data["Algo"].replace(
        meths,
        new_meths,
        inplace=True
    )
    ########
    agent_seeds = sorted(list(data["Actor_Seed"].drop_duplicates()))
    print("++ Actor seeds: ", agent_seeds)
    ########
    if ccs:
        if "DClaw" in env_name:
            columns=["Env_Params", "Algo", "Varpi", "Env. Returns", "CCS Return",
                    "Env. FinalPos", "CCS FinalPos", "Actor_Seed"
            ]
        else:
            columns=["Env_Params", "Algo", "Varpi", "Env. Returns", "CCS Return",
                    "Actor_Seed"
            ]
    else:
        columns=["Env_Params", "Algo", "Env. Returns", "Actor_Seed"]

    drops = []
    for col in data.columns:
        if col not in columns:
            drops.append(col)

    print("drops: ", drops)
    data = data.drop(drops, axis=1)
    data.fillna(str([0.]*n_regions), inplace=True)
    return data, meths, agent_seeds

################################################################################
def latexify_eval_results(env_name, data, meths, ccs, her_flag, osi_flag, Agent_Seeds, Varpis, My_Algo_Names, Env_Params, continuous, n_regions, Regions, summ_ret=True, keep_best_model=False):
    #"""
    latex_file = "\\documentclass{article}\n\\usepackage[landscape,hmargin=0.1cm]{geometry}\n"
    if not ccs:
        latex_file += "\\usepackage{amssymb, bbold}\n"
    latex_file += "\\usepackage{longtable}\n\\usepackage{multirow}\n\\usepackage{makecell}\n\\usepackage[usenames,dvipsnames]{xcolor}\n\n"
    #latex_file += "\\definecolor{mydarkgreen}{RGB}{23, 163, 13}\n\\renewcommand*{\\arraystretch}{1.12}\n\n"
    latex_file += "\\definecolor{mydarkgreen}{RGB}{26, 184, 15}\n\\renewcommand*{\\arraystretch}{1.12}\n\n"
    latex_file += "\\definecolor{gold(metallic)}{rgb}{0.83, 0.69, 0.22}\n\\renewcommand*{\\arraystretch}{1.12}\n\n"
    latex_file += "\\definecolor{silver}{rgb}{0.75, 0.75, 0.75}\n\\renewcommand*{\\arraystretch}{1.12}\n\n"
    latex_file += "\\definecolor{copper}{rgb}{0.72, 0.45, 0.2}\n\\renewcommand*{\\arraystretch}{1.12}\n\n"
    latex_file += "\\begin{document}\n\\begin{center}\n"

    print("#############################")
    if ccs:
        meths = My_Algo_Names
        final_table_lines = {"Head": "$\\varpi$ & " + " & ".join(meths) + " \\\\"}
        print(final_table_lines)
        all_values_dict = None
        add_varpi_value = (len(list(convert_string_to_numpy(Varpis[0]))) < 8)
        for v, selected_varpi in enumerate(Varpis):
            if add_varpi_value:
                final_table_lines["$\\varpi_{{{}}}$".format(v+1)] = "$\\varpi_{{{}}} = {{{}}}$".format(v+1, list(convert_string_to_numpy(selected_varpi)))
            else:
                final_table_lines["$\\varpi_{{{}}}$".format(v+1)] = "$\\varpi_{{{}}}$".format(v+1)
            ####
            std_line_string = " \\\\"
            print("++ ", v, "/", len(Varpis))
            selected_data = data.loc[data["Varpi"].isin([selected_varpi])]

            ####
            if summ_ret == True:
                latex_table_lines, values_dict = get_latex_table_lines_ccs(env_name, selected_data, Agent_Seeds, My_Algo_Names, Env_Params, continuous, n_regions, Regions, keys=["Env. Returns", "CCS Return"], keep_best_model=keep_best_model)
            else:
                latex_table_lines, values_dict = get_latex_table_lines_ccs(env_name, selected_data, Agent_Seeds, My_Algo_Names, Env_Params, continuous, n_regions, Regions, keys=["Env. FinalPos", "CCS FinalPos"], keep_best_model=keep_best_model)

            ################################################################################
            if all_values_dict is None:
                all_values_dict = values_dict
            else:
                for key, (scores, stds) in values_dict.items():
                    all_values_dict[key][0].extend(scores)
                    all_values_dict[key][1].extend(stds)

            ################################################################################
            latex_file += "\t\\begin{longtable}{|c|c|c|c|c|}\n\t\t\\hline\\hline\n"
            tabulation = "\t\t"
            #latex_file += "{}\\\\\n".format(tabulation)
            for key in latex_table_lines.keys():
                if key != "Head":
                    l = latex_table_lines[key].split("\\ &")
                    for m in meths:
                        for i, elt in enumerate(l):
                            if " " + m + " " in elt:
                                final_table_lines["$\\varpi_{{{}}}$".format(v+1)] += " & " + elt.split(" & ")[-1].replace("\\", "").replace("$textbf", "$\\textbf").replace("{textbf", "{\\textbf").replace("$textcolor", "$\\textcolor")
                                #.replace("$ \\\\", "$").replace("$ \ &", "$ &")
                                std_line_string += " & " + l[i+1].split(" & ")[-1].replace("\n\\hline", "").replace("\\", "")#.replace("$ \\", "$")
                    #####
                latex_file += "{}{}\n".format(tabulation, latex_table_lines[key])
                if key != "P{}".format(len(Env_Params)):
                    latex_file += "{}\\\\\n".format(tabulation)
                if key == "Head" or "std" in key:
                    #latex_file += "{}\\hline\n{}\\\\\n".format(tabulation, tabulation)
                    latex_file += "{}\\hline\n".format(tabulation)
            ###
            final_table_lines["$\\varpi_{{{}}}$".format(v+1)] += std_line_string + " \\\\\n"
            tabulation = "\t"
            if add_varpi_value:
                latex_file += "{}\\caption{{$\\varpi = {}$}}\n".format(tabulation, list(convert_string_to_numpy(selected_varpi)))
            else:
                latex_file += "{}\\caption{{$\\varpi$}}\n".format(tabulation)
            #selected_varpi.replace("[[", "[").replace("]]", "]"))#.replace(" ", ","))
            latex_file += "{}\\end{{longtable}}\n".format(tabulation)
            #####################
            #if selected_varpi != Varpis[-1]:
            latex_file += "{}\\newpage\n".format(tabulation)
        #################
        if n_regions == 5:
            w = "6cm"
        else:
            w = "2cm"
        latex_file += "\t\\begin{{longtable}}{{p{{{}}}|{}}}\n\t\t\\hline\\hline\n".format(w, "|".join(["c" for _ in range(len(meths))]))
        tabulation = "\t\t"
        #latex_file += "{}\\\\\n".format(tabulation)
        for key in final_table_lines.keys():
            latex_file += "{}{}\n".format(tabulation, final_table_lines[key])
            latex_file += "{}\\hline\n".format(tabulation)
        ####
        tabulation = "\t"
        latex_file += "{}\\caption{{Full summary}}\n".format(tabulation)#selected_varpi.replace("[[", "[").replace("]]", "]"))#.replace(" ", ","))
        latex_file += "{}\\end{{longtable}}\n".format(tabulation)
        #################
        print("Plot the results:")
        plt.clf()
        #sns.set_theme(style="whitegrid")
        #sns.set_palette("tab10")
        colors = sns.color_palette("Set2", 10)
        fig, ax = plt.subplots(nrows=1, ncols=len(all_values_dict.items()), sharey=True)#(figsize=(8, 5))
        N = None
        all_values_dict = dict(sorted(all_values_dict.items()))
        all_values_dict = dict(sorted(all_values_dict.items(), key=lambda s: (len(s[0]), s[0][0])))
        ## Collect the best values for each varpi:
        max_scores = None
        min_scores = None
        for _, (scores, _) in all_values_dict.items():
            if max_scores is None:
                max_scores = np.array(scores)
                min_scores = np.array(scores)
            else:
                max_scores = np.max((max_scores, scores), axis=0)
                min_scores = np.min((min_scores, scores), axis=0)
        ## Plot the scores and stds, along with the max_scores
        column_names = []
        for key in all_values_dict.keys():
            column_names += [key + " Scores", key + " Stds"]
        aggregated_data = pd.DataFrame(columns=column_names)
        #min_scores = 2000.
        for i, (key, (scores, stds)) in enumerate(all_values_dict.items()):
            ls = 'dotted'#'-'#
            marker = 'o'
            markersize = 8
            capsize = 3
            if N is None:
                N = len(scores)
                print(N)
                x = np.array([i+1 for i in range(N)])
            else:
                assert N == len(scores), "{} vs {}".format(N, len(scores))
            scores = np.array(scores)
            stds = np.array(stds)
            ###
            aggregated_data[key + " Scores"] = scores
            aggregated_data[key + " Stds"] = stds
            ###
            scores = (scores - min_scores) / (max_scores - min_scores)
            if not summ_ret:
                scores = 1. - scores
            """
            ax.errorbar(x, scores, yerr=stds, linestyle=ls, marker=marker, markersize=markersize, capsize=capsize, label=key)
            """
            ax[i].barh(x, np.ones_like(max_scores), color=colors[1], alpha=0.1)
            ax[i].barh(x, scores, color=colors[0])
            ax[i].set_title(key, fontsize=8)

            #ax[i].tick_params(axis='both', which='major', labelsize=18)
        #plt.legend(fontsize=18)
        ticks_labels = [r"$\varpi_{{{}}}$".format(k) for k in x]
        #ticks_labels[0] += " (Param1)"
        #ticks_labels[1] += " (Full range)"
        plt.yticks(x, ticks_labels)
        plt.tight_layout()
    else:
        colors = sns.color_palette("tab10", 10)
        latex_table_lines = get_latex_table_lines_uevol(env_name, data, colors, Agent_Seeds, My_Algo_Names)

        ################################################################################
        latex_file += "\t\\begin{longtable}{|c|c|c|c|c|}\n\t\t\\hline\\hline\n"
        tabulation = "\t\t"
        #latex_file += "{}\\\\\n".format(tabulation)
        for key in latex_table_lines.keys():
            latex_file += "{}{}\n".format(tabulation, latex_table_lines[key])
            if key != "P{}".format(len(Env_Params)):
                latex_file += "{}\\\\\n".format(tabulation)
            if key == "Head" or "std" in key:
                #latex_file += "{}\\hline\n{}\\\\\n".format(tabulation, tabulation)
                latex_file += "{}\\hline\n".format(tabulation)
        ###
        tabulation = "\t"
        latex_file += "{}\\caption{{{}}}\n".format(tabulation, Env_Params[0].replace("_", "\_"))
        latex_file += "{}\\end{{longtable}}\n".format(tabulation)

    latex_file += "\n\\end{center}\n\\end{document}"

    ################################################################################
    if ccs:
        result_type = ("her" if her_flag else "no_her") + ("_20240710" if her_flag else "_20240729")
        result_type += "_osi" if osi_flag else ""
        if n_regions == 5:
            result_type = result_type.replace("_20240710", "_20240802")
        TEX_DIR = BASE_DIR + "tex_files/{}/{}/".format(result_type, env_name + "_{}d".format((n_regions - 1)//2))
    else:
        TEX_DIR = BASE_DIR + "tex_files_uevol/{}/".format(env_name)
    make_Dirs(TEX_DIR)
    filename = "{}_test_target_results{}{}.tex".format(env_name, "" if summ_ret else "_finalpos", "_bestModel" if keep_best_model else "")
    with open(TEX_DIR + filename, "w") as tex_file:
        tex_file.write(latex_file)
    if ccs:
        plt.savefig(TEX_DIR + filename.replace(".tex", "_plot.pdf"))
    #"""
    ################################################################################
    cmd = "cd {} && pdflatex -halt-on-error {} >/dev/null".format(TEX_DIR, filename)
    proc = sp.Popen(cmd, shell=True)
    cmdRes = proc.wait()
    print("==> pdf file successfully created!!" if cmdRes == 0 else "==> pdflatex failed!!")
    #"""
    if summ_ret and ccs:
        print(aggregated_data)
        aggregated_data.to_csv(TEX_DIR + filename.replace(".tex", ".csv"))

################################################################################
def main():
    args = fetch_args()
    model_type = ""
    add_observation_noise = False
    env_name = args.env_name
    use_encoder = False
    #####
    if use_encoder:
        config_filename = env_name.split("-")[0] + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw.yaml"
    else:
        config_filename = env_name.split("-")[0] + "_for_test" + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw_for_test.yaml"
    ###
    with open("./pytorch_rl_collection/configs/" + config_filename, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ######
    if args.multi_dim:
        config = config["multi_dim"]
    else:
        config = config["2d"]
    ######
    params_idxs = config["params_idxs"]
    env_key_list = list(params_idxs.keys())
    #####
    filename_suffix = ""
    filenames = []
    paths = []
    prev_algo_name = None
    for algo_name in args.algos:
        output_path_suffix = get_date_suffix(algo_name.replace("_no_solver", "-v1"), env_name, args.multi_dim, args.her)
        algo = algo_name.replace("_no_solver", "") + ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP") + ("_cont" if args.continuous else "_disc")
        algo += ("_no_solver" if "_no_solver" in algo_name else "") + "_her"
        ####
        if "sirsa" == algo_name:
            algo = algo.replace("_her", "")
        elif "drsac" == algo_name:
            algo = algo_name
        ####
        if not args.her:
            algo = algo.replace("_her", "_no_her")#.replace("_UT", "_MC")
        ####
        algo_name = algo_name.replace("_no_solver", "")
        for actor_seed in args.actor_seeds:
            output_path = "./trained_agents/{}/{}_{}/{}/".format(
                algo + (("_" + model_type) if model_type != "" else ""),
                env_name + ("_ObsNoise" if add_observation_noise else ""),
                ("all" if len(env_key_list) > 2 else "_".join(env_key_list)) + output_path_suffix,
                actor_seed
            )
            if algo_name != prev_algo_name:
                paths.append(output_path)
                prev_algo_name = algo_name
            ######
            filename = "{}_test_target".format(env_name.split("-")[0]) + ("_ccs" if args.ccs else "")#("_osi" if args.use_osi else ""))
            if "sirsa" == algo_name:# or "cmdsac" == algo_name
                filename = filename.replace("_ccs", "_osi")
            if args.use_osi and "md" in algo_name:
                filename = filename.replace("_ccs", "_osi")
            filename += ("_ut_on_ut_cont_no_her" if args.continuous else "") + "_rndseed{}{}_{}.csv".format(1234, filename_suffix, algo_name)
            filenames.append(output_path + filename)
    #####
    HER = args.her
    CCS = args.ccs
    OSI = args.use_osi
    CONTINUOUS = args.continuous
    env_name = args.env_name.split("-")[0]

    if CONTINUOUS:
        if "Ant" in env_name:
            n_regions = 63 if args.multi_dim else 5#
        elif "Hopper" in env_name:
            n_regions = 25 if args.multi_dim else 5#
        elif "Walker2d" in env_name:
            n_regions = 43 if args.multi_dim else 5#
        elif "DClaw" in env_name:
            n_regions = 45 if args.multi_dim else 5#63
        else:
            n_regions = 5
        ####
        VARPI_REPR = {}
        REGIONS = []
    else:
        raise NotImplementedError
    ###################
    print(filenames[-1], args.use_osi)
    data, meths, Agent_Seeds = get_data(filenames, CCS, n_regions, env_name)
    meths.sort()
    meths.sort(key=lambda s: (len(s), s[0]))
    print(data)
    MY_ALGO_NAMES = list(data["Algo"].drop_duplicates())
    MY_ALGO_NAMES.sort()
    MY_ALGO_NAMES.sort(key=lambda s: (len(s), s[0]))#(key=lambda s: len(s))
    ENV_PARAMS = list(data["Env_Params"].drop_duplicates())
    print(MY_ALGO_NAMES)
    print(len(ENV_PARAMS))
    if CCS:
        VARPIS = list(data["Varpi"].drop_duplicates())
        print(len(VARPIS), VARPIS[2])
    else:
        VARPIS = []
    #####
    latexify_eval_results(env_name, data, meths, CCS, HER, OSI, Agent_Seeds, VARPIS, MY_ALGO_NAMES, ENV_PARAMS, CONTINUOUS, n_regions, REGIONS, summ_ret=True, keep_best_model=False)
    if CCS:
        latexify_eval_results(env_name, data, meths, CCS, HER, OSI, Agent_Seeds, VARPIS, MY_ALGO_NAMES, ENV_PARAMS, CONTINUOUS, n_regions, REGIONS, summ_ret=True, keep_best_model=True)
        if "DClaw" in env_name:
            latexify_eval_results(env_name, data, meths, CCS, HER, OSI, Agent_Seeds, VARPIS, MY_ALGO_NAMES, ENV_PARAMS, CONTINUOUS, n_regions, REGIONS, summ_ret=False, keep_best_model=False)
            latexify_eval_results(env_name, data, meths, CCS, HER, OSI, Agent_Seeds, VARPIS, MY_ALGO_NAMES, ENV_PARAMS, CONTINUOUS, n_regions, REGIONS, summ_ret=False, keep_best_model=True)
    #####
    print(paths)

################################################################################
if __name__ == "__main__":
    main()
