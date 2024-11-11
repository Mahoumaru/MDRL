import argparse
import subprocess as sp

###########################################
parser = argparse.ArgumentParser(description='PyTorch RL Collection Launcher Arguments')
##
parser.add_argument('--algos', default=["cmdarddpg", "emdarddpg"], nargs='+', type=str, help='The algos to run. (default: ["cmdarddpg", "emdarddpg"])')
parser.add_argument('--env_name', default='HalfCheetah-v2', type=str, help='Open-ai gym environment (default: HalfCheetah-v2)')
parser.add_argument('--n_runs', default=1, type=int, help='Number of runs (default: 1)')
parser.add_argument('--first_run_idx', default=0, type=int, help='Index of the first run (default: 0)')
parser.add_argument('--uncertainty_alpha', default=0.1, type=float, help='Uncertainty parameters controlling the strength of the disturber. Value must be in [0., 0.5) (default: 0.1)')
parser.add_argument('--rel_mass_range', default=[0.5, 2.], type=lambda s: [float(item) for item in s.split(',')], help='Range of relative mass (default: [0.5, 2])')
parser.add_argument('--alpha_type', default='scalar', choices=["scalar", "vector"], type=str, help='Type of the alpha: ["scalar", "vector"] (default: scalar)')
parser.add_argument('--use_kdtree', action='store_true', help='Set the agent to use the kd_tree for sampling alpha (default: False)')
parser.add_argument('--algo_type', default='cmd', choices=["cmd", "umd", "emd"], type=str, help='Type of the multi-domain algorithm: ["cmd", "umd", "emd"] (default: cmd)')
parser.add_argument('--hypernet', action='store_true', help='Use the hypernetwork structure for critic and actor (default: False)')
parser.add_argument('--resample', action='store_true', help='Re-sample the environment at each time step. (default: False)')
parser.add_argument('--continuous', action='store_true', help='Run with continuous varpi distribution. (default: False)')
parser.add_argument('--n_regions', default=10, type=int, help='Number of subdomains (default: 10)')
parser.add_argument('--ratio', default='1_1', type=str, help='Agent to Adversary training ratio (default: 1_1)')
parser.add_argument('--cuda_id', default=0, type=int, help='Set GPU id to use with CUDA (default: 0)')
parser.add_argument('--no_encoding', action='store_true', help="Don't employ the encoded parameters. (default: False)")
parser.add_argument('--use_ut', action='store_true', help='Set agent to use the Unscented Transform in the actor update (default: False)')
parser.add_argument('--explicit_scal', action='store_true', help='Whether to use explicit scalarization for the critic (default: False)')
parser.add_argument('--just_run_test_target', action='store_true', help='Only evaluate trained policy. (default: False)')
parser.add_argument('--multi_dim', action='store_true', help="Use high dimension (i.e. > 2) parameter space. (default: False)")
parser.add_argument('--use_solver', action='store_true', help='Set uMDSAC agent to use a solver (i.e. selects uMDSAC-v2 instead of uMDSAC-v1) (default: False)')
parser.add_argument('--use_her', action='store_true', help='Set algorithm to use Hindsight Experience Replay (default: False)')
##
args = parser.parse_args()

###########################################
if args.n_runs == 1:
    ## Automatically set the first_run_idx
    ##
    ## Automatically set the Cuda ID:
    #from torch.cuda import device_count
    N_GPUS = 4#device_count()
    args.cuda_id = (args.first_run_idx + 0) % N_GPUS

###########################################
cmdResult = 0
IS_TRAIN = True
hidden_layers = [64, 64]#[128, 128]#
batch_size = 64#128

###########################################
def get_sac_cmd(algo, seed):
    n_steps = 1000000
    tau = 0.005
    beta = 0.
    max_alpha = 0.3
    batch_size = 256
    hidden_layers = [256, 256]
    num_evals = 10
    ###########################################
    if "HalfCheetah" in args.env_name:
        warmup = 10000
    elif "Hopper" in args.env_name:
        warmup = 1000
    elif "Ant" in args.env_name:
        #batch_size = 512
        warmup = 10000
    elif "Humanoid" in args.env_name:
        warmup = 1000
    elif "Walker" in args.env_name:
        warmup = 1000
    elif "Hexapod" in args.env_name:
        warmup = 1000
        num_evals = 1
    elif "DClaw" in args.env_name:
        warmup = 1000
        #n_steps = 500000
    elif "Pendulum" in args.env_name:
        warmup = batch_size
        n_steps = 60000
        max_alpha = 0.1
        batch_size = 64
        hidden_layers = [64, 64]
    else:
        raise NotImplementedError
    ################
    if "sac" == algo or "sac_no_target" == algo:
        cmd = "python main.py --algo {} --env_name {} --seed {} --batch_size {} \
                   --hidden_layers {} {} --warmup {} --total_num_steps {} --tau {} \
                   --number_of_updates {} --alpha {} --automatic_entropy_tuning \
                   --cuda_id {} --num_evals {} --cuda --verbose".format(algo, args.env_name, seed + args.first_run_idx,#--save_best
                   batch_size, *hidden_layers, warmup, n_steps, tau, 1, 0.2, args.cuda_id, num_evals)
    elif "arsac" in algo:
        cmd = "python main.py --algo {} --env_name {} --seed {} --batch_size {} \
                   --hidden_layers {} {} --warmup {} --total_num_steps {} --tau {} \
                   --number_of_updates {} --alpha {} --uncertainty_alpha {} --ratio {} \
                    --automatic_entropy_tuning --cuda_id {} --cuda --verbose".format(algo, args.env_name, seed + args.first_run_idx,# --save_best
                   batch_size, *hidden_layers, warmup, n_steps, tau, 1, 0.2, args.uncertainty_alpha, args.ratio, args.cuda_id)
        if "mdv" in algo:
            cmd += " --alpha_type {}".format(args.alpha_type)
    elif "mdsac" in algo or "drsac" in algo or "sirsa" in algo:
        cmd = "python main.py --algo {} --env_name {} --seed {} --batch_size {} \
                   --hidden_layers {} {} --warmup {} --total_num_steps {} --tau {} \
                   --number_of_updates {} --alpha {} --n_regions {} --automatic_entropy_tuning \
                   --cuda_id {} --cuda --verbose".format(algo, args.env_name, seed + args.first_run_idx,# --save_best
                   batch_size, *hidden_layers, warmup, n_steps, tau, 1, 0.2, args.n_regions, args.cuda_id)
        cmd += " --rel_mass_range {},{}".format(*args.rel_mass_range)
        if args.resample:
            cmd += " --resample"
        if args.continuous:
            cmd += " --continuous"
        if args.no_encoding:
            cmd += " --no_encoding"
        if args.use_ut:
            cmd += " --use_ut"
        if args.explicit_scal:
            cmd += " --explicit_scal"
        if args.just_run_test_target:
            cmd += " --just_run_test_target"
        if args.multi_dim:
            cmd += " --multi_dim"
        if args.use_her:
            cmd += " --use_her"
        if args.use_solver:
            cmd += " --use_solver"
    else:
        raise NotImplementedError
    #####
    return cmd

###########################################
ALGOS = args.algos
for algo in ALGOS:
    seed = 0
    for seed in range(1, args.n_runs+1):
        print("############ SEED {} on GPU:{} ############".format(seed + args.first_run_idx, args.cuda_id))
        if "sac" in algo or "sirsa" in algo:
            cmd = get_sac_cmd(algo, seed)
        else:
            print("{} not a valid algorithm name".format(algo))
            raise NotImplementedError
        ####
        if IS_TRAIN:
            proc = sp.Popen(cmd, shell=True)
            cmdResult = proc.wait() #catch return code
            print(cmdResult)
        if cmdResult == 1:
            break
    if cmdResult == 1:
        break
    #####################################
    #### TESTING
    #####################################
    if seed == 0:
        continue
    elif cmdResult == 1:
        seed = seed - 1
    ######
    print(algo)
    algo_name = algo
    ######
    print("###################")
