import argparse

###########################################
def sac_arguments():
    parser = argparse.ArgumentParser(description='PyTorch RL Collection SAC Arguments')
    parser.add_argument('--algo', default='sac', type=str)
    ## Used by the runner, i.e. the run_sac_algorithm function in sac_algo.py
    parser.add_argument('--env_name', default='Pendulum-v0', type=str, help='Open-ai gym environment (default: Pendulum-v0)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed (default: 0)')
    ## Used in the sac algorithm
    #parser.add_argument('--total_num_episodes', default=1000, type=int, help='Total number of episodes (default: 1000)')
    parser.add_argument('--total_num_steps', default=1000000, type=int, help='Total number of episodes (default: 1000000)')
    parser.add_argument('--number_of_updates', default=1, type=int, help='Number of updates per steps (default: 1)')
    parser.add_argument('--batch_size', default=64, type=int, help='Minibatch size (default: 64)')
    parser.add_argument('--save_best', action='store_true', help='Set the algorithm to save the best model at each evaluation (default: False)')
    parser.add_argument('--num_evals', default=1, type=int, help='If save_best is enabled, defines the number of evaluation to perform to assess the model performance (default: 10)')
    ## Used by the sac agent
    parser.add_argument('--warmup', default=100, type=int, help='Time without training but only filling the replay memory (default: 100)')
    parser.add_argument('--hidden_layers', default=[400, 300], nargs='+', type=int, help='Number of neurons unit per hidden layer (default: [400, 300])')
    parser.add_argument('--init_w', default=0.003, type=float, help='Bound on the parameters of the output layer, i.e. params ~ U(-init_w, init_w) (default: 0.003)')
    parser.add_argument('--actor_lr', default=0.0003, type=float, help='Actor network learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', default=0.0003, type=float, help='Critic network learning rate (default: 0.001)')
    parser.add_argument('--entropy_temp_lr', default=0.0003, type=float, help='Entropy temperature alpha learning rate (default: 0.001)')
    parser.add_argument('--rmsize', default=1000000, type=int, help='Replay memory size (default: 1000000)')
    parser.add_argument('--window_length', default=1, type=int, help='Size of the recent observations in the replay memory (default: 1)')
    parser.add_argument('--alpha', default=0.2, type=float, help='Temperature parameter which determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', action='store_true', help='Automatically adjust alpha (default: False)')
    parser.add_argument('--tau', default=0.005, type=float, help='Moving average decay factor for target network (default: 0.001)')
    parser.add_argument('--discount', default=0.99, type=float, help='Reward discount factor for the return (default: 0.99)')
    parser.add_argument('--cuda', action='store_true', help='Set the agent to use CUDA (default: False)')
    parser.add_argument('--cuda_id', default=0, type=int, help='Set GPU id to use with CUDA (default: 0)')
    parser.add_argument('--verbose', action='store_true', help='Print agent details, i.e. networks structures, etc. (default: False)')

    return parser.parse_args()

###########################################
def mdsac_arguments():
    parser = argparse.ArgumentParser(description='PyTorch RL Collection MD-SAC Arguments')
    parser.add_argument('--algo', default='mdvarsac', type=str)
    ## Used by the runner, i.e. the run_sac_algorithm function in sac_algo.py
    parser.add_argument('--env_name', default='Pendulum-v0', type=str, help='Open-ai gym environment (default: Pendulum-v0)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed (default: 0)')
    parser.add_argument('--multi_dim', action='store_true', help="Use high dimension (i.e. > 2) parameter space. (default: False)")
    ## Used in the sac algorithm
    #parser.add_argument('--total_num_episodes', default=1000, type=int, help='Total number of episodes (default: 1000)')
    parser.add_argument('--total_num_steps', default=1000000, type=int, help='Total number of episodes (default: 1000000)')
    parser.add_argument('--number_of_updates', default=1, type=int, help='Number of updates per steps (default: 1)')
    parser.add_argument('--batch_size', default=64, type=int, help='Minibatch size (default: 64)')
    parser.add_argument('--save_best', action='store_true', help='Set the algorithm to save the best model at each evaluation (default: False)')
    parser.add_argument('--num_evals', default=1, type=int, help='If save_best is enabled, defines the number of evaluation to perform to assess the model performance (default: 10)')
    parser.add_argument('--resample', action='store_true', help='Re-sample the environment at each time step. (default: False)')
    parser.add_argument('--just_run_test_target', action='store_true', help='Only evaluate trained policy. (default: False)')
    parser.add_argument('--use_her', action='store_true', help='Set algorithm to use Hindsight Experience Replay (default: False)')
    ## Used by the sac agent
    parser.add_argument('--warmup', default=100, type=int, help='Time without training but only filling the replay memory (default: 100)')
    parser.add_argument('--rel_mass_range', default=[0.5, 2.], type=lambda s: [float(item) for item in s.split(',')], help='Range of relative mass (default: [0.5, 2])')
    parser.add_argument('--hidden_layers', default=[400, 300], nargs='+', type=int, help='Number of neurons unit per hidden layer (default: [400, 300])')
    parser.add_argument('--init_w', default=0.003, type=float, help='Bound on the parameters of the output layer, i.e. params ~ U(-init_w, init_w) (default: 0.003)')
    parser.add_argument('--actor_lr', default=0.0003, type=float, help='Actor network learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', default=0.0003, type=float, help='Critic network learning rate (default: 0.001)')
    parser.add_argument('--entropy_temp_lr', default=0.0003, type=float, help='Entropy temperature alpha learning rate (default: 0.001)')
    parser.add_argument('--rmsize', default=1000000, type=int, help='Replay memory size (default: 1000000)')
    parser.add_argument('--window_length', default=1, type=int, help='Size of the recent observations in the replay memory (default: 1)')
    parser.add_argument('--alpha', default=0.2, type=float, help='Temperature parameter which determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', action='store_true', help='Automatically adjust alpha (default: False)')
    parser.add_argument('--tau', default=0.005, type=float, help='Moving average decay factor for target network (default: 0.001)')
    parser.add_argument('--discount', default=0.99, type=float, help='Reward discount factor for the return (default: 0.99)')
    parser.add_argument('--cuda', action='store_true', help='Set the agent to use CUDA (default: False)')
    parser.add_argument('--cuda_id', default=0, type=int, help='Set GPU id to use with CUDA (default: 0)')
    parser.add_argument('--verbose', action='store_true', help='Print agent details, i.e. networks structures, etc. (default: False)')
    parser.add_argument('--use_ut', action='store_true', help='Set agent to use the Unscented Transform in the actor update (default: False)')
    parser.add_argument('--explicit_scal', action='store_true', help='Whether to use explicit scalarization for the critic (default: False)')
    parser.add_argument('--n_osi', default=4, type=int, help='System identification model ensemble size (default: 4)')
    ## Used by both sac agent and algo
    parser.add_argument('--n_regions', default=10, type=int, help='Number of subdomains (default: 10)')
    parser.add_argument('--continuous', action='store_true', help='Run with continuous varpi distribution. (default: False)')
    parser.add_argument('--no_encoding', action='store_true', help="Don't employ the encoded parameters. (default: False)")
    parser.add_argument('--use_solver', action='store_true', help='Set uMDSAC agent to use a solver (i.e. selects uMDSAC-v2 instead of uMDSAC-v1) (default: False)')

    return parser.parse_args()
