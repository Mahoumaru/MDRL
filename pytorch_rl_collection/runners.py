################################################################################
##### SAC
from pytorch_rl_collection.sac_codes.sac.sac_algo import run_sac_algorithm
##### Domain Randomization SAC
from pytorch_rl_collection.sac_codes.drsac.drsac_algo import run_drsac_algorithm
##### Scalarized MD-SAC
from pytorch_rl_collection.sac_codes.umdsac.umdsac_algo import run_umdsac_algorithm
##### Conditioned Degenerate MD-SAC
from pytorch_rl_collection.sac_codes.cmdsac.cmdsac_algo import run_cmdsac_algorithm
##### Envelope Degenerate MD-SAC
from pytorch_rl_collection.sac_codes.emdsac.emdsac_algo import run_emdsac_algorithm
################################################################################
##### SIRSA
from pytorch_rl_collection.sac_codes.sirsa.sirsa_algo import run_sirsa_algorithm
##### SMD SIRSA
from pytorch_rl_collection.sac_codes.umd_sirsa.umd_sirsa_algo import run_umd_sirsa_algorithm
################################################################################
##### JOINT ALGORITHM
from pytorch_rl_collection.sac_codes.mdsac.mdsac_algo import run_mdsac_algorithm

################################################################################
##### Domain Randomization SAC
from pytorch_rl_collection.sac_codes.sdrsac.sdrsac_algo import run_sdrsac_algorithm

################################################################################
##### RNN SAC
from pytorch_rl_collection.sac_codes.sac_rnn.sac_rnn_algo import run_sac_rnn_algorithm
##### RNN Domain Randomization SAC
from pytorch_rl_collection.sac_codes.drsac_rnn.drsac_rnn_algo import run_drsac_rnn_algorithm

################################################################################
RUNNER_FUNCTIONS_DICT = {
   ###
   "sac": run_sac_algorithm,
   "drsac": run_drsac_algorithm,
   "umdsac": run_umdsac_algorithm,
   "cmdsac": run_cmdsac_algorithm,
   "emdsac": run_emdsac_algorithm,
   ###
   "sirsa": run_sirsa_algorithm,
   "umd_sirsa": run_umd_sirsa_algorithm,
   ###
   "sdrsac": run_sdrsac_algorithm,
   ###
   "sac_rnn": run_sac_rnn_algorithm,
   "drsac_rnn": run_drsac_rnn_algorithm,
}

################################################################################
def get_runner(algo_name="ddpg"):
    #return RUNNER_FUNCTIONS_DICT[algo_name]
    if algo_name == "sac":
        return run_sac_algorithm
    elif algo_name == "sac_rnn":
        return run_sac_rnn_algorithm
    elif algo_name == "drsac_rnn":
        return run_drsac_rnn_algorithm
    #elif "sdrsac" in algo_name:
    #    return run_sdrsac_algorithm
    elif "sac" in algo_name or "umd" in algo_name:
        return run_mdsac_algorithm
    elif algo_name == "sirsa":
        return run_sirsa_algorithm
    else:
        raise NotImplementedError
