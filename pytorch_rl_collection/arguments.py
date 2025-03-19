from pytorch_rl_collection.sac_codes.sac_arguments_file import *

###########################################
def get_args(algo_name="ddpg"):
    if "mdsac" in algo_name or "drsac" in algo_name or "sirsa" in algo_name:
        algo_name = "mdsac"
    #else:
    #    raise NotImplementedError
    return fct[algo_name]()

###########################################
fct = {
  #####
  "sac": sac_arguments,
  "mdsac": mdsac_arguments,
  "sac_rnn": rnn_sac_arguments,
  "drsac_rnn": mdsac_arguments,
  #####
}
