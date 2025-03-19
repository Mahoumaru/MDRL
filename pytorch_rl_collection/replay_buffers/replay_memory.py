import random
import numpy as np
from pytorch_rl_collection.utils import RandomDict
from hilbertcurve.hilbertcurve import HilbertCurve
from collections import deque
from collections import defaultdict

from torch import FloatTensor, cat as torch_cat

###########################################
class ReplayMemory:
    def __init__(self, capacity, seed, window_length=None):
        #random.seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, info=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, info)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #batch = random.sample(self.buffer, batch_size)
        batch = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, info = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, info

    def reset(self):
        del self.buffer[:]
        self.position = 0
        self.rng = random.Random(self.seed)

    def __len__(self):
        return len(self.buffer)

###########################################
## Taken from https://github.com/maywind23/LSTM-RL/blob/master/common/buffers.py
class ReplayMemoryLSTM:
    """
    Replay buffer for agent with LSTM network additionally using previous action, can be used
    if the hidden states are not stored (arbitrary initialization of lstm for training).
    And each sample contains the whole episode instead of a single step.
    """
    def __init__(self, capacity, seed, window_length=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack,
                                                      zip(*batch))  # stack for each element
        '''
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, last_action, reward, next_state, done

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

###########################################
class TruncatedReplayMemoryLSTM:
    def __init__(self, capacity, seed, max_sequence_length, max_episode_steps,
            adjust_sampling_lengths=True, adjustment_percentage=0.5, window_length=None
        ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.capacity = capacity
        ####
        self.lengths = None
        self.cumul_lenghts = None
        ####
        self.buffer = None
        self.position = 0
        self.max_sequence_length = max_sequence_length # truncation length
        self.max_episode_steps = max_episode_steps
        assert self.max_sequence_length < self.max_episode_steps
        ####
        # Whether or not to adjust the sampling indices so that we never
        # sample a sub-sequence that is too short
        self.adjust_sampling_lengths = adjust_sampling_lengths
        # Adjustment percentage between 0 and 1, with 1 being full adjustment
        # and 0 being no adjustment at all
        if self.adjust_sampling_lengths == False:
            self.adjustment_percentage = 0
        else:
            self.adjustment_percentage = adjustment_percentage
        ##
        if self.adjustment_percentage == 0.:
            self.adjust_sampling_lengths = False
        ####
        self.adjustment_length = int(self.adjustment_percentage * self.max_sequence_length)
        self.additional_padding = int((1. - self.adjustment_percentage) * self.max_sequence_length)
        ####
        self.episode_capacity = max(1, int(self.capacity / (self.max_episode_steps - self.adjustment_length)))
        ####
        self.key_to_dim = None

    def push(self, state, action, last_action, reward, next_state, done):
        episode_length = state.shape[0]
        assert episode_length == action.shape[0]
        if self.buffer is None:
            state_dim = state.shape[-1]
            action_dim = action.shape[-1]
            #####
            self.lengths = np.array([None])
            #####
            self.key_to_dim = {
                "states": state_dim,
                "actions": action_dim,
                "last_actions": action_dim,
                "rewards": 1,
                "next_states": state_dim,
                "dones": 1,
            }
            #####
            self.buffer = {
                "states": np.zeros((1, self.max_episode_steps+self.additional_padding, state_dim)),
                "actions": np.zeros((1, self.max_episode_steps+self.additional_padding, action_dim)),
                "last_actions": np.zeros((1, self.max_episode_steps+self.additional_padding, action_dim)),
                "rewards": np.zeros((1, self.max_episode_steps+self.additional_padding, 1)),
                "next_states": np.zeros((1, self.max_episode_steps+self.additional_padding, state_dim)),
                "dones": np.zeros((1, self.max_episode_steps+self.additional_padding, 1)),
            }
        elif self.buffer["states"].shape[0] < self.episode_capacity:
            ######
            self.lengths = np.concatenate([self.lengths, np.array([None])])
            ######
            for key, val in self.buffer.items():
                self.buffer[key] = np.concatenate([
                    val,
                    np.zeros((1, self.max_episode_steps+self.additional_padding, self.key_to_dim[key]))
                ],axis=0)
        else:
            self.buffer["states"][self.position] *= 0.
            self.buffer["actions"][self.position] *= 0.
            self.buffer["last_actions"][self.position] *= 0.
            self.buffer["rewards"][self.position] *= 0.
            self.buffer["next_states"][self.position] *= 0.
            self.buffer["dones"][self.position] *= 0.
        #########
        self.lengths[self.position] = episode_length
        if self.adjust_sampling_lengths:
            adjusted_lengths = np.clip(self.lengths - self.adjustment_length + 1, a_min=1, a_max=None)
            self.cumul_lenghts = np.cumsum(adjusted_lengths)
        else:
            self.cumul_lenghts = np.cumsum(self.lengths)
        #########
        self.buffer["states"][self.position, :episode_length] = state
        self.buffer["actions"][self.position, :episode_length] = action
        self.buffer["last_actions"][self.position, :episode_length] = last_action
        self.buffer["rewards"][self.position, :episode_length] = reward.reshape(-1, 1)
        self.buffer["next_states"][self.position, :episode_length] = next_state
        self.buffer["dones"][self.position, :episode_length] = done.reshape(-1, 1)
        #########
        self.position = int((self.position + 1) % self.episode_capacity)  # as a ring buffer

    def sample(self, batch_size):
        flat_batch_idxs = self.rng.sample(list(range(self.cumul_lenghts[-1])), batch_size)
        #### Map flat indices to (row, col)
        rows = np.searchsorted(self.cumul_lenghts, flat_batch_idxs, side='right')
        # Compute the starting offset for each row:
        offsets = np.array(list(np.hstack(([0], self.cumul_lenghts[:-1]))))
        cols = flat_batch_idxs - offsets[rows]
        ####
        mask = (cols+self.max_sequence_length) <= self.lengths[rows]
        batch_ep_lengths = self.lengths[rows] - cols
        batch_ep_lengths[mask] = self.max_sequence_length
        ####
        idxs = [cols+i for i in range(self.max_sequence_length)]
        #print("Rows:", rows, rows.dtype)
        #print("Cols:", cols, cols.dtype)
        #print("idxs:", idxs)
        #print("offsets:", offsets, offsets.dtype)
        ## Transpose to put the batch dimension first! This is undone before
        ## passing into the LSTM/GRU and then redone again after. (see model_networks/model_networks.py)
        states_batch = self.buffer["states"][rows, idxs].transpose(1,0,2)
        actions_batch = self.buffer["actions"][rows, idxs].transpose(1,0,2)
        last_actions_batch = self.buffer["last_actions"][rows, idxs].transpose(1,0,2)
        rewards_batch = self.buffer["rewards"][rows, idxs].transpose(1,0,2)
        next_states_batch = self.buffer["next_states"][rows, idxs].transpose(1,0,2)
        dones_batch = self.buffer["dones"][rows, idxs].transpose(1,0,2)
        ####
        #print(states_batch.shape, actions_batch.shape, last_actions_batch.shape,
        #rewards_batch.shape, next_states_batch.shape, dones_batch.shape)
        assert states_batch.shape[0] == batch_size
        assert (batch_ep_lengths > 0.).all()
        batch_ep_lengths = list(batch_ep_lengths)
        assert len(batch_ep_lengths) == batch_size
        ####
        return states_batch, actions_batch, last_actions_batch, rewards_batch, next_states_batch, dones_batch, batch_ep_lengths

    def __len__(self):
        if self.cumul_lenghts is None:
            return 0
        else:
            return self.cumul_lenghts[-1]

###########################################
## Taken from https://github.com/maywind23/LSTM-RL/blob/master/common/buffers.py
class ReplayMemoryLSTM2:
    """
    Replay buffer for agent with LSTM network additionally storing previous action,
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity, seed, window_length=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = self.rng.sample(self.buffer, batch_size)
        lengths = []
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            lengths.append(len(state))
            s_lst.append(FloatTensor(np.array(state)))
            a_lst.append(FloatTensor(np.array(action)))
            la_lst.append(FloatTensor(np.array(last_action)))
            r_lst.append(FloatTensor(np.array(reward)))
            ns_lst.append(FloatTensor(np.array(next_state)))
            d_lst.append(FloatTensor(np.array(done)))
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch_cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch_cat(ho_lst, dim=-2).detach()
        ci_lst = torch_cat(ci_lst, dim=-2).detach()
        co_lst = torch_cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)
        #s_lst = np.array(s_lst)
        #a_lst = np.array(a_lst)
        #la_lst = np.array(la_lst)
        #r_lst = np.array(r_lst)
        #ns_lst = np.array(ns_lst)
        #d_lst = np.array(d_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, lengths

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

###########################################
## Taken from https://github.com/maywind23/LSTM-RL/blob/master/common/buffers.py
class ReplayMemoryGRU:
    """
    Replay buffer for agent with GRU network additionally storing previous action,
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """
    def __init__(self, capacity, seed, window_length=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst, d_lst=[],[],[],[],[],[],[],[]
        batch = self.rng.sample(self.buffer, batch_size)
        lengths = []
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state, done = sample
            lengths.append(len(state))
            s_lst.append(FloatTensor(np.array(state)))
            a_lst.append(FloatTensor(np.array(action)))
            la_lst.append(FloatTensor(np.array(last_action)))
            r_lst.append(FloatTensor(np.array(reward)))
            ns_lst.append(FloatTensor(np.array(next_state)))
            d_lst.append(FloatTensor(np.array(done)))
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch_cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch_cat(ho_lst, dim=-2).detach()
        #s_lst = np.array(s_lst)
        #a_lst = np.array(a_lst)
        #la_lst = np.array(la_lst)
        #r_lst = np.array(r_lst)
        #ns_lst = np.array(ns_lst)
        #d_lst = np.array(d_lst)

        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, lengths

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

###########################################
class MDReplayMemory:
    def __init__(self, capacity, seed, window_length=None):
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, alpha, varpi):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, alpha, varpi)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, alpha, varpi = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, alpha, varpi

    def __len__(self):
        return len(self.buffer)

###########################################
class DiversedMDReplayMemory(MDReplayMemory):
    def __init__(self, capacity, seed, dim, window_length=None, hilbert_order=4):
        super(DiversedMDReplayMemory, self).__init__(capacity, seed, window_length)
        self.current_size = 0
        self.hilbert_buffer = RandomDict(seed)
        self.regions_idxs = [0]
        #####
        self.dim = dim
        self.hilbert_order = hilbert_order  # Order of the Hilbert curve (2^p bins per dimension)
        self.bin_size = 1. / (2.**hilbert_order)  # Size of each bin
        self.hilbert_curve = HilbertCurve(hilbert_order, dim)

    def _is_full(self):
        return len(self.buffer) >= self.capacity

    def get_hilbert_index(self, sample):
        scaled_sample = [int(coord // self.bin_size) for coord in sample]
        hilbert_index = self.hilbert_curve.distance_from_point(scaled_sample)
        return hilbert_index

    def push(self, state, action, reward, next_state, done, alpha, varpi):
        hilbert_index = self.get_hilbert_index(alpha)
        if not self._is_full():
            self.buffer.append(None)
        else:
            # If we're full, we need to delete the oldest entry first
            old_idx = self.get_hilbert_index(self.buffer[self.position][5])
            old_index_deque = self.hilbert_buffer[old_idx]
            old_index_deque.popleft()
            if not old_index_deque:
                #self.hilbert_buffer.pop(old_idx)
                del self.hilbert_buffer[old_idx]
        #######
        self.buffer[self.position] = (state, action, reward, next_state, done, alpha, varpi)
        if hilbert_index not in self.hilbert_buffer:
            self.hilbert_buffer[hilbert_index] = deque()
            """if self.regions_idxs[-1] < self.hilbert_buffer.last_index:
                self.regions_idxs.append(self.hilbert_buffer.last_index)"""
        self.hilbert_buffer[hilbert_index].append(self.position)
        #######
        self.position = (self.position + 1) % self.capacity

    def _sample_hilbert_index(self):
        return self.hilbert_buffer.random_key()

    def sample(self, batch_size):
        #batch = self.rng.sample(self.buffer, batch_size)
        orig_batch_size = batch_size
        last_index = self.hilbert_buffer.last_index + 1
        self.regions_idxs = list(range(last_index))
        batch_regions = self.rng.sample(self.regions_idxs, min(batch_size, last_index))
        #####
        size = len(batch_regions)
        indi_size = batch_size // size
        #####
        batch_regions_hilert_idxs = []
        individual_batch_sizes = []
        regions_with_spare_space = []
        for i, idx in enumerate(batch_regions):
            hilbert_idx = self.hilbert_buffer.get_key_from_idx(idx)
            batch_regions_hilert_idxs.append(hilbert_idx)
            try:
                N = len(self.hilbert_buffer[hilbert_idx])
            except KeyError:
                print(idx, hilbert_idx)
                print(self.hilbert_buffer.keys())
                raise KeyError
            if N > indi_size:
                regions_with_spare_space.append((i, N - indi_size))
            else:
                indi_size = N
            individual_batch_sizes.append(indi_size)
        #####
        batch_size = batch_size - sum(individual_batch_sizes)
        while batch_size > 0:
            indi_size = batch_size // len(regions_with_spare_space)
            min_idx, min_N = min(regions_with_spare_space, key=lambda item: item[1])
            regions_with_spare_space.remove((min_idx, min_N))
            min_N = min(indi_size, min_N)
            individual_batch_sizes[min_idx] += min_N
            batch_size -= min_N
        #####
        assert sum(individual_batch_sizes) == orig_batch_size, "{} vs {}; {}, {}".format(sum(individual_batch_sizes), orig_batch_size, individual_batch_sizes, indi_size)
        #####
        assert len(batch_regions_hilert_idxs) == len(individual_batch_sizes), "{} vs {}".format(batch_regions_hilert_idxs, individual_batch_sizes)
        #####
        batch = []
        for key, n in zip(batch_regions_hilert_idxs, individual_batch_sizes):
            idx_deque = self.hilbert_buffer[key]
            s = self.rng.sample(idx_deque, n)
            if type(s) != list:
                s = [s]
            for p in s:
                batch.append(self.buffer[p])
        #####
        state, action, reward, next_state, done, alpha, varpi = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, alpha, varpi

    def __len__(self):
        return len(self.buffer)

###########################################
class ReplayMemorySet:
    def __init__(self, size, capacity, seed, window_length=None):
        self.rmset = []
        self.rng = random.Random(seed)
        self.size = size
        self.rm_idxs = []
        for i in range(size):
            self.rmset.append(ReplayMemory(capacity=capacity, seed=seed, window_length=window_length))
            self.rm_idxs.append(i)

    def push(self, indices, states, actions, rewards, next_states, dones, infos=None):
        j = 0
        if infos is None:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j])
                    j += 1
        else:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j], infos[j])
                    j += 1

    def sample(self, batch_size):
        #####
        indi_size = batch_size // self.size
        individual_batch_sizes = [indi_size for _ in range(self.size)]
        for i in range(batch_size - self.size * individual_batch_sizes[0]):
            individual_batch_sizes[i] += 1
        assert sum(individual_batch_sizes) == batch_size, "{} vs {}; {}, {}".format(sum(individual_batch_sizes), batch_size, individual_batch_sizes, indi_size)
        #####
        self.rng.shuffle(self.rm_idxs)
        for i in range(self.size):
            yield self.rmset[self.rm_idxs[i]].sample(individual_batch_sizes[i])

    def __len__(self):
        return sum([len(self.rmset[i]) for i in range(self.size)])

###########################################
class NumpyMDReplayMemory:
    def __init__(self, capacity, seed, window_length=None):
        #random.seed(seed)
        self.rng = random.Random(seed)
        self.capacity = capacity
        self.buffer = None
        self.position = 0

    def push(self, state, action, reward, next_state, done, alpha, varpi):
        new_entry = np.array([(state, action, reward, next_state, done, alpha, varpi)], dtype=object)
        if self.buffer is None:
            self.buffer = new_entry
        elif self.buffer.shape[0] < self.capacity:
            self.buffer = np.concatenate((self.buffer, new_entry))
        else:
            self.buffer[self.position] = new_entry
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_idxs = self.rng.sample(list(range(self.buffer.shape[0])), batch_size)
        state, action, reward, next_state, done, alpha, varpi = self.buffer[batch_idxs].T
        return np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), np.array(list(done)), np.array(list(alpha)), np.array(list(varpi)), np.array(batch_idxs)

    def get_previous_transition(self, current_idxs):
        state, action, reward, next_state, done, alpha, varpi = self.buffer[current_idxs-1].T
        state, action, reward, next_state, done, alpha, varpi = np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), np.array(list(done)), np.array(list(alpha)), np.array(list(varpi))
        ####
        return state, action, reward, next_state, done, alpha, varpi

    def __len__(self):
        return 0 if self.buffer is None else self.buffer.shape[0]

###########################################
class MDReplayMemorySet:
    def __init__(self, size, capacity, seed, dim=None, window_length=None, diversified=False, numpy_buffer=False, union_sampling=False):
        self.rmset = []
        self.rng = random.Random(seed)
        self.size = size
        self.rm_idxs = []
        self.diversified = diversified
        self.numpy_buffer = numpy_buffer
        self.union_sampling = union_sampling
        args_dict = {}
        if diversified:
            MemCls = DiversedMDReplayMemory
            assert dim is not None, "DiversedMDReplayMemory requires a valid space dimemsion. Got 'None' instead."
            args_dict = {"dim": dim}
        elif numpy_buffer:
            MemCls = NumpyMDReplayMemory
        else:
            MemCls = MDReplayMemory
        for i in range(size):
            self.rmset.append(MemCls(capacity=capacity, seed=seed, window_length=window_length, **args_dict))
            self.rm_idxs.append(i)
        #####
        if self.union_sampling:
            self.sample = self.sample_union
            self.get_previous_transition = self.get_union_previous_transition
        else:
            self.sample = self.sample_all
            self.get_previous_transition = self.get_all_previous_transition

    def push(self, indices, states, actions, rewards, next_states, dones, kappas=None, varpis=None):
        j = 0
        if varpis is None and kappas is not None:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j], kappas[j], None)
                    j += 1
        elif varpis is not None and kappas is None:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j], None, varpis[j])
                    j += 1
        elif varpis is None and kappas is None:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j], None, None)
                    j += 1
        else:
            for i, rm in enumerate(self.rmset):
                if i in indices:
                    rm.push(states[j], actions[j], rewards[j], next_states[j], dones[j], kappas[j], varpis[j])
                    j += 1

    def sample_union(self, batch_size):
        if self.numpy_buffer:
            buffer = None
            for i in range(self.size):
                if buffer is None and self.rmset[i].buffer is not None:
                    buffer = self.rmset[i].buffer
                elif self.rmset[i].buffer is not None:
                    buffer = np.concatenate((buffer, self.rmset[i].buffer))
            assert buffer.shape[0] >= batch_size, "{} vs {}".format(buffer.shape, batch_size)
            batch_idxs = self.rng.sample(list(range(buffer.shape[0])), batch_size)
            state, action, reward, next_state, done, alpha, varpi = buffer[batch_idxs].T
            return np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), np.array(list(done)), np.array(list(alpha)), np.array(list(varpi)), np.array(batch_idxs)
        else:
            buffer = []
            for i in range(self.size):
                buffer += self.rmset[i].buffer
            assert len(buffer) >= batch_size, "{} vs {}".format(len(buffer), batch_size)
            batch = self.rng.sample(buffer, batch_size)
            state, action, reward, next_state, done, alpha, varpi = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done, alpha, varpi

    def sample_all(self, batch_size):
        #####
        indi_size = batch_size // self.size
        individual_batch_sizes = [indi_size for _ in range(self.size)]
        for i in range(batch_size - self.size * individual_batch_sizes[0]):
            individual_batch_sizes[i] += 1
        assert sum(individual_batch_sizes) == batch_size
        #####
        self.rng.shuffle(self.rm_idxs)
        for i in range(self.size):
            #state, action, reward, next_state, done, alpha, varpi = rm.sample(batch_size)
            yield self.rmset[self.rm_idxs[i]].sample(individual_batch_sizes[i])
            """
            rm = self.rmset[self.rm_idxs[i]]
            if len(rm) < batch_size:
                yield (None, None, None, None, None, None, None)
            else:
                yield rm.sample(batch_size)
            """

    def get_union_previous_transition(self, current_idxs):
        batch_size = current_idxs.shape[0]
        #####
        if self.numpy_buffer:
            buffer = None
            for i in range(self.size):
                if buffer is None and self.rmset[i].buffer is not None:
                    buffer = self.rmset[i].buffer
                elif self.rmset[i].buffer is not None:
                    buffer = np.concatenate((buffer, self.rmset[i].buffer))
            assert buffer.shape[0] >= batch_size, "{} vs {}".format(buffer.shape, batch_size)
            state, action, reward, next_state, done, alpha, varpi = buffer[current_idxs-1].T
            return np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), np.array(list(done)), np.array(list(alpha)), np.array(list(varpi))
        else:
            buffer = []
            for i in range(self.size):
                buffer += self.rmset[i].buffer
            assert len(buffer) >= batch_size, "{} vs {}".format(len(buffer), batch_size)
            state, action, reward, next_state, done, alpha, varpi = np.array(buffer)[current_idxs-1].T
            state, action, reward, next_state, done, alpha, varpi = np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), np.array(list(done)), np.array(list(alpha)), np.array(list(varpi))
            ####
            return state, action, reward, next_state, done, alpha, varpi

    def get_all_previous_transition(self, current_idxs):
        batch_size = current_idxs.shape[0]
        #####
        indi_size = batch_size // self.size
        individual_batch_sizes = [indi_size for _ in range(self.size)]
        for i in range(batch_size - self.size * individual_batch_sizes[0]):
            individual_batch_sizes[i] += 1
        assert sum(individual_batch_sizes) == batch_size
        #####
        total_batch_size = 0
        for i in range(self.size):
            #state, action, reward, next_state, done, alpha, varpi = rm.sample(batch_size)
            yield self.rmset[self.rm_idxs[i]].get_previous_transition(current_idxs[total_batch_size:total_batch_size+individual_batch_sizes[i]])
            total_batch_size += individual_batch_sizes[i]

    def __len__(self):
        return sum([len(self.rmset[i]) for i in range(self.size)])

###########################################
class MDReplayMemoryNumpy:
    def __init__(self, capacity, seed, window_length=None):
        #random.seed(seed)
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.state_buffer = np.empty(self.capacity, dtype=object) # all entries are None
        self.action_buffer = np.empty(self.capacity, dtype=object)
        self.reward_buffer = np.empty(self.capacity, dtype=object)
        self.next_state_buffer = np.empty(self.capacity, dtype=object)
        self.done_buffer = np.empty(self.capacity, dtype=object)
        self.alpha_buffer = np.empty(self.capacity, dtype=object)
        self.varpi_buffer = np.empty(self.capacity, dtype=object)
        self.length = 0
        self.position = 0

    def push(self, state, action, reward, next_state, done, alpha, varpi):
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.alpha_buffer[self.position] = alpha
        self.varpi_buffer[self.position] = varpi
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.choice(self.length, batch_size, replace=False)
        state, action, reward, next_state, done, alpha, varpi = self.state_buffer[idx], self.action_buffer[idx], \
             self.reward_buffer[idx], self.next_state_buffer[idx], self.done_buffer[idx], self.alpha_buffer[idx], \
             self.varpi_buffer[idx]
        return np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), \
            np.array(list(done)), np.array(list(alpha)), np.array(list(varpi))

    def __len__(self):
        return self.length


if __name__ == "__main__":
    #import numpy as np
    #import gymnasium as gym
    from itertools import count
    from pytorch_rl_collection.envs_utils.env_set import MDEnvSet
    #from pytorch_rl_collection.replay_buffers.replay_memory import MDReplayMemory, DiversedMDReplayMemory, MDReplayMemorySet

    seed = 0
    env_name = "Hopper-v3"
    env = MDEnvSet(env_name=env_name, multi_dim=False, n_regions=1, seed=seed, use_encoder=False,
          add_noise=False, use_quasi_random=False
    )

    np.random.seed(seed)
    env.seed(seed)

    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    domains_dim = env.subdomains.domain_dim

    capacity = 100
    rm1 = MDReplayMemory(capacity, seed)
    rm2 = DiversedMDReplayMemory(capacity, seed, domains_dim)
    rm3 = MDReplayMemorySet(1, capacity, seed, domains_dim, diversified=True)

    n_episodes = 1
    for ep in range(n_episodes):
        print("++ Episode", ep+1)
        state = env.reset()
        I = env.sample_env_params(use_varpi=False)

        episode_rewards = np.zeros(env.n_regions)
        episode_loss = 0.
        done = False
        for t in count(0):
            pre_not_dones = ~env.get_dones()
            valid_idxs = env.get_stillActiveIdxs()
            ##
            action = np.random.uniform(-1., 1., actions_dim)
            action = action.reshape(1, -1).repeat(state.shape[0], axis=0)
            ##
            next_state, reward, done, _, _ = env.step(action)
            ##
            episode_rewards[pre_not_dones] += reward
            ##
            mask = np.ones(done.shape) if t+1 == env._max_episode_steps else (~done).astype(float)
            rm1.push(state[0], action[0], reward[0], next_state[0], done[0], I[0], None)
            rm2.push(state[0], action[0], reward[0], next_state[0], done[0], I[0], None)
            rm3.push([0], state, action, reward, next_state, done, I, [None])
            ####
            if (done == True).all() or (t+1) == env._max_episode_steps:
                break

        print(len(rm1), len(rm2), len(rm3))
        _, _, _, _, _, kappas, _ = rm1.sample(10); print(kappas)
        _, _, _, _, _, kappas, _ = rm2.sample(10); print(kappas)
        for _, _, _, _, _, kappas, _ in rm3.sample(10):
            print(kappas)

        print(rm2.hilbert_buffer._random_vector)
        print(rm2.hilbert_buffer._keys)
