import random
import numpy as np

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
        #batch = random.sample(self.buffer, batch_size)
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
    def __init__(self, size, capacity, seed, window_length=None, stratified=False, numpy_buffer=False, union_sampling=False):
        self.rmset = []
        self.rng = random.Random(seed)
        self.size = size
        self.rm_idxs = []
        self.stratified = stratified
        self.numpy_buffer = numpy_buffer
        self.union_sampling = union_sampling
        if stratified:
            MemCls = StratifiedMDReplayMemory
        elif numpy_buffer:
            MemCls = NumpyMDReplayMemory
        else:
            MemCls = MDReplayMemory
        for i in range(size):
            self.rmset.append(MemCls(capacity=capacity, seed=seed, window_length=window_length))
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
        #batch = random.sample(self.buffer, batch_size)
        idx = self.rng.choice(self.length, batch_size, replace=False)
        state, action, reward, next_state, done, alpha, varpi = self.state_buffer[idx], self.action_buffer[idx], \
             self.reward_buffer[idx], self.next_state_buffer[idx], self.done_buffer[idx], self.alpha_buffer[idx], \
             self.varpi_buffer[idx]
        return np.array(list(state)), np.array(list(action)), np.array(list(reward)), np.array(list(next_state)), \
            np.array(list(done)), np.array(list(alpha)), np.array(list(varpi))

    def __len__(self):
        return self.length
