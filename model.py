import numpy as np, torch, torch.nn as nn, torch.optim as optim

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((max_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, input_shape), dtype=np.float32)
        act_dtype = np.int8 if discrete else np.float32
        self.action_memory = np.zeros((max_size, n_actions), dtype=act_dtype)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        if self.discrete:
            a = np.zeros(self.action_memory.shape[1])
            a[action] = 1.0
            self.action_memory[idx] = a
        else:
            self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
        )

class QNet(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DDQNAgent:
    def __init__(
        self,
        alpha,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        epsilon_dec=0.999995,
        epsilon_end=0.10,
        mem_size=25000,
        fname="ddqn_model.pt",
        replace_target=25,
        device=None,
    ):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.eval_net = QNet(input_dims, n_actions).to(self.device)
        self.target_net = QNet(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.model_file = fname

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        state = torch.tensor([state], dtype=torch.float32, device=self.device)
        actions = self.eval_net(state)
        return torch.argmax(actions).item()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_ = torch.tensor(states_, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        indices = torch.arange(self.batch_size)
        action_indices = torch.matmul(actions, torch.arange(self.n_actions, device=self.device, dtype=torch.int64))

        q_pred = self.eval_net(states)[indices, action_indices]
        q_next = self.target_net(states_)
        q_eval = self.eval_net(states_)
        max_actions = torch.argmax(q_eval, dim=1)
        q_target = rewards + self.gamma * q_next[indices, max_actions] * dones

        loss = self.loss_fn(q_pred, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target == 0:
            self.update_network_parameters()

    def update_network_parameters(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def save_model(self):
        torch.save(self.eval_net.state_dict(), self.model_file)

    def load_model(self):
        self.eval_net.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.target_net.load_state_dict(self.eval_net.state_dict())