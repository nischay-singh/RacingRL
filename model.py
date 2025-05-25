import numpy as np, torch, torch.nn as nn, torch.optim as optim

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size   = max_size
        self.mem_cntr   = 0
        self.discrete   = discrete
        self.state_memory      = np.zeros((max_size, input_shape), dtype=np.float32)
        self.new_state_memory  = np.zeros((max_size, input_shape), dtype=np.float32)
        self.action_memory     = np.zeros(max_size, dtype=np.int8)
        self.reward_memory     = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory   = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx]     = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx]    = action
        self.reward_memory[idx]    = reward
        self.terminal_memory[idx]  = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch   = np.random.choice(max_mem, batch_size, replace=False)
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
        replace_target=1000,
        device=None,
    ):
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size  = batch_size
        self.replace_target = replace_target
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else device
        )
        self.eval_net   = QNet(input_dims, n_actions).to(self.device)
        self.target_net = QNet(input_dims, n_actions).to(self.device)
        self.optimizer  = optim.Adam(self.eval_net.parameters(), lr=1e-3)
        self.loss_fn    = nn.MSELoss()
        self.model_file = fname

    def remember(self, s, a, r, s_, done):
        self.memory.store_transition(s, a, r, s_, done)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = self.eval_net(state)
        return torch.argmax(actions).item()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        s, a, r, s_, done = self.memory.sample_buffer(self.batch_size)
        s   = torch.tensor(s,   dtype=torch.float32, device=self.device)
        a   = torch.tensor(a,   dtype=torch.int64,   device=self.device)
        r   = torch.tensor(r,   dtype=torch.float32, device=self.device)
        s_  = torch.tensor(s_,  dtype=torch.float32, device=self.device)
        done= torch.tensor(done,dtype=torch.float32, device=self.device)

        idx = torch.arange(self.batch_size)
        q_pred = self.eval_net(s)[idx, a]
        with torch.no_grad():
            max_a = torch.argmax(self.eval_net(s_), dim=1)
            q_next = self.target_net(s_)[idx, max_a]
        q_target = r + self.gamma * q_next * done

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 5)
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