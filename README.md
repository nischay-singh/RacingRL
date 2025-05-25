# DDQN Car Racing

This project implements a Double Deep Q-Network (DDQN) agent to play a custom car racing environment built from scratch using Pygame. The agent learns to control a car through a racing track using reinforcement learning.

## Project Structure

```
.
├── model.py          # DDQN implementation and neural network architecture
├── train.py          # Training script
├── evaluate.py       # Evaluation and visualization script
├── smoke_test.py     # Quick test script
├── env/             # Custom Pygame-based racing environment
├── checkpoints/     # Directory for saved model checkpoints
└── assets/          # Additional resources
```

## Features

- Custom Pygame-based racing environment
- Double Deep Q-Network (DDQN) implementation
- Experience replay buffer for stable learning
- Target network for improved training stability
- Epsilon-greedy exploration strategy
- Model checkpointing and loading
- GIF recording of agent performance
- GPU support for faster training

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pygame
- ImageIO (for GIF recording)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DDQN-Car-Racing.git
cd DDQN-Car-Racing
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent:

```bash
python train.py --episodes 500 --device cuda  # Use GPU
# or
python train.py --episodes 500 --device cpu   # Use CPU
```

Training parameters:
- `--episodes`: Number of training episodes (default: 500)
- `--device`: Device to use for training (default: "cpu")
- `--ckpt_dir`: Directory to save checkpoints (default: "checkpoints")
- `--save_every`: Save model every N steps (default: 5000)

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --ckpt checkpoints/ddqn.pt --record episode.gif
```

Evaluation parameters:
- `--ckpt`: Path to the model checkpoint (required)
- `--episodes`: Number of evaluation episodes (default: 1)
- `--record`: Output GIF filename for recording
- `--device`: Device to use for evaluation (default: "cpu")

## Model Architecture

The DDQN implementation consists of:

1. **Q-Network**: A simple neural network with:
   - Input layer (19 dimensions)
   - Hidden layer (256 units) with ReLU activation
   - Output layer (5 actions)

2. **Replay Buffer**: Stores and samples transitions for stable learning
   - Maximum size: 100,000 transitions
   - Batch size: 512

3. **Training Parameters**:
   - Learning rate: 1e-3
   - Discount factor (gamma): 0.99
   - Epsilon decay: 0.999995
   - Minimum epsilon: 0.1
   - Target network update frequency: 10,000 steps

## Acknowledgments

This project was inspired by Code Bullet's video on implementing a car racing AI using reinforcement learning. While this implementation is my own, the initial concept and motivation came from his work. You can find the original video here: [Code Bullet's Car Racing AI Video](https://www.youtube.com/watch?v=r428O_CMcpI&t=9s)