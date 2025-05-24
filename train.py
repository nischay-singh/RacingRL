import argparse, numpy as np, torch, os, time
from env.wrapper import CarRacingWrapper
from model import DDQNAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=5000)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    env = CarRacingWrapper()
    state_size = 19
    agent = DDQNAgent(
        alpha=5e-5,
        gamma=0.99,
        n_actions=env.n_actions,
        epsilon=1.0,
        epsilon_end=0.1,
        epsilon_dec=0.999995,
        batch_size=64,
        input_dims=state_size,
        replace_target=10000,
        mem_size=100000,
        device=torch.device(args.device),
        fname=os.path.join(args.ckpt_dir, "ddqn.pt"),
    )

    global_steps, scores = 0, []
    for ep in range(args.episodes):
        obs = env.reset()
        done, ep_ret = False, 0
        while not done:
            action = agent.choose_action(obs)
            next_obs, r, done, _ = env.step(action)
            next_obs = next_obs
            agent.remember(obs, action, r, next_obs, done)
            agent.learn()
            obs, ep_ret = next_obs, ep_ret + r
            global_steps += 1

            if global_steps % args.save_every == 0:
                agent.save_model()

        scores.append(ep_ret)
        if ep % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"Ep {ep:4d} | EpRet {ep_ret:6.1f} | Avg10 {avg:6.1f} | ε {agent.epsilon:5.3f}")

    env.close()
    agent.save_model()

if __name__ == "__main__":
    main()