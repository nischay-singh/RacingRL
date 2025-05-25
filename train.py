import argparse, numpy as np, torch, os
from env.wrapper import CarRacingWrapper
from model import DDQNAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    env = CarRacingWrapper()

    agent = DDQNAgent(
        alpha=1e-3,
        gamma=0.99,
        n_actions=env.n_actions,
        epsilon=1.0,
        epsilon_end=0.1,
        epsilon_dec=0.999995,
        batch_size=512,
        input_dims=19,
        replace_target=1000,
        mem_size=100000,
        device=torch.device(args.device),
        fname=os.path.join(args.ckpt_dir, "ddqn.pt"),
    )

    agent.save_model()
    ckpt_path = os.path.abspath(agent.model_file)
    print(f"initial checkpoint → {ckpt_path}")

    global_steps, scores = 0, []
    for ep in range(args.episodes):
        obs = env.reset()
        done, ep_ret = False, 0
        while not done:
            a = agent.choose_action(obs)
            obs_, r, done, _ = env.step(a)

            agent.remember(obs, a, r, obs_, done)
            if agent.memory.mem_cntr > 5000:
                agent.learn()

            obs = obs_
            ep_ret += r
            global_steps += 1

            if global_steps % args.save_every == 0:
                agent.save_model()

        scores.append(ep_ret)
        if ep % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"Ep {ep:4d} | EpRet {ep_ret:6.1f} | Avg10 {avg:6.1f} | ε {agent.epsilon:5.3f}")

    env.close()
    agent.save_model()
    print(f"final model → {ckpt_path}")

if __name__ == "__main__":
    main()