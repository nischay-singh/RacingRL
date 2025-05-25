import argparse, imageio, numpy as np, torch
from env.wrapper import CarRacingWrapper
from model import DDQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--record", metavar="GIF", help="output GIF filename")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    agent = DDQNAgent(
        alpha=0.0,   
        gamma=0.99,
        n_actions=5,
        epsilon=0.0,  
        batch_size=1,
        input_dims=19,
        device=torch.device(args.device),
        fname=args.ckpt,
    )
    agent.load_model()

    env = CarRacingWrapper()
    gif_frames = []

    for ep in range(args.episodes):
        obs = env.reset()
        done, ep_ret, steps = False, 0, 0
        while not done:
            a = agent.choose_action(obs)
            obs, r, done, _ = env.step(a)
            ep_ret += r
            steps += 1

            if args.record:
                frame = env.pg.surfarray.array3d(env.pg.display.get_surface())
                gif_frames.append(frame.swapaxes(0, 1))  
        print(f"episode {ep}  reward {ep_ret:.1f}  steps {steps}")

    env.close()

    if args.record:
        print(f"writing {args.record} …")
        imageio.mimsave(args.record, gif_frames, fps=30)


if __name__ == "__main__":
    main()