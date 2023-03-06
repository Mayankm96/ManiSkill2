# Import required packages
import argparse
import os.path as osp
import time
import tqdm

import gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode


# Defines a continuous, infinite horizon, task where done is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000,
        help="Total timesteps for training",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    total_step = args.total_timesteps
    control_freq = 250
    sim_freq = 500

    obs_mode = "state"
    control_mode = "pd_joint_pos"
    reward_mode = "dense"
    if args.seed is not None:
        set_random_seed(args.seed)


    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                control_mode=control_mode,
                control_freq=control_freq,
                sim_freq=sim_freq,
            )
            # Get frame skip
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(
                    env, record_dir, info_on_video=True, render_mode="cameras"
                )
            return env

        return _init

    # Create vectorized environments for training
    env = SubprocVecEnv(
        [
            make_env(env_id, max_episode_steps=max_episode_steps)
            for _ in range(num_envs)
        ]
    )
    env.seed(args.seed)
    env.reset()
    # Get the action space
    low = env.action_space.low
    high = env.action_space.high
    frame_skip = sim_freq // control_freq

    # Define the policy configuration and algorithm configuration
    t = time.perf_counter()
    for _ in tqdm.trange(total_step):
        # sample random action
        action = np.random.uniform(low, high, size=(num_envs, env.action_space.shape[0]))
        # step
        obs, reward, done, _ = env.step(action)
        # reset env if done
        # if not args.ignore_done:
        #     terminated_envs = np.where(done)[0]
        #     if len(terminated_envs) > 0:
        #         env.reset(id=np.where(done)[0])
        # render
        if num_envs == 1 and not args.headless:
            env.render()
    # FPS
    time_elapsed = time.perf_counter() - t
    fps = frame_skip * total_step * num_envs / time_elapsed

    print(f"FPS = {fps:.2f}")

if __name__ == "__main__":
    main()
