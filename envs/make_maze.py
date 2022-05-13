from logging import raiseExceptions
from envs.maze_env import Env
from dm_env import specs
import ipdb

import numpy as np

class MakeTimestep:
    def __init__(self, maze_type, state, maximum_timestep, 
                timesteps_so_far=None, prev_obs=None):
        self.maze_type = maze_type
        self._state = state
        self.maximum_timestep = maximum_timestep
        self.reward = 0.0  # No Extrinsic Reward
        self.final = None
        self.discount = 1.0
        self.action = None

        if self.maze_type == 'AntU':
            self.timesteps_so_far = timesteps_so_far
            self.observation = state
            self.prev_observation = prev_obs
        else:
            self.timesteps_so_far = self._state['n']
            self.observation = self._state['state'].numpy()
            self.prev_observation = self._state['prev_state'].numpy()

    def last(self):
        if self.timesteps_so_far >= self.maximum_timestep:
            return True
        else:
            return False

class DMCStyleWrapper:
    def __init__(self, env=None, maximum_timestep=None, maze_type=None, obs_dtype=np.float32):
        self._env = env
        self.maximum_timestep = maximum_timestep
        self.maze_type = maze_type
        self.obs_dtype = obs_dtype

        if self.maze_type == 'AntU':
            self.obs_space = self._env.obs_space.shape
            self.action_space = self._env.action_space.shape
            self.act_range = 1  # FIXME: AntU에서는 dimension 8, low,high=1
        elif self.maze_type == 'maze':
            self.obs_space = (self._env.state.shape[0],)
            self.action_space = (self._env.action_size, )
            self.act_range = self._env.action_range
        else:
            raise Exception('wrong maze_type')

    def observation_spec(self):
        return specs.Array(self.obs_space, self.obs_dtype, 'observation')
    
    def final_spec(self):
        return specs.Array(self.obs_space, self.obs_dtype, 'final')

    def action_spec(self):
        return specs.BoundedArray(self.action_space, self.obs_dtype, -self.act_range, self.act_range, 'action')

    def action_range(self):
        return self.act_range

    def reset(self, state=None, goal=None):
        self._env.reset(state=state, goal=goal)

        if self.maze_type == 'AntU':
            obs = self._env._cur_obs['observation'].astype(self.obs_dtype)
            if self._env._prev_obs is None:
                prev_obs = obs
            else:
                prev_obs = self._env._prev_obs['observation'].astype(self.obs_dtype)

            return MakeTimestep(self.maze_type, obs, self.maximum_timestep, 
            self._env.timesteps_so_far, prev_obs)

        else:
            return MakeTimestep(self.maze_type, self._env._state, self.maximum_timestep)
    
    def step(self, action):
        self._env.step(action)

        if self.maze_type == 'AntU':
            obs = self._env._cur_obs['observation'].astype(self.obs_dtype)
            if self._env._prev_obs is None:
                prev_obs = obs
            else:
                prev_obs = self._env._prev_obs['observation'].astype(self.obs_dtype)

            return MakeTimestep(self.maze_type, obs, self.maximum_timestep, 
            self._env.timesteps_so_far, prev_obs)

        else:
            return MakeTimestep(self.maze_type, self._env._state, self.maximum_timestep)

    def plot_trajectory(self, trajectory, save_dir, step, use_wandb, goal=None):
        if self.maze_type == 'AntU':
            self._env.plot_trajectory(trajectory_all=trajectory, save_dir = save_dir, step = step, use_wandb=use_wandb)
        else:
            self._env.maze.plot_trajectory(trajectory_all=trajectory, save_dir = save_dir, step = step, use_wandb=use_wandb, goal=goal)

    def state_coverage(self, trajectory_all, skill_dim):
        if self.maze_type == 'AntU':
            state_cov_avg = self._env.state_coverage(trajectory_all=trajectory_all, skill_dim=skill_dim)
        else:
            state_cov_avg = self._env.maze.state_coverage(trajectory_all=trajectory_all, skill_dim=skill_dim)
        
        return state_cov_avg


def make(maze_type=None, maximum_timestep=None, random=False, num_skills=None, train_random=False):
    env = Env(n = maximum_timestep, maze_type = maze_type, random=random, num_skills=num_skills, train_random = train_random)
    env = DMCStyleWrapper(env, maximum_timestep, maze_type='maze')

    return env

def make_antmaze(maze_type=None, maximum_timestep=None, dtype=None):
    if dtype == 'float32':
        dtype = np.float32
    else:
        dtype = np.float64

    # AntU map register
    import gym
    from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs
    register_mujoco_envs()
    train_env_name = "AntULongTrainEnv-v0"
    test_env_name = "AntULongTestEnv-v0"
    train_env = gym.make(train_env_name)
    eval_env = gym.make(test_env_name)

    # 카메라 앵글 변경
    from multiworld.envs.mujoco.cameras import ant_u_bk_camera
    train_env.initialize_camera(ant_u_bk_camera)
    eval_env.initialize_camera(ant_u_bk_camera)
    
    train_env = DMCStyleWrapper(train_env, maximum_timestep, maze_type='AntU', obs_dtype=dtype)
    eval_env = DMCStyleWrapper(eval_env, maximum_timestep, maze_type='AntU', obs_dtype=dtype)

    return train_env, eval_env

    
