#!/bin/sh
python pretrain_maze.py agent.smm_args.vae_beta=0.25
python pretrain_maze.py agent.smm_args.vae_beta=0.5
python pretrain_maze.py agent.smm_args.vae_beta=0.75
python pretrain_maze.py agent.smm_args.vae_beta=1.0
python pretrain_maze.py agent.smm_args.vae_beta=1.25

sibling_epsilon=2.5
sibling_epsilon=5.0
sibling_epsilon=7.0

maze_type=square_bottleneck
maze_type=square_upside
maze_type=square_large
