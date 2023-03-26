#! /bin/bash

# Traffic
#### Horizon 3
tmux new -s traf_causal_3 -d
tmux send -t traf_causal_3_mean "ca stock" Enter
tmux send -t traf_causal_3_mean "python learn.py --model_name Causal --save ./model/model-traffic-causal-mean-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU --device cuda:1" Enter