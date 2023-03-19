# ==============
#    Traffic
# ==============

#### Horizon 3
## Causal
python learn.py --model_name Causal --save ./model/model-traffic-causal-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256
## MTGNN
python learn_mtgnn.py --save ./model-traffic-mtgnn-3.pt --data ./data/traffic.txt --num_nodes 862 --batch_size 16 --epochs 30 --horizon 3

#### Horizon 6
## Causal
python learn.py --model_name Causal --save ./model/model-traffic-causal-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256

#### Horizon 12
## Causal
python learn.py --model_name Causal --save ./model/model-traffic-causal-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256

#### Horizon 24
## Causal
python learn.py --model_name Causal --save ./model/model-traffic-causal-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256
