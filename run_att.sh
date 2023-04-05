# ==============
#    Traffic
# ==============

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-traffic-causal-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-traffic-causal-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256 --base_model GRU

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-traffic-causal-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256 --base_model GRU

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-traffic-causal-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256 --base_model GRU


# =================
#    Electricity
# =================

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-electricity-causal-3.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 3 --hidden_size 512 --base_model GRU

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-electricity-causal-6.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 6 --hidden_size 512 --base_model GRU

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-electricity-causal-12.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 12 --hidden_size 512 --base_model GRU

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-electricity-causal-24.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 24 --hidden_size 512 --base_model GRU


# ==============
#    Solar
# ==============

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-solar-causal-3.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-solar-causal-6.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 6 --hidden_size 256 --base_model GRU

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-solar-causal-12.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 12 --hidden_size 256 --base_model GRU

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal_ATT --save ./model/model-solar-causal-24.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 24 --hidden_size 256 --base_model GRU

