# ==============
#    Traffic
# ==============

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-traffic-causal-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256
## MTGNN
python learn_mtgnn.py --save ./model-traffic-mtgnn-3.pt --data ./data/traffic.txt --num_nodes 862 --batch_size 16 --epochs 30 --horizon 3

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-traffic-causal-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-traffic-causal-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-traffic-causal-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-traffic-gru-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256


# =================
#    Electricity
# =================

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-electricity-causal-3.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 3 --hidden_size 512 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-electricity-gru-3.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 3 --hidden_size 512
## MTGNN
python learn_mtgnn.py --save ./model-electricity-mtgnn-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 16 --epochs 30 --horizon 3
## SCINet
python learn_scinet.py --model_name SCINet --save ./model/model-electricity-scinet-3.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 3

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-electricity-causal-6.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 6 --hidden_size 512 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-electricity-gru-6.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 6 --hidden_size 512

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-electricity-causal-12.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 12 --hidden_size 512 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-electricity-gru-12.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 12 --hidden_size 512

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-electricity-causal-24.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 24 --hidden_size 512 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-electricity-gru-24.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 24 --hidden_size 512


# ==============
#    Solar
# ==============

#### Horizon 3
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-solar-causal-3.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 3 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-solar-gru-3.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 3 --hidden_size 256

#### Horizon 6
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-solar-causal-6.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 6 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-solar-gru-6.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 6 --hidden_size 256

#### Horizon 12
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-solar-causal-12.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 12 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-solar-gru-12.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 12 --hidden_size 256

#### Horizon 24
## Causal_GRU
python learn.py --model_name Causal --save ./model/model-solar-causal-24.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 24 --hidden_size 256 --base_model GRU
## GRU
python learn.py --model_name GRU --save ./model/model-solar-gru-24.pt --data ./data/solar_AL.txt --num_nodes 137 --epoch 100 --horizon 24 --hidden_size 256

