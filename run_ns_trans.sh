# ==============
#    Traffic
# ==============
python learn_ns_trans.py --model_name ns_transformer --save ./model/model-traffic-ns-trans-3.pt --data ./data/traffic.txt --horizon 3 --num_nodes 862 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-traffic-ns-trans-6.pt --data ./data/traffic.txt --horizon 6 --num_nodes 862 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-traffic-ns-trans-12.pt --data ./data/traffic.txt --horizon 12 --num_nodes 862 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-traffic-ns-trans-24.pt --data ./data/traffic.txt --horizon 24 --num_nodes 862 --hidden_size 256 --epoch 100


# =================
#    Electricity
# =================
python learn_ns_trans.py --model_name ns_transformer --save ./model/model-electricity-ns-trans-3.pt --data ./data/electricity.txt --horizon 3 --num_nodes 321 --hidden_size 512 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-electricity-ns-trans-6.pt --data ./data/electricity.txt --horizon 6 --num_nodes 321 --hidden_size 512 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-electricity-ns-trans-12.pt --data ./data/electricity.txt --horizon 12 --num_nodes 321 --hidden_size 512 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-electricity-ns-trans-24.pt --data ./data/electricity.txt --horizon 24 --num_nodes 321 --hidden_size 512 --epoch 100

# ==============
#    Solar
# ==============
python learn_ns_trans.py --model_name ns_transformer --save ./model/model-solar-ns-trans-3.pt --data ./data/solar_AL.txt --horizon 3 --num_nodes 137 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-solar-ns-trans-6.pt --data ./data/solar_AL.txt --horizon 6 --num_nodes 137 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-solar-ns-trans-12.pt --data ./data/solar_AL.txt --horizon 12 --num_nodes 137 --hidden_size 256 --epoch 100

python learn_ns_trans.py --model_name ns_transformer --save ./model/model-solar-ns-trans-24.pt --data ./data/solar_AL.txt --horizon 24 --num_nodes 137 --hidden_size 256 --epoch 100
