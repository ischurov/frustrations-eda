echo "Starting multi_nn_2023_02_01.sh"
python ./multi_nn_2023_02_01.py 0.,0.1,0.2 >./experiments/multi_nn_2023_02_01/0_0.1_0.2.log 2>./experiments/multi_nn_2023_02_01/0_0.1_0.2.err.log &
python ./multi_nn_2023_02_01.py 0.3,0.4,0.5 >./experiments/multi_nn_2023_02_01/0.3_0.4_0.5.log 2>./experiments/multi_nn_2023_02_01/0.3_0.4_0.5.err.log &
python ./multi_nn_2023_02_01.py 0.6,0.7,0.8 >./experiments/multi_nn_2023_02_01/0.6_0.7_0.8.log 2>./experiments/multi_nn_2023_02_01/0.6_0.7_0.8.err.log &
python ./multi_nn_2023_02_01.py 0.9,1.0,0.57 >./experiments/multi_nn_2023_02_01/0.9_1.0_0.57.log 2>./experiments/multi_nn_2023_02_01/0.9_1.0_0.57.err.log &
python ./multi_nn_2023_02_01.py 0.43,0.47,0.53 >./experiments/multi_nn_2023_02_01/0.43_0.47_0.53.log 2>./experiments/multi_nn_2023_02_01/0.43_0.47_0.53.err.log &