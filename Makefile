SHELL := /bin/bash

# run7:
# 	CUDA_VISIBLE_DEVICES=1 python FedQGPM_test.py # 42.61 / 2.7

# run6:
# 	CUDA_VISIBLE_DEVICES=1 python FedQGPM_test.py --gpmflag=True # 42.61 / -2.7

# run5:
# 	CUDA_VISIBLE_DEVICES=1 python FedQGPM_baseline.py --gpmflag=True # 44.39 / -1.22

run4:
	CUDA_VISIBLE_DEVICES=5 python main.py --baseline='agem' 

run3:
	CUDA_VISIBLE_DEVICES=5 python main.py --baseline='derpp' 

run2:
	CUDA_VISIBLE_DEVICES=5 python main.py --baseline='er' 

run1:
	CUDA_VISIBLE_DEVICES=5 python main.py --baseline='fdr' 

run_all: run1 run2 run3 run4
	@echo "All experiments finished."
