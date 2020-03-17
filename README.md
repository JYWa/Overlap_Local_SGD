
# Overlap-Local-SGD

Code to reproduce the experiments reported in this paper:
> Jianyu Wang, Hao Liang, Gauri Joshi, "Overlap Local-SGD: An Algorithmic Approach to Hide Communication Delays in Distributed SGD," ICASSP 2020. [(arXiv)](https://arxiv.org/abs/2002.09539)

This repo contains the implementations of the following algorithms:
- Local SGD [Stich ICLR 2018](https://arxiv.org/abs/1805.09767), [Yu et al. AAAI 2019](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4514), [Wang and Joshi 2018](https://arxiv.org/abs/1808.07576)
- Overlap-Local-SGD (proposed in this paper)
- Elastic Averaging SGD [Zhang et al. NeurIPS 2015](http://papers.nips.cc/paper/5761-deep-learning-with-elastic-averaging-sgd)
- CoCoD-SGD [Shen et al. IJCAI 2019](https://arxiv.org/abs/1906.12043)
- Blockwise Model-update Filtering (BMUF) [Chen and Huo ICASSP 2016](https://ieeexplore.ieee.org/abstract/document/7472805/) 

Please cite this paper if you use this code for your research/projects.

## Dependencies and Setup
The code runs on Python 3.5 with PyTorch 1.0.0 and torchvision 0.2.1.
The non-blocking communication is implemented using Python threading package.

## Training examples
We implement all the above mentioned algorithms as subclasses of [torch.optim.optimizer](https://pytorch.org/docs/stable/optim.html). A typical usage is shown as follows:
```python
import distoptim

# Before training
# define the optimizer
# One can use: 1) LocalSGD (including BMUF); 2) OverlapLocalSGD; 
#              3) EASGD; 4) CoCoDSGD
# tau is the number of local updates / communication period
optimizer = distoptim.SELECTED_OPTIMIZER(tau)
...... # define model, criterion, logging, etc..

# Start training
for batch_id, (data, label) in enumerate(data_loader):
	# same as serial training
	output = model(data) # forward
	loss = criterion(output, label)
	loss.backward() # backward
	optimizer.step() # gradient step
	optimizer.zero_grad()

	# additional line to average local models at workers
	# communication happens after every tau iterations
	# optimizer has its own iteration counter inside
	optimizer.average()
```
In addition, one need to initialize the process group as described in this [documentation](https://pytorch.org/docs/stable/distributed.html). In our private cluster, each machine has one GPU.
```python
# backend = gloo or nccl
# rank: 0,1,2,3,...
# size: number of workers
# h0 is the host name of worker0, you need to change it
torch.distributed.init_process_group(backend=args.backend, 
                                     init_method='tcp://h0:22000', 
                                     rank=args.rank, 
                                     world_size=args.size)
```

## Citation
```
@article{wang2020overlap,
	title={Overlap Local-{SGD}: An Algorithmic Approach to Hide Communication Delays in Distributed {SGD}},
	author={Wang, Jianyu and Liang, Hao and Joshi, Gauri},
	journal={arXiv preprint arXiv:2002.09539},
	year={2020}
}
```
    
