# UTSP
Code repository for paper:

Yimeng Min, Yiwei Bai, and Carla P. Gomes.  
"Unsupervised Learning for Solving the Travelling Salesman Problem."  
NeurIPS 2023  



Here we provide an example for TSP 200. 

# Unsupervised Learning
### Training on TSP 200:

`python train.py --num_of_nodes 200 --EPOCHS 300 --batch_size 32 --temperature 3.5  --C1_penalty 20.0 --nlayers 2 --hidden 64 --rescale 2.0 --moment 1 --lr 5e-3 --stepsize 20
`

Generate TSP 200 heatmaps (test file):

`python loadmodel.py --num_of_nodes 200 --batch_size 128 --temperature 3.5  --nlayers 2 --hidden 64 --rescale 2.0 --moment 1
`
### Search
`cd Search/
`

on TSP 200
`./new-solve-200.sh  0 5 100 0 50 2 1 1 
`

## Search Parameters for TSP-500 and TSP-1000



on TSP 500
`
./new-solve-500.sh 0 5 100 0 50 2 1 1
`

on TSP 1000
`
./new-solve-1000.sh 0 5 10 0 150 3 1 1
`

on TSP-500 and TSP-1000, we set T=0.04, (change `Param_T` in `code/include/TSP_IO.h`)



Regard the I/O Time:

On TSP-1000/500, the I/O can be time consuming.

In fact, for fair comparison with Att-GCN+MCTS, the process in the loadmodel section and the subsequent search involve storing and retrieving a sparse heat map in an N by N format. This approach results in increased I/O time. However, if the heat map is saved and loaded in a more compact, sparse manner, it would significantly cut down the time required. This improvement should be reflected in a reduced overall time reported in the UTSP paper.


---
## Detail of Search
This code follows https://github.com/Spider-scnu/Monte-Carlo-tree-search-for-TSP.
### Input data format
We put all the input files of our experiments into the instances folder. the link is here:
https://drive.google.com/drive/folders/1PyHUkPjtqo2lMRX3w986SqF89mvl0dc_?usp=sharing

If you want to construct your customized input file, you need to follow the following format:

Assume n is the number of cities

Each line represents one input

The first 2 * n floats are the coordinates following (x1, y1, ..., xn, yn)

Then there is a word "output". After it, we have n + 1 numbers, which shows the optimal solution.

Then there is a word "indices". After it, there are n * n ints, for the i-th consecutive n ints, these n ints are 1 to n except for i and we will replace i with -1 (since we do not want to have self-loop). (Please check TSP20_Input.txt for better understanding)

Then there is a word output. After it, there are n * n ints, which shows the corresponding heatmap value for each edge. Note the diagonal value should be all zero.

## Command Line Usage

(1) Tune some paraters of code/include/TSP_IO.h and code/include/TSP_Markov_Devision.h
    Please search "!!!" in these two cpp healder files, and we comment what you should modifiy. 
(2) make
(3) ./solve.sh $rec_only $M $K $alpha $beta $ph $restart 0

Note M, K, alpha and beta are exactly the same as we defined in the paper. And for their value, please check the Table 3 of the appendix.

Note for ph, we have a equation here: ph * n = T. T can also be found in the Table 3 and n is the number of cities.

rec_only: whether we only select edges from the heatmap that are greater than a threshold when we expand one node

M: maximal nodes we consider when we select the next node given one node. 

K: maximal depths

restart: whether we use restart to randomly select the maximal depths

