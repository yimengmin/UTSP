# UTSP

Code repository for paper:

Yimeng Min, Yiwei Bai, and Carla P. Gomes.  
"Unsupervised Learning for Solving the Travelling Salesman Problem."  
NeurIPS 2023  



Here we provide an example for TSP 200,500 and 1000. 

# Unsupervised Learning
### Training on TSP 200:

`python train.py --num_of_nodes 200 --EPOCHS 300 --batch_size 32 --temperature 3.5  --C1_penalty 20.0 --nlayers 2 --hidden 64 --rescale 2.0 --moment 1 --lr 5e-3 --stepsize 20
`

Generate TSP 200 heatmaps (test file):

`python loadmodel.py --num_of_nodes 200 --batch_size 128 --temperature 3.5  --nlayers 2 --hidden 64 --rescale 2.0 --moment 1
`

### Training on TSP 500:
`python train.py --num_of_nodes 500 --EPOCHS 300 --batch_size 64 --temperature 3.5 --C1_penalty 10.0 --nlayers 2 --hidden 64 --lr 3e-3 --rescale 4. --stepsize 20`

`python loadmodel.py --num_of_nodes 500 --batch_size 128 --temperature 3.5  --nlayers 2 --hidden 64 --rescale 4.0 --moment 1`


### Training on TSP 1000:
`python train.py --num_of_nodes 1000 --EPOCHS 300 --batch_size 64 --temperature 3.5  --nlayers 2 --hidden 128 --rescale 4. --C1_penalty 10.0  --lr 3e-3 --stepsize 20`

`python loadmodel.py  --num_of_nodes 1000 --batch_size 128 --temperature 3.5  --nlayers 2 --hidden 128 --rescale 4.0 --moment 1`


## Search
`cd Search/`



### note:

remember to `mkdir Search/results/n` first

change the `#define Max_City_Num to n` in `Search/code/include/TSP_IO.h`

change the `instancenum` in new-solve-n.sh


## Search Parameters for TSP-200, TSP-500 and TSP-1000
on TSP 200
`./new-solve-200.sh  0 5 100 0 50 2 1 1` 


on TSP 500
`
./new-solve-500.sh 0 5 100 0 50 2 1 1
`

on TSP 1000
`
./new-solve-1000.sh 0 5 10 0 150 3 1 1
`

on TSP-500 and TSP-1000, we set T=0.04, (change `Param_T` in `Search/code/include/TSP_IO.h`)

on TSP-200, we set `double Param_T=0.08;`


#### search time
the `double Param_T` controls the search time, longer time should have better results

for example, on TSP-500, if you set `double Param_T=0.3;` and run `./new-solve-500.sh 0 5 100 0 50 2 1 1`,
you will have a gap around 0.42 % in around 10 minutes.


---
### on TSP-100

`python train.py --num_of_nodes 100 --EPOCHS 300 --batch_size 32 --temperature 3.5  --C1_penalty 20.0 --nlayers 2 --hidden 64 --rescale 1.0 --moment 1`

`python loadmodel.py --num_of_nodes 100 --batch_size 256 --temperature 3.5  --nlayers 2 --hidden 64 --rescale 1.0 --moment 1 --topk 10`

when search TSP-100

in `Search/code/include/TSP_IO.h`

change `#define Max_Inst_Num 128` to `#define Max_Inst_Num 10000`

set `#define Max_City_Num       100`

change `int Total_Instance_Num = 128;` to `int Total_Instance_Num = 10000;`

set `int Inst_Num_Per_Batch = 313;` 

set `double Param_T=0.01;`

set `int Rec_Num = 10;`  in `code/include/TSP_IO.h`;  the `Rec_Num` should be equal `--topk` in `loadmodel.py`

also remember `mkdir 100` in `Search/results`


run 

`./new-solve-100.sh  0.1 6 10 0 30 3 0 0`



---
## Detail of Search
This code follows https://github.com/Spider-scnu/Monte-Carlo-tree-search-for-TSP.

If you want to construct your customized input file, you need to follow the following format:

Assume n is the number of cities

Each line represents one input

The first 2 * n floats are the coordinates following (x1, y1, ..., xn, yn)

Then there is a word "output". After it, we have n + 1 numbers, which shows the optimal solution.

Then there is a word "indices". After it, there are n * topk ints, 

Then there is a word output. After it, there are n * topk float, which shows the corresponding heatmap value for each edge. 



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

