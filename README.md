# UTSP
This is the (very brief) code repository for paper:

*Unsupervised Learning for Solving the Travelling Salesman Problem* 

currently it provides an example for TSP 100. 

## Unsupervised Learning
Training on TSP 100:

`
python -u UTSPmain.py --num_of_nodes 100 --EPOCHS 100 --batch_size 32 --temperature 3.5 --C1_penalty 20.0 --nlayers 2 --hidden 64 --rescale 2.0 --moment 1
`

Generate TSP 100 test file:

`
python -u loadmodel.py --num_of_nodes 100 --batch_size 32 --temperature 3.5 --nlayers 2 --hidden 64 --rescale 2.0 --moment 1
`

## Search
This code follows https://github.com/Spider-scnu/Monte-Carlo-tree-search-for-TSP.
### Input data format
We put all the input files of our experiments into the instances folder. the link is here:
https://drive.google.com/file/d/1D8C4nXA0wQuKyVIyqIDZfRDjJyO9hFcV/view?usp=sharing

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

