import os, sys
import numpy as np

n = 200

with open("xxx", "r") as f:
    for line in f:
        pos = {}
        line = line.split(" ")
        for i in range(n):
            pos[i + 1] = (float(line[2 * i]), float(line[2 * i + 1]))
        idx = None
        for j, item in enumerate(line):
            if item == "value":
                idx = j
                break
        print (idx)
        break
