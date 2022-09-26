"""
作用；针对非benchmark问题，没有PF_true问题，从算法解集中提取PF_true，并写入文本中，作为参照计算性能指标
最小化问题
"""

import os.path

import numpy as np
import pandas as pd
from typing import List

def read_solution(filepath: str = None, Algname: str = None) -> List:
    """
    Returns:
        object: solutions
    """
    solutions: dict={}
    with open(filepath) as file:
        index=0
        for line in file:
            vector=[float(x) for x in line.split()]
            solution={Algname+str(index):vector}
            solutions.update(solution)
            index+=1
    return solutions

def solve(sol_index):
    sol=data_array[:, sol_index]
    obj1_not_worse=np.where(sol[0]>=data_array[0, :])[0]
    obj2_not_worse=np.where(sol[1]>=data_array[1, :])[0]
    not_worse_candidates=set.intersection(set(obj1_not_worse), set(obj2_not_worse))

    obj1_better=np.where(sol[0]>data_array[0, :])[0]
    obj2_better=np.where(sol[1]>data_array[1, :])[0]
    better_candidates=set.intersection(set(obj1_better), set(obj2_better))

    dominating_solution=list(set.intersection(not_worse_candidates, better_candidates))
    if len(dominating_solution)==0:
        return True
    else:
        return False

if __name__=='__main__':
    # 读入一个算例，所有算法，多次运行，解集，全部混合一块
    input_dir='data'
    solutions={}
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem=dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem=dirname.split('\\')[-2:]

            if 'FUN' in filename:
                solutions.update(read_solution(filepath=os.path.join(dirname, filename), \
                                               Algname = algorithm + filename.split('.')[1]))

    # 提取目标值部分
    df=pd.DataFrame(data=solutions).T
    data_labels=list(df.index)
    data_array=np.array(df).T

    # get PF_True
    dominating_set=[]
    for k in range(data_array.shape[1]):
        if solve(k):
            dominating_set.append(data_labels[k])

    # Save PF_True to file
    with open('..\\resources\\reference_front\\MOKP.pf', 'w') as of:
        for i in range(len(dominating_set)):
            value = solutions.get(dominating_set[i])
            str_temp = ""
            for j in range(len(value)):
                str_temp += '\t' + str(value[j])
            of.write(dominating_set[i] + str_temp)
            of.write('\n')