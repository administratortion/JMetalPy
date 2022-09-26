"""
    2022/02/05
    森辉，三种初始化方法，
    暂时未用，因为不比随机产生效果好
"""
import copy
import random
from typing import TypeVar, Generic, List

R = TypeVar("R")


class Create_Three_Solution(Generic[R]):
    # 根据物品的重量进行非上升排序
    def create_solution_item_weight_list(self, items_weight_list: List) -> List:
        items_weight_dictionary = {x: items_weight_list[x] for x in range(len(items_weight_list))}
        items_weight_dictionary = sorted(items_weight_dictionary.items(), key=lambda l: l[1], reverse=True)
        solution = list(dict(items_weight_dictionary).keys())
        return solution

    def create_solution_item_knapsack_penalty_list(self, items_weight: List,
                                                   knapsacks_weight: List,
                                                   item_knapsack_list: List
                                                   ) -> List:
        solution = []
        max_value = 99999
        item_integer_list = [-1 for _ in range(len(items_weight))]              # 装包初始化，全为 -1
        accumulate_total_weight = [0.0 for _ in range(len(knapsacks_weight))]   # 累计重量
        item_knapsack_dictionary = {}                       # 使用字典的方式保存数据 {’03‘：5} => 背包0，物品3，差异5
        for knap in range(len(item_knapsack_list)):
            item_knapsack_dictionary.update({str(knap)+str('_')+str(item): item_knapsack_list[knap][item] for item in range(len(item_knapsack_list[knap]))})
        item_knapsack_dictionary = sorted(item_knapsack_dictionary.items(), key=lambda l: l[1], reverse=False)      # 按照字典的第二个元素排序
        item_knapsack_dictionary = dict(item_knapsack_dictionary)

        for key, value in item_knapsack_dictionary.items():
            _knap, _item = [int(x) for x in key.split('_')]
            if value < max_value and item_integer_list[_item] == -1 \
                    and accumulate_total_weight[_knap] + items_weight[_item] <= knapsacks_weight[_knap]:
                item_integer_list[_item] = _knap
                accumulate_total_weight[_knap] += items_weight[_item]

        _key = [x for x in range(len(knapsacks_weight))]
        _key.append(-1)
        for for_i in range(len(_key)):
          for for_j in range(len(item_integer_list)):
              if item_integer_list[for_j] == _key[for_i]:
                  solution.append(for_j)

        # print(F"{item_knapsack_dictionary}：{len(item_knapsack_dictionary)}")
        # print(accumulate_total_weight)
        # print(f"{item_integer_list}：{len(item_integer_list)}")
        # print(F"{solution}：{len(solution)}")
        # print(F"{sorted(solution)}")
        return solution

    def create_solution_item_item_penalty_list(self,items_weight: List,
                                                   knapsacks_weight: List,
                                                   item_item_list: List) -> List:
        solution = []
        max_value = 99999
        item_integer_list = [-1 for _ in range(len(items_weight))]
        accumulate_total_weight = [0.0 for _ in range(len(knapsacks_weight))]
        item_item_dictionary = {}
        for i in range(len(item_item_list)):
            _tmp = []
            for x in range(i+1, len(item_item_list[0]) + 1, 1):
                _tmp.append(x)
            for j in range(len(item_item_list[i])):
                item_item_dictionary.update({
                    str(i)+str('_')+str(_tmp[j]): item_item_list[i][j]
                })
        item_item_dictionary = sorted(item_item_dictionary.items(), key=lambda l: l[1], reverse=False)
        item_item_dictionary = dict(item_item_dictionary)

        for key, value in item_item_dictionary.items():
            _item1, _item2 = [int(x) for x in key.split('_')]
            if value < max_value:
                if item_integer_list[_item1] == -1 and item_integer_list[_item2] == -1:     # 都为-1
                    for _knap in range(len(knapsacks_weight)):
                        if accumulate_total_weight[_knap] + items_weight[_item1] + items_weight[_item2] <= knapsacks_weight[_knap]:
                            item_integer_list[_item1] = _knap
                            item_integer_list[_item2] = _knap
                            accumulate_total_weight[_knap] += items_weight[_item1] + items_weight[_item2]
                            break
                elif item_integer_list[_item1] == -1 or item_integer_list[_item2] == -1:
                    if item_integer_list[_item1] == -1:
                        tmp_knap = item_integer_list[_item2]
                        if accumulate_total_weight[tmp_knap] + items_weight[_item1]  <= knapsacks_weight[tmp_knap]:
                            item_integer_list[_item1] = tmp_knap
                            accumulate_total_weight[tmp_knap] += items_weight[_item1]
                    elif item_integer_list[_item2] == -1:
                        tmp_knap = item_integer_list[_item1]
                        if accumulate_total_weight[tmp_knap] + items_weight[_item2] <= knapsacks_weight[tmp_knap]:
                            item_integer_list[_item2] = tmp_knap
                            accumulate_total_weight[tmp_knap] += items_weight[_item2]

        _key = [x for x in range(len(knapsacks_weight))]
        _key.append(-1)

        for for_i in range(len(_key)):
            for for_j in range(len(item_integer_list)):
                if item_integer_list[for_j] == _key[for_i]:
                    solution.append(for_j)

        # print(F"{item_item_dictionary}：{len(item_item_dictionary)}")
        # print()
        # print(accumulate_total_weight)
        # print(f"{item_integer_list}：{len(item_integer_list)}")
        # print(F"{solution}：{len(solution)}")
        # print(F"{sorted(solution)}")

        return solution


_create_three_solution = Create_Three_Solution()
