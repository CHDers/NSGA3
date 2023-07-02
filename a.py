# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/2 11:57
# @Author  : Yanjun Hao
# @Site    : 
# @File    : a.py
# @Software: PyCharm 
# @Comment :

num_obj=2
ref_vectors = []
for i in range(num_obj):
    vec = [0] * num_obj
    vec[i] = 1.0
    ref_vectors.append(vec)
print(ref_vectors)