# -*-coding:utf-8 -*-
"""
# File       : dataselect.py
# Time       ：2023/9/1 9:13
# version    : 
# Author: Jun_fei Cai
# Description: Randomly selecting data from the directory for
"""

import os
import numpy as np


def data_needed(filepath):
    file_name = list()
    for i in os.listdir(filepath):
        data_collect = "".join(i)
        file_name.append(data_collect)
        