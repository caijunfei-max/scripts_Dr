# -*-coding:utf-8 -*-
"""
# File       : data_aquisition.py
# Time       ：2023/10/25 15:13
# version    : 
# Author: Jun_fei Cai
# Description: 
"""
from mp_api.client.mprester import MPRester


API_KEY = "fqPPo7Czb5mkbFh8mltlZd0I33csuKv0"
mpr = MPRester(API_KEY)

docs = mpr.summary.search(elements=['Li', 'O'],
                          all_fields=False,
                          fields=['material_id', 'composition_reduced',
                                  'chemsys', 'density',
                                  'structure', 'energy_above_hull',
                                  'band_gap', 'efermi',
                                  'is_gap_direct', 'theoretical'])
for i in docs:
    if i[]
