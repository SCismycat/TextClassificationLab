#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 10:37
# @Author  : Leslee
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_root = path_root.replace('\\', '/')
print(path_root)
