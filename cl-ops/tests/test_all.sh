#!/bin/bash
python3 -m unittest tests.test_unary_ops
python3 -m unittest tests.test_binary_ops
python3 -m unittest tests.test_reduce_ops
python3 -m unittest tests.test_pad
python3 -m unittest tests.test_conv
python3 -m unittest tests.test_maxpool
