import pytest
import data_generation as dg
import numpy as np
import pdb

def test_rebalance_3d():
    src = np.asarray([np.asarray((0, 3, 0)), np.asarray((0, 1, 0)), np.asarray((0, 2, 0))])
    dest = np.asarray([np.asarray((0, 7, 0))])
    src_center = np.asarray((0, 2, 0))
    dest_center = np.asarray((0, 8, 0))

    l1, l2 = dg.rebalance(dest, src, dest_center, src_center, 2, 2)
    assert l1.shape[0] == 2
    assert l2.shape[0] == 2
    assert l1.shape[1] == 3
    assert l2.shape[1] == 3

def test_rebalance_2d():
    src = np.asarray([np.asarray((0, 3)), np.asarray((0, 1)), np.asarray((0, 2))])
    dest = np.asarray([np.asarray((0, 7))])
    src_center = np.asarray((0, 2))
    dest_center = np.asarray((0, 8))

    l1, l2 = dg.rebalance(dest, src, dest_center, src_center, 2, 2)
    assert l1.shape[0] == 2
    assert l2.shape[0] == 2
    assert l1.shape[1] == 2
    assert l2.shape[1] == 2

def test_distance_2d():
    p1 = np.asarray((0, 3))
    p2 = np.asarray((4, 0))
    assert dg.distance(p1, p2, 2) == 5

def test_distance_3d():
    p1 = np.asarray((0, 3, 0))
    p2 = np.asarray((0, 0, 4))
    assert dg.distance(p1, p2, 3) == 5

def test_data_generation_2d():
    lst = dg.generate_numbers(0, 10, 10, 2)
    assert lst.shape[0] == 10
    assert lst.shape[1] == 2
    for i in lst:
        for j in i:
            assert 0 <= j <= 10

def test_data_generation_3d():
    lst = dg.generate_numbers(0, 10, 10, 3)
    assert lst.shape[0] == 10
    assert lst.shape[1] == 3
    for i in lst:
        for j in i:
            assert 0 <= j <= 10

def test_cluster_2d():
    lst = np.asarray([np.asarray((0, 3)), np.asarray((0, 1)), np.asarray((0, 2)), np.asarray((0, 7))])
    lst1, lst2, kmeans = dg.cluster(lst, 4)
    assert lst1.shape == (2, 2)
    assert lst2.shape == (2, 2)

def test_cluster_3d():
    lst = np.asarray([np.asarray((0, 3, 0)), np.asarray((0, 1, 0)), np.asarray((0, 2, 0)), np.asarray((0, 7, 0))])
    lst1, lst2, kmeans = dg.cluster(lst, 4)
    assert lst1.shape == (2, 3)
    assert lst2.shape == (2, 3)

def test_cluster_3d_complex():
    lst = np.asarray([np.asarray((1.4, 3.99, 20)), np.asarray((3, 7, 40)), np.asarray((10, 10, 10)), np.asarray((2, 7, 1))])
    lst1, lst2, kmeans = dg.cluster(lst, 4)
    assert lst1.shape == (2, 3)
    assert lst2.shape == (2, 3)
