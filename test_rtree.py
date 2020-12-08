import pytest
import rtree as rt
import pdb
import numpy as np

def test_distance_2d():
    p1 = np.asarray((0, 3))
    p2 = np.asarray((4, 0))
    assert rt.distance(p1, p2, 2) == 25

def test_distance_3d():
    p1 = np.asarray((1,0,4))
    p2 = np.asarray((0, 2, 2))
    assert rt.distance(p1, p2, 3) == 9

def test_bounding_rect_2d():
    points = np.asarray([np.asarray((1,0)), np.asarray((50, 10)), np.asarray((40, 20)), np.asarray((20, 10))])
    result = rt.bounding_rectangle(points)
    assert result.shape == (2, 2)
    assert (result[0] == (1, 0)).all()
    assert (result[1] == (50, 20)).all()

def test_bounding_rect_3d():
    points = np.asarray([np.asarray((1,0, 82)), np.asarray((50, 10, 100)), np.asarray((40, 20, 54)), np.asarray((20, 10, 3))])
    result = rt.bounding_rectangle(points)
    assert result.shape == (2, 3)
    assert (result[0] == (1, 0, 3)).all()
    assert (result[1] == (50, 20, 100)).all()

def test_choose_leaf_2d():
    M = 2
    R = rt.RTree(2)
    n0 = rt.RTree_node(np.asarray(((0, 0), (20, 20))), [], None, [])
    n1 = rt.RTree_node(np.asarray(((0,0), (1,1))), [], n0, [(0, 0), (1,1)])
    n2 = rt.RTree_node(np.asarray(((10, 10), (20, 20))), [], n0, [(10, 10), (20, 20)])
    n0.children = [n1, n2]
    lst = [(0,0), (1,1), (10, 10), (20, 20)]
    assert n2 == R.chooseLeaf(n0, (15, 15))

def test_choose_leaf_3d():
    M = 2
    R = rt.RTree(3)
    n0 = rt.RTree_node(np.asarray(((0, 0, 0), (20, 20, 20))), [], None, [])
    n1 = rt.RTree_node(np.asarray(((0,0, 0), (1,1,1))), [], n0, [(0, 0,0), (1,1,1)])
    n2 = rt.RTree_node(np.asarray(((10, 10, 10), (20, 20, 10))), [], n0, [(10, 10, 10), (20, 20, 20)])
    n0.children = [n1, n2]
    assert n2 == R.chooseLeaf(n0, (15, 15, 15))
    assert n1 == R.chooseLeaf(n0, (0.3, 0.5, 0.4))
    assert n1 == R.chooseLeaf(n0, (2, 2, 2))

def test_overlap_2d():
    rect = np.asarray(((35, 40), (70, 80)))
    point = np.asarray((20, 80))
    assert rt.overlap(point, rect) is False
    point = np.asarray((50, 50))
    assert rt.overlap(point, rect) is True

def test_overlap_3d():
    rect = np.asarray(((35, 40, 60), (70, 80, 70)))
    point = (20, 80, 30)
    assert rt.overlap(point, rect) is False
    point = (40, 20, 50)
    assert rt.overlap(point, rect) is False
    point = (35, 40, 60)
    assert rt.overlap(point, rect) is True
    point = (70, 80,70)
    assert rt.overlap(point, rect) is True
    point = (40, 40, 80)
    assert rt.overlap(point, rect) is False
    point = np.asarray((50, 50, 60))
    assert rt.overlap(point, rect) is True

def test_overlap_1d():
    line1 = np.asarray((10, 20))
    line2 = np.asarray((15, 30))
    assert rt.overlap_1d(line1, line2) == True

    line1 = np.asarray((10, 20))
    line2 = np.asarray((30, 40))
    assert rt.overlap_1d(line1, line2) == False

def test_overlap_rect_2d():
    box1 = np.asarray(((10, 20), (30, 40)))
    box2 = np.asarray(((15, 15), (50, 50)))

    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20), (30, 40)))
    box2 = np.asarray(((0, 25), (50, 30)))
    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20), (30, 40)))
    box2 = np.asarray(((10, 20), (30, 40)))

    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20), (30, 40)))
    box2 = np.asarray(((15, 100), (24, 300)))
    assert rt.overlap_rectangles(box1, box2) is False

def test_overlap_rect_3d():
    box1 = np.asarray(((10, 20, 30), (30, 40, 50)))
    box2 = np.asarray(((15, 15, 20), (50, 50, 40)))
    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20, 30), (30, 40, 50)))
    box2 = np.asarray(((0,15, 35),(50,30, 60)))
    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20, 30), (30, 40, 50)))
    box2 = np.asarray(((10, 20, 30), (30, 40, 50)))
    assert rt.overlap_rectangles(box1, box2) is True

    box1 = np.asarray(((10, 20, 30), (30, 40, 50)))
    box2 = np.asarray(((0, 0, 60), (490, 30, 70)))
    assert rt.overlap_rectangles(box1, box2) is False

def test_mindist_2d():
    box = np.asarray(((10, 20), (30, 40)))
    # outside all
    point = (0, 0)
    assert rt.min_dist(point, box) == 500

    # inbetween x
    point = (11, 50)
    assert rt.min_dist(point, box) == 100

    point = (11, 0)
    assert rt.min_dist(point, box) == 400
    
    # Inside  box
    point = (11, 30)
    assert rt.min_dist(point, box) == 0
    
    # Inbetween y
    point = (50, 30)
    assert rt.min_dist(point, box) == 400

    point = (0, 30)
    assert rt.min_dist(point, box) == 100

def test_mindist_3d():
    box = np.asarray(((10, 20, 30), (30, 40, 50)))
    point = (0, 0, 0)
    # outside all
    assert rt.min_dist(point, box) == 1400

    # Inside
    point = (11, 21, 31)
    assert rt.min_dist(point, box) == 0

    # inbetween x
    point = (11, 60, 10)
    assert rt.min_dist(point, box) == 400 + 400

    point = (30, 10, 40)
    assert rt.min_dist(point, box) == 100

    # inbetween y
    point = (40, 23, 29)
    assert rt.min_dist(point, box) == 100 + 0 + 1 ** 2

    # inbetween z
    point = (0, 50, 31)
    assert rt.min_dist(point, box) == 100 + 100

def test_minmaxdist_2d():
    box = np.asarray(((10, 20), (30, 40)))
    # Take x = 10 and y = 20 as the two lines to check against
    point = (0, 0)
    assert rt.min_max_dist(point, box) == 900 + 400

    # Take x = 20 and y = 40 as the two lines to check against
    point = (25, 35)
    assert rt.min_max_dist(point, box) == 250

def test_minmaxdist_3d():
    box = np.asarray(((10, 20, 30), (30, 40, 50)))

    # Take x = 10, y= 40, z = 30
    point = (9, 41, 0)
    assert rt.min_max_dist(point, box) == 1782

    # Take x = 30, y = 20, z = 50
    point = (21, 21, 40)
    assert rt.min_max_dist(point, box) == (21 - 10) ** 2 + (21 - 20) ** 2 + (40- 30)** 2

