import pytest
import lisa
import pdb
import numpy as np

def create_data_2d():
    lst = []
    for i in range(6):
        for j in range(6):
            lst.append(np.asarray((i,j)))
    return np.asarray(lst)

def create_data_3d():
    lst = []
    for i in range(30):
        for j in range(30):
            for k in range(30):
                lst.append(np.asarray((i,j,k)))
    return np.asarray(lst)

def test_border_points():
    lst = np.asarray(list(range(10)))
    res = lisa.calculate_border_points(lst, 3, 3)
    assert (res == np.asarray((0, 4, 7, 9))).all()

def test_create_cells_2d():
    lst = create_data_2d()
    T_i = [3, 3]
    res = lisa.create_cells(lst, T_i)

    assert (res == np.asarray(((0, 2, 4, 5), (0, 2, 4, 5)))).all()

    T_i = [2, 2]
    res = lisa.create_cells(lst, T_i)
    assert (res == np.asarray(((0, 3, 5), (0, 3, 5)))).all()

def test_create_cells_3d():
    lst = create_data_3d()
    T_i = [6, 6, 6]
    res = lisa.create_cells(lst, T_i)

    assert (res == np.asarray(((0, 5, 10, 15, 20, 25, 29), (0, 5, 10, 15, 20, 25, 29), (0, 5, 10, 15, 20, 25, 29)))).all()
   
    lst = create_data_3d()
    T_i = [3, 3, 3]
    res = lisa.create_cells(lst, T_i)
    
    assert (res == np.asarray(((0, 10, 20, 29), (0, 10, 20, 29), (0, 10, 20, 29)))).all()

def test_find_interval_2d():
    lst = create_data_2d()
    T_i = [3,3]
    Theta = lisa.create_cells(lst, T_i)

    point = np.asarray((1,4))
    interval = lisa.find_interval(point, Theta, T_i)

    assert (interval == np.asarray(((0, 2), (4, 5)))).all()

    point = np.asarray((0, 6))
    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((0, 2), (4, 5)))).all()

    point = np.asarray((3, 3))
    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((2, 4), (2, 4)))).all()

def test_find_interval_3d():
    lst = create_data_3d()
    T_i = [5, 5, 5]

    # SHould be intervals: [0, 6, 12, 18, 24, 29]
    Theta = lisa.create_cells(lst, T_i)
    point = np.asarray((27, 25, 11))

    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((24, 29), (24, 29), (6, 12)))).all()

    point = np.asarray((0, 0, 0))
    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((0, 6), (0, 6), (0, 6)))).all()

    point = np.asarray((29, 29, 29))
    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((24, 29), (24, 29), (24, 29)))).all()

    point = np.asarray((3, 26, 19))
    interval = lisa.find_interval(point, Theta, T_i)
    assert (interval == np.asarray(((0, 6), (24, 29), (18, 24)))).all()

def test_cell_index_2d():
    lst = create_data_2d()
    T_i = [3, 3]
    Theta = lisa.create_cells(lst, T_i)

    # We should now have 0-9 indices
    # Looks like:
    # 2 5 8
    # 1 4 7
    # 0 3 6

    # Should be intervals [0, 2, 4, 5]

    # 3rd row, 2nd column.
    point = np.asarray((3, 4))
    assert lisa.cell_index(point, Theta, T_i) == 1 * 3 + 2

    point = np.asarray((0, 0))
    assert lisa.cell_index(point, Theta, T_i) == 0 * 3 + 0

    point = np.asarray((5, 0))
    assert lisa.cell_index(point, Theta, T_i) == 2 * 3 + 0

    point = np.asarray((5, 2))
    assert lisa.cell_index(point, Theta, T_i) == 2 * 3 + 1

def test_cell_index_3d():
    lst = create_data_3d()
    T_i = [3, 3, 3]
    # We have 27 indices.
    Theta = lisa.create_cells(lst, T_i)

    # Should have intervals [0, 10, 20, 29]
    # x moves 9 per 1, y moves 3 per 1, z moves 1 per 1
    # Check indices (0, 0, 0) is index 0
    point = np.asarray((0, 0, 0))
    assert lisa.cell_index(point, Theta, T_i) == (0 * 3 + 0) * 3 + 0
    
    # Check indices (0, 0, 1) is index 1
    point = np.asarray((0, 0, 11))
    assert lisa.cell_index(point, Theta, T_i) == (0 * 3 + 0) * 3 + 1

    # Check indices (0, 1, 0) is index 3
    point = np.asarray((0, 11, 0))
    assert lisa.cell_index(point, Theta, T_i) == (0 * 3 + 1) * 3 + 0

    # Check indices (1, 0, 0) is index 9
    point = np.asarray((11, 0, 0))
    assert lisa.cell_index(point, Theta, T_i) == (1 * 3 + 0) * 3 + 0

    # Check indices( 1, 1, 1) is index 13
    point = np.asarray((11, 14, 19))
    assert lisa.cell_index(point, Theta, T_i) == (1 * 3 + 1) * 3 + 1

    # Check indices (2, 1, 0) is index 21
    point = np.asarray((29, 12.333, 1.11))
    assert lisa.cell_index(point, Theta, T_i) == (2 * 3 + 1) * 3 + 0

    # Check indices (2, 2, 2) is index 26
    point = np.asarray((24.8, 20, 21.4))
    assert lisa.cell_index(point, Theta, T_i) == (2 * 3 + 2) * 3 + 2

def test_mapping_function_index_2d():
    lst = create_data_2d()
    T_i = [3, 3]
    Theta = lisa.create_cells(lst, T_i)

    point = np.asarray((0.4, 0.3))
    assert lisa.mapping_function_with_index(3, np.asarray((0,0)), point, Theta, T_i) == 0.12/4 + 3

    point = np.asarray((3.5, 1.2))
    assert lisa.mapping_function_with_index(0, np.asarray((2, 0)), point, Theta, T_i) == 1.5 * 1.2 / 4 + 0

def test_mapping_function_index_3d():
    lst = create_data_3d()
    T_i = [3, 3, 3]
    # We have 27 indices.
    Theta = lisa.create_cells(lst, T_i)

    # Should have intervals [0, 10, 20, 29]
    # x moves 9 per 1, y moves 3 per 1, z moves 1 per 1
    # Index 0. Cell is 10 x 10 x 10. Fractional is 0 x 0 x 0
    point = np.asarray((0, 0, 0))
    assert lisa.mapping_function_with_index(10, np.asarray((0, 0, 0)), point, Theta, T_i) == (0 * 0 * 0) / (10 * 10 * 10) + 10
    
    point = np.asarray((23, 3, 16))
    assert lisa.mapping_function_with_index(0, np.asarray((20, 0, 10)), point, Theta, T_i) == (3 * 3 * 6) / (9 * 10 * 10) + 0

def test_mapping_function_2d():
    lst = create_data_2d()
    T_i = [3, 3]
    Theta = lisa.create_cells(lst, T_i)

    # Should be intervals [0, 2, 4, 5]

    # IND is 0. Area of cell is 2 x 2. Area of given is 0.4 x 0.3. Ratio is 0.4 * 0.3 / 1
    point = np.asarray((0.4, 0.3))
    assert lisa.mapping_function(point, Theta, T_i) == 0.12/4 + 0

    # Ind is 3 + 0. Area of cell is 2 x 2. Area of given is 1.5 * 1.2
    point = np.asarray((3.5, 1.2))
    assert lisa.mapping_function(point, Theta, T_i) == 1.5 * 1.2 / 4 + 3

    # IND is 6 + 2. Area of cell is 1x1. Area of given is 0.9 * 0.9
    point = np.asarray((4.9, 4.9))
    assert lisa.mapping_function(point, Theta, T_i) ==  0.9 * 0.9 / 1 + 8 

def test_mapping_function_3d():
    lst = create_data_3d()
    T_i = [3, 3, 3]
    # We have 27 indices.
    Theta = lisa.create_cells(lst, T_i)

    # Should have intervals [0, 10, 20, 29]
    # x moves 9 per 1, y moves 3 per 1, z moves 1 per 1
    # Index 0. Cell is 10 x 10 x 10. Fractional is 0 x 0 x 0
    point = np.asarray((0, 0, 0))
    assert lisa.mapping_function(point, Theta, T_i) == (0 * 0 * 0) / (10 * 10 * 10) + 0

    # Index is 0. Cell is 10 x 10 x 10. Fractional is 1 x 2 x 3
    point = np.asarray((1, 2, 3))
    assert lisa.mapping_function(point, Theta, T_i) == (1 * 2 * 3) / (1000) + 0
    
    # Check indices (0, 0, 1) is index 1
    point = np.asarray((0, 0, 11))
    assert lisa.mapping_function(point, Theta, T_i) == (0 * 0 * 1)/ (10 * 10 * 10) + 1

    # Check indices (0, 1, 0) is index 3
    point = np.asarray((0, 11, 0))
    assert lisa.mapping_function(point, Theta, T_i) == (0 * 1 * 0)/ (1000) + 3

    # Check indices (1, 0, 0) is index 9
    point = np.asarray((11, 0, 0))
    assert lisa.mapping_function(point, Theta, T_i) == (1 * 0 * 0) / (1000) + 9

    # Index is 0. Cell is 10 x 10 x 10. Fractional is 4.5 * 1.3 * 7.9
    point = np.asarray((4.5, 1.3, 7.9))
    assert lisa.mapping_function(point, Theta, T_i) == (4.5 * 1.3 * 7.9) / (1000) + 0

    # Check indices( 1, 1, 1) is index 13
    point = np.asarray((11, 14, 19))
    assert lisa.mapping_function(point, Theta, T_i) == (1 * 4 * 9)/(1000) + 13

    # Check indices (2, 1, 0) is index 21. Cell is 9 x 10 x 10. Fractional is 9 * 2.333 * 0.11
    point = np.asarray((29, 12.333, 1.11))
    assert lisa.mapping_function(point, Theta, T_i) == (9 * 2.333 * 1.11)/(900) + 21

    # Check indices (2, 2, 2) is index 26. Cell is 9x 9x 9
    point = np.asarray((24.8, 20, 21.4))
    assert lisa.mapping_function(point, Theta, T_i) == (4.8 * 0 * 1.4)/ (9*9*9) + 26

    # Index is 26. 
    point = np.asarray((24.8, 23.3, 21.4))
    assert lisa.mapping_function(point, Theta, T_i) == (4.8 * 3.3 * 1.4) / (9 * 9* 9) + 26

def test_decompose_query_3d():
    lst = create_data_3d()
    T_i = [3, 3, 3]
    Theta = lisa.create_cells(lst, T_i)
    # Should have intervals [0, 10, 20, 29]

    # Overlaps one bounding box
    bottom = np.asarray((5, 4, 1))
    top = np.asarray((8, 6, 9.9))
    assert (lisa.decompose_query(Theta, bottom, top) == np.asarray(((5, 4, 1), (8, 6, 9.9)))).all()

    # Overlaps 2 on x, but y,z stays the same
    bottom = np.asarray((11, 9, 4))
    top = np.asarray((28.5, 9.5, 6))
    assert (lisa.decompose_query(Theta, bottom, top) == np.asarray((((11, 9, 4), (20, 9.5, 6)),((20, 9, 4), ((28.5, 9.5, 6)))))).all()

    # Overlaps 3 on y, but z,x stays the same
    bottom = np.asarray((21, 7, 11))
    top = np.asarray((27, 25, 12))
    assert (lisa.decompose_query(Theta, bottom, top) == np.asarray((((21, 7, 11), (27, 10, 12)), ((21, 10, 11), (27, 20, 12)), ((21, 20, 11), (27, 25, 12))))).all() 
    
    # Overlaps 2 on z, but x,y stays the same.
    bottom = np.asarray((0, 20, 5))
    top = np.asarray((5, 25, 15))
    assert (lisa.decompose_query(Theta, bottom, top) == np.asarray((((0, 20, 5), (5, 25, 10)),((0, 20, 10), ((5, 25, 15)))))).all()
    
    # Overlap a 2 x 2 x 2.
    bottom = np.asarray((12, 9.5, 4))
    top = np.asarray((25, 19.5, 13))
    assert len(lisa.decompose_query(Theta, bottom, top)) == 8

    # Overlap a 3 x 2 x 2
    bottom = np.asarray((4, 14, 3))
    top = np.asarray((25, 23, 14))
    assert len(lisa.decompose_query(Theta, bottom, top)) == 12

    # Overlap a 1 x 2 x 2
    bottom = np.asarray((1, 1, 1))
    top = np.asarray((10, 14, 19))
    assert len(lisa.decompose_query(Theta, bottom, top)) == 4

    # Overlap a 3 x 3 x 3
    bottom = np.asarray((0, 0, 0))
    top = np.asarray((29, 29, 29))
    assert len(lisa.decompose_query(Theta, bottom, top)) == 27

    lst = []
    for i in range(len(Theta[0]) - 1):
        lst.append((Theta[0][i], Theta[0][i+1]))
    full_points = [tuple(zip(a,b,c)) for a in lst for b in lst for c in lst]
    assert lisa.decompose_query(Theta, bottom, top) == full_points

def test_decompose_query_2d():
    lst = create_data_2d()
    T_i = [3, 3]
    Theta = lisa.create_cells(lst, T_i)

    # Should be intervals [0, 2, 4, 5]
    # Overlaps purely one.
    bottom_left = np.asarray((1,1))
    top_right = np.asarray((1.3, 1.3))
    assert (lisa.decompose_query(Theta, bottom_left, top_right) == np.asarray(((1, 1), (1.3, 1.3)))).all()

    bottom_left = np.asarray((1,1))
    top_right = np.asarray((3, 2))
    assert len(lisa.decompose_query(Theta, bottom_left, top_right)) == 2 
    assert lisa.decompose_query(Theta, bottom_left, top_right) == [((1,1), (2, 2)), ((2,1), (3, 2))]
    
    bottom_left = np.asarray((2,0))
    top_right = np.asarray((4.5, 5))
    assert len(lisa.decompose_query(Theta, bottom_left, top_right)) == 6

    bottom_left = np.asarray((0, 0))
    top_right = np.asarray((5, 5))
    assert len(lisa.decompose_query(Theta, bottom_left, top_right)) == 9
    lst = []
    for i in range(len(Theta[0]) - 1):
        lst.append((Theta[0][i], Theta[0][i+1]))
    full_points = [tuple(zip(a,b)) for a in lst for b in lst]
    assert lisa.decompose_query(Theta, bottom_left, top_right) == full_points

def test_in_query_2d():
    # Point is contained
    point = np.asarray((1,1))
    rect = np.asarray(((0, 0), (10, 10)))

    assert lisa.in_query(point, rect)
    
    # Point is outside
    point = np.asarray((100, 100))
    rect = np.asarray(((0, 0), (10, 10)))
    assert not lisa.in_query(point, rect)

    # Point is on perimeter
    point = np.asarray((0, 5))
    rect = np.asarray(((0, 0), (10, 10)))
    assert lisa.in_query(point, rect)

    # Point has a value == to a side
    point = np.asarray((40, 10))
    rect = np.asarray(((0, 0), (10, 10)))
    assert not lisa.in_query(point, rect)

def test_distance_2d():
    p1, p2 = (0, 0), (3, 4)
    assert lisa.distance(p1, p2) == 25

    p1, p2 = np.asarray((75, 11)), np.asarray((75 - 6, 11 - 8))
    assert lisa.distance(p1, p2) == 100

def test_distance_3d():
    p1, p2 = (0, 0, 0), (1, 2, 2)
    assert lisa.distance(p1, p2) == 9

    p1, p2 = np.asarray((14.2, 10.7, 100)), np.asarray((14.2 + 2, 10.7 + 3, 100 + 6))
    assert lisa.distance(p1, p2) == 49

def test_nearest_2d():
    point = (1,1)
    results = [(1, 3, 1), (1, 2, 1)]
    assert (lisa.find_nearest(results, point, [], 1)[0] == (1,2,1)).all()
    
    assert (lisa.find_nearest(results, point, [], 2) == np.asarray(((16,3,1), (1,2,1)))).all()

def test_nearest_3d():
    point = (1.0, 1.0, 1.0)
    results = [(1, 3.0, 4.0, 5.0), (2.0, 2.0, 2.0, 2.0)]
    pdb.set_trace()
    assert (lisa.find_nearest(results, point, [], 1)[0] == (9, 2, 2, 2)).all()
