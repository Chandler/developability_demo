import numpy as np

def get_face_normals(verts, faces):
    face_normals  = np.cross( verts[ faces[:,1] ] - verts[ faces[:,0] ], verts[ faces[:,2] ] - verts[ faces[:,1] ] )
    face_areas    = np.sqrt((face_normals**2).sum(axis=1))
    face_normals /= face_areas[:,np.newaxis]
    face_areas   *= 0.5
    return face_normals, face_areas

def get_face_angles(verts, faces):
    NUM_FACES = faces.shape[0]
   
    triangles = np.array(verts[faces.reshape(NUM_FACES*3)].reshape(NUM_FACES, 3, 3))

    # get a unit vector for each edge of the triangle
    u = triangles[:, 1] - triangles[:, 0]
    v = triangles[:, 2] - triangles[:, 0]
    w = triangles[:, 2] - triangles[:, 1]

    # norm per- row of each vector
    u /= row_norm(u).reshape((-1, 1))
    v /= row_norm(v).reshape((-1, 1))
    w /= row_norm(w).reshape((-1, 1))

    # run the cosine and per-row dot product
    a = np.arccos(np.clip(diagonal_dot(u, v), -1, 1))
    b = np.arccos(np.clip(diagonal_dot(-u, w), -1, 1))
    c = np.pi - a - b

    # convert NaN to 0.0
    angles = np.nan_to_num(np.column_stack([a, b, c]))

    return angles
def assert_shape(m, shape):
    if m.shape != shape:
        raise ValueError("incorrect shape expected: {} found: {}".format(m.shape, shape))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def diagonal_dot(a, b):
    """
    Dot product by row of a and b.
    There are a lot of ways to do this though
    performance varies very widely. This method
    uses the dot product to sum the row and avoids
    function calls if at all possible.
    Comparing performance of some equivalent versions:
    ```
    In [1]: import numpy as np; import trimesh
    In [2]: a = np.random.random((10000, 3))
    In [3]: b = np.random.random((10000, 3))
    In [4]: %timeit (a * b).sum(axis=1)
    1000 loops, best of 3: 181 us per loop
    In [5]: %timeit np.einsum('ij,ij->i', a, b)
    10000 loops, best of 3: 62.7 us per loop
    In [6]: %timeit np.diag(np.dot(a, b.T))
    1 loop, best of 3: 429 ms per loop
    In [7]: %timeit np.dot(a * b, np.ones(a.shape[1]))
    10000 loops, best of 3: 61.3 us per loop
    In [8]: %timeit trimesh.util.diagonal_dot(a, b)
    10000 loops, best of 3: 55.2 us per loop
    ```
    Parameters
    ------------
    a : (m, d) float
      First array
    b : (m, d) float
      Second array
    Returns
    -------------
    result : (m,) float
      Dot product of each row
    """
    # make sure `a` is numpy array
    # doing it for `a` will force the multiplication to
    # convert `b` if necessary and avoid function call otherwise
    # a = np.asanyarray(a)
    # 3x faster than (a * b).sum(axis=1)
    # avoiding np.ones saves 5-10% sometimes
    result = np.dot(a * b, np.array([1.0] * a.shape[1]))
    return result

def row_norm(data):
    """
    Compute the norm per- row of a numpy array.
    This is identical to np.linalg.norm(data, axis=1) but roughly
    three times faster due to being less general.
    In [3]: %timeit trimesh.util.row_norm(a)
    76.3 us +/- 651 ns per loop
    In [4]: %timeit np.linalg.norm(a, axis=1)
    220 us +/- 5.41 us per loop
    Parameters
    -------------
    data : (n, d) float
      Input 2D data to calculate per- row norm of
    Returns
    -------------
    norm : (n,) float
      Norm of each row of input array
    """
    return np.sqrt(np.dot(data ** 2, np.array([1] * data.shape[1])))

def roll_matrix(A, r):
    """
    roll each row of A an amount of each entry of r
    """
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]