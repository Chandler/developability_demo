import numpy as np
from trimesh import TriMesh

def assert_shape(m, shape):
    if m.shape != shape:
        raise ValueError("incorrect shape expected: {} found: {}".format(m.shape, shape))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def deduplicate_trimesh_vertices(trimesh):
    """
    return a TriMesh with duplicate vertices removed
    """
    def do_vert_hash(vert):
        return hash(str(set(vert)))

    new_verts = []
    new_faces = []

    seen_verts = {}
    for face in trimesh.faces:
        face_points = [trimesh.vs[i] for i in face]
        new_face = []
        for vert in face_points:
            vert_hash = do_vert_hash(vert)
            if vert_hash in seen_verts:
                index = seen_verts[vert_hash]
            else:
                new_verts.append(vert)
                index = len(new_verts)-1
                seen_verts[vert_hash] = index

            new_face.append(index)
        new_faces.append(new_face)

    new_trimesh = TriMesh()
    new_trimesh.vs = np.array(new_verts)
    new_trimesh.faces = np.array(new_faces)
    return new_trimesh
