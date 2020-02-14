import sys
sys.path.append("src")
from src.trimesh import TriMesh
from hinge_energy import hinge_energy
from hinge_energy_fast import hinge_energy as hinge_energy_fast
from scipy import optimize
import argparse
from numpy.linalg import norm
from jax import grad, jit, vmap
from viz import CallbackContainer
import numpy as np

# different mesh energy functions
# each one has the interface
#  > energy(input_vector, trimesh, gradient_mode, NUM_VERTS)
#
# where gradient mode is an int:
#     0: return energy
#     1: return grad
#     2: return (energy, grad)

def deduplicate_trimesh_vertices(trimesh):
    """
    Accepts a `TriMesh` and returns a new `TriMesh` with duplicate vertices removed
    """
    def do_vert_hash(vert):
        return hash(str(vert))

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


energy_lookup = {
    "hinge": hinge_energy
}

def optimize_energy(args):
    print("loading {}".format(args.in_obj))
    trimesh = TriMesh.FromOBJ_FileName(args.in_obj)

    energy_function = energy_lookup[args.energy]

    # a lot of meshes have duplicate vertices associated
    # with adjacent faces, clean that up.
    trimesh = deduplicate_trimesh_vertices(trimesh)

    NUM_VERTS = len(trimesh.vs)

    lookup_vertex_face_neighbors = {}
    for i_index in range(0, NUM_VERTS):
        lookup_vertex_face_neighbors[i_index] = trimesh.vertex_face_neighbors(i_index)
    
    faces = trimesh.faces
    
    energy_only      = lambda x: hinge_energy_fast(x, faces, lookup_vertex_face_neighbors, gradient_mode=0, NUM_VERTS=NUM_VERTS)
    energy_and_grad  = lambda x: (energy_only(x), grad(energy_only)(x))

    # the first guess at the solution is the object itself
    first_guess = trimesh.vs.reshape(NUM_VERTS*3)
    
    callback = \
        CallbackContainer(
            trimesh.vs,
            trimesh.faces)

    print("starting optimization for mesh with {} verts".format(NUM_VERTS))
    result = \
        optimize.minimize(
            fun=energy_and_grad if args.exact_grad else energy_only,
            x0=first_guess,
            options={'disp': True},
            # method="CG",
            # method="L-BFGS-B",
            method="BFGS",
            constraints=None,
            callback=callback,
            jac=args.exact_grad)

    final_mesh_verts = result.x.reshape(NUM_VERTS, 3)
    
    trimesh.vs = final_mesh_verts

    print("writing: {}".format(args.out_obj))
    trimesh.write_OBJ(args.out_obj)

    print("recomputing gradient for saving")
    callback.compute(energy_and_grad)
    out_npy = args.out_obj + ".npy"
    
    print("saving: {}".format(out_npy))
    callback.save(out_npy)

def check(args):
    """
    Compare the exact gradient to the result of
    the finite differences algorithm `optimize.approx_fprime`

    This is for debugging the gradient
    """
    trimesh = TriMesh.FromOBJ_FileName(args.in_obj)
    
    trimesh = deduplicate_trimesh_vertices(trimesh)

    NUM_VERTS = len(trimesh.vs)
    
    energy_function = energy_lookup[args.energy]

    # the first guess at the solution is the object itself
    first_guess = trimesh.vs.reshape(NUM_VERTS*3)

    lookup_vertex_face_neighbors = {}
    for i_index in range(0, NUM_VERTS):
        lookup_vertex_face_neighbors[i_index] = trimesh.vertex_face_neighbors(i_index)

    # energy_only      = lambda x: hinge_energy(x, trimesh.faces, lookup_vertex_face_neighbors, gradient_mode=0, NUM_VERTS=NUM_VERTS)
    energy_only_fast = lambda x: hinge_energy_fast(x, trimesh.faces, lookup_vertex_face_neighbors, gradient_mode=0, NUM_VERTS=NUM_VERTS)

    # print(energy_only(first_guess))
    print(energy_only_fast(first_guess))

# different functions this cli can invoke, each one takes
# all the args
action_lookup = {
    "optimize": optimize_energy,
    "check": check
}

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--in_obj', default="data/bunny.obj")
    aparser.add_argument('--out_obj', default="data/optimized_bunny.obj")
    aparser.add_argument('--save', action='store_true', default=True)
    aparser.add_argument('--exact_grad', action='store_true', default=False, help="if true use exact gradient, if false use numerical approximation")
    aparser.add_argument('--energy', default="hinge", help="name of the cost function to use")
    aparser.add_argument('--action', default="optimize", help="what function to run")

    args = aparser.parse_args()

    action_lookup[args.action](args)

