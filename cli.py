import sys
sys.path.append("src")
from trimesh import TriMesh
import util
from hinge_energy import hinge_energy
from scipy import optimize
import argparse
from numpy.linalg import norm

# different mesh energy functions
# each one has the interface
#  > energy(input_vector, trimesh, gradient_mode, NUM_VERTS)
#
# where gradient mode is an int:
#     0: return energy
#     1: return grad
#     2: return (energy, grad)

energy_lookup = {
    "hinge": hinge_energy
}

def optimize_energy(args):
    print "loading {}".format(args.in_obj)
    trimesh = TriMesh.FromOBJ_FileName(args.in_obj)

    energy_function = energy_lookup[args.energy]

    # a lot of meshes have duplicate vertices associated
    # with adjacent faces, clean that up.
    trimesh = util.deduplicate_trimesh_vertices(trimesh)

    NUM_VERTS = len(trimesh.vs)

    # curry different versions of the energy function and the starting mesh
    # these are now functions of a single variable that can be optimized
    energy_only     = lambda x: energy_function(x, trimesh, gradient_mode=0, NUM_VERTS=NUM_VERTS)
    energy_and_grad = lambda x: energy_function(x, trimesh, gradient_mode=2, NUM_VERTS=NUM_VERTS)
    
    # the first guess at the solution is the object itself
    first_guess = trimesh.vs.reshape(NUM_VERTS*3)

    print "starting optimization for mesh with {} verts".format(NUM_VERTS)
    result = \
        optimize.minimize(
            fun=energy_and_grad if args.exact_grad else energy_only,
            x0=first_guess,
            options={'disp': True},
            method="L-BFGS-B",
            constraints=None,
            jac=args.exact_grad)

    final_mesh_verts = result.x.reshape(NUM_VERTS, 3)
    
    trimesh.vs = final_mesh_verts
    print "writing: {}".format(args.out_obj)
    trimesh.write_OBJ(args.out_obj)

def check_gradient(args):
    """
    Compare the exact gradient to the result of
    the finite differences algorithm `optimize.approx_fprime`

    This is for debugging the gradient
    """
    trimesh = TriMesh.FromOBJ_FileName(args.in_obj)
    
    trimesh = util.deduplicate_trimesh_vertices(trimesh)

    NUM_VERTS = len(trimesh.vs)
    
    energy_function = energy_lookup[args.energy]

    energy_only = lambda x: energy_function(x, trimesh, gradient_mode=0, NUM_VERTS=NUM_VERTS)
    grad_only   = lambda x: energy_function(x, trimesh, gradient_mode=1, NUM_VERTS=NUM_VERTS)

    # the first guess at the solution is the object itself
    first_guess = trimesh.vs.reshape(NUM_VERTS*3)

    approximate_gradient = \
        optimize.approx_fprime(first_guess, energy_only, [0.01])

    extact_gradient = grad_only(first_guess)

    print "The difference between exact and numerical grad is: {}".format(norm(approximate_gradient - extact_gradient))

# different functions this cli can invoke, each one takes
# all the args
action_lookup = {
    "optimize": optimize_energy,
    "check_gradient": check_gradient
}

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--in_obj', default="data/bunny.obj")
    aparser.add_argument('--out_obj', default="data/optimized_bunny.obj")
    aparser.add_argument('--exact_grad', action='store_true', default=False, help="if true use exact gradient, if false use numerical approximation")
    aparser.add_argument('--energy', default="hinge", help="name of the cost function to use")
    aparser.add_argument('--action', default="optimize", help="what function to run")

    args = aparser.parse_args()

    action_lookup[args.action](args)

