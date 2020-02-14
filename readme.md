demo of covariance energy for minimizing guassian curvature in a triangular mesh

based on

http://www.cs.columbia.edu/cg/developability/developability-of-triangle-meshes.pdf
https://github.com/odedstein/DevelopabilityOfTriangleMeshes


morning commands:
python3 cli.py \ #coffee
        --exact_grad \
    --in_obj=data/high_res_bunbun.obj \
    --out_obj=data/optimized_bunny2.obj \
    --energy=hinge \
    --action=optimize

python3 cli.py \
        --exact_grad \
    --in_obj=data/high_res_bunbun.obj \
    --out_obj=data/high_res_bunbun.obj \
    --energy=hinge \
    --action=optimize

### run using scipy numerical gradient
python3 cli.py \
 	--exact_grad \
    --in_obj=/Users/chandler/Downloads/octahedron.obj \
    --out_obj=data/test.obj \
    --energy=hinge \
    --action=optimize

python3 cli.py \
	--exact_grad \
    --in_obj=data/high_res_bunbun.py \
    --out_obj=data/optimized_bunny2.obj \
    --energy=hinge \
    --action=optimize

### run using exact gradient
python3 cli.py \
	--exact_grad \
    --in_obj=data/bunny.obj \
    --out_obj=data/optimized_bunny.obj \
    --energy=hinge \
    --action=optimize

### debug exact gradient
python3 cli.py \
	--in_obj=data/bunny.obj \
	--energy=hinge \
	--action=check

python3 view.py test.npy