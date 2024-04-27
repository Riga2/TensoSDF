# TensoSDF
TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction

The method is based on NeRO, please refer to it to setup the environment.

### Datasets
Download the [TensoSDF synthetic dataset](https://drive.google.com/file/d/1JI2kMvi_79JIUBGbUBckxeCAgrWEW0kl/view?usp=drive_link) and the [ORB real dataset](https://stanfordorb.github.io/). For the ORB dataset, We use the *blender_LDR.tar.gz* for training and *ground_truth.tar.gz* for evaluation.

## TensoSDF synthetic dataset
### Geometry reconstruction

Below take the "compressor" scene as an example:

```
# you need to modify the "dataset_dir" in configs/shape/syn/compressor.yaml first.

# reconstruct the geometry
python run_training.py --cfg configs/shape/syn/compressor.yaml

# evaluate the geometry reconstruction results via normal MAE metric
python eval_geo.py --cfg configs/shape/syn/compressor.yaml

# extract mesh from the model
python extract_mesh.py --cfg configs/shape/syn/compressor.yaml
```

Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. NVS results will be saved at ```data/nvs```. Extracted mesh will be saved at ```data/meshes```.

### Material reconstruction

```
# you need to modify the "dataset_dir" in configs/mat/syn/compressor.yaml first.

# estimate the material
python run_training.py --cfg configs/mat/syn/compressor.yaml

# evaluate the relighting results using the estimated materials via PSNR, SSIM and LPIPS metrics
python eval_mat.py --cfg configs/shape/syn/compressor.yaml --blender your_blender_path --env_dir your_environment_lights_dir
```
Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. Extracted materials will be saved at ```data/materials```. Relighting results will be saved at ```data/relight```.

## ORB real dataset
### Geometry reconstruction

Below take the "teapot" scene as an example:

```
# you need to modify the "dataset_dir" in configs/shape/orb/teapot.yaml first.

# reconstruct the geometry
python run_training.py --cfg configs/shape/orb/teapot.yaml

# extract mesh from the model
python extract_mesh.py --cfg configs/shape/orb/teapot.yaml

# evaluate the geometry reconstruction results via the CD metric
python eval_orb_shape.py --out_mesh_path data/meshes/teapot_scene006_shape-180000.ply --target_mesh_path your_ORB_GT_mesh_path
```

Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. Extracted mesh will be saved at ```data/meshes```.

### Material reconstruction

```
# you need to modify the "dataset_dir" in configs/mat/orb/teapot.yaml first.

# estimate the material
python run_training.py --cfg configs/mat/orb/teapot.yaml

# extract the materials and relight with new environment lights
python eval_mat.py --cfg configs/mat/orb/teapot.yaml --blender your_blender_path --orb_relight_gt_dir your_ORB_GT_relighting_dir --orb_relight_env your_relighting_env_name --orb_blender_dir your_orb_dataset_dir

# evaluate the relighting results via PSNR, SSIM and LPIPS metrics
python eval_orb_relight.py --relight_dir your_relighting_results_dir --gt_dir your_GT_relighting_in_orb_dataset_dir
e.g., python eval_orb_relight.py --relight_dir
```
Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. Extracted materials will be saved at ```data/materials```. Relighting results will be saved at ```data/relight```.