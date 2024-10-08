## TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction (SIGGRAPH 2024)

### [Paper](https://arxiv.org/abs/2402.02771) | [Project page](https://riga2.github.io/tensosdf/)

![Teaser](https://github.com/Riga2/TensoSDF/blob/main/user-imgs/teaser.png)

The method is based on [NeRO](https://github.com/liuyuan-pal/NeRO), please refer to it to setup the environment.
And then use pip to install the requirements.txt in this project.
```
cd TensoSDF
pip install -r requirements.txt
```

## Datasets
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
```
Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. Extracted materials will be saved at ```data/materials```. Relighting results will be saved at ```data/relight```.

## BibTeX
```
@article{Li:2024:TensoSDF,
  title={TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction},
  author={Jia Li and Lu Wang and Lei Zhang and Beibei Wang},
  journal ={ACM Transactions on Graphics (Proceedings of SIGGRAPH 2024)},
  year = {2024},
  volume = {43},
  number = {4},
  pages={150:1--13}
}
```
