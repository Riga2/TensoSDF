import argparse
import subprocess


if __name__ == "__main__":
    env_name = 'cactus_scene007'
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', type=str, default='/home/riga/blender-3.6.5-linux-x64/blender')
    parser.add_argument('--mesh', type=str, default='data/meshes/orb/cactus_scene001_shape-180000.ply')
    parser.add_argument('--material', type=str, default='data/materials/cactus_scene001_mat-100000')
    parser.add_argument('--env_name', type=str, default=env_name)
    parser.add_argument('--hdr', type=str, default=f'/home/riga/NeRF/nerf_data/ground_truth/{env_name}/env_map')
    parser.add_argument('--gt', type=str, default='/home/riga/NeRF/nerf_data/blender_LDR/cactus_scene001')
    parser.add_argument('--name', type=str, default=f'cactus_scene001_relighting_{env_name}')
    parser.add_argument('--trans', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='orb', choices=['nerf', 'nero', 'tensoSDF', 'orb'])
    args = parser.parse_args()

    cmds=[
        args.blender, '--background', '--python', 'blender_backend/relight_backend.py', '--',
        '--output', f'data/relight/orb/noScale/{args.name}',
        '--mesh', args.mesh,
        '--material', args.material,
        '--env_fn', args.hdr,
        '--gt', args.gt,
        '--dataset', args.dataset,
        '--env_name', args.env_name,
    ]
    if args.trans:
        cmds.append('--trans')
    subprocess.run(cmds)
