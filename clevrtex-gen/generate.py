import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime as dt
from pathlib import Path

import bpy
import numpy as np

sys.path.append('.')
import blender_utils as utils
import clevr_qa
import img_utils
from sampling import SamplingConfig
from scene import SceneWrapper

MAX_RETRIES = 50

CAMERA_JITTER = 0.5
KEY_LIGHT_JITTER = 1.0
FILL_LIGHT_JITTER = 1.0
BACK_LIGHT_JITTER = 1.0
PIXELS_PER_OBJECT = 200

trans_t = lambda t : np.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0],
    [0,np.cos(phi),-np.sin(phi)],
    [0,np.sin(phi), np.cos(phi)],
], dtype=np.float32)

rot_alpha = lambda alpha : np.array([
    [np.cos(alpha),-np.sin(alpha),0],
    [np.sin(alpha), np.cos(alpha),0],
    [0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th)],
    [0,1,0],
    [np.sin(th),0, np.cos(th)],
], dtype=np.float32)

def pose_spherical(alpha, phi, radius):
    c2w = np.array([0.,0.,radius])
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_alpha(alpha/180.*np.pi) @ c2w
    
    return c2w

class TriesExceededError(Exception):
    pass


def main(args):
    name = f'{args.filename_prefix}_{args.variant}'.lower()
    output_image_dir_dest = Path(args.output_dir)
    output_image_dir_dest = output_image_dir_dest / name
    print(f'Output dest {output_image_dir_dest}')
    output_image_dir_dest.mkdir(parents=True, exist_ok=True)
    output_base_path = output_image_dir_dest

    if args.temp_dir is not None:
        # Redirect to temp dir; useful e.g. for fast IO to local storage
        tmp_dir = Path(args.temp_dir) / output_image_dir_dest.name
        tmp_dir.mkdir(parents=True)
        output_base_path = tmp_dir
        print(f'Output redirect {output_base_path}')

    step = int(args.num_images / float(args.num_shards))
    start = args.shard * step
    stop = min((args.shard + 1) * step, args.num_images)
    print(f'Will generate from {start} to {stop}')
    for i in range(start, stop):
        TRIES = 0
        idx = i + args.start_idx

        # Split into chunks to prevent laggy directory navigation
        chunk = str(idx // 1000)

        basename_prefix = f'{args.filename_prefix}_{args.variant}'
        output_path = output_base_path / chunk
        output_path.mkdir(exist_ok=True)
        output_prefix_path = output_path / basename_prefix

        # num_objects = random.randint(args.min_objects, args.max_objects)
        num_objects = 2
        while True:
            try:
                TRIES += 1
                render_scene(args, num_objects, output_path, idx)
                break
            except TriesExceededError:
                if TRIES > 10000:
                    print(f"Too many retries on a single scene: {TRIES}")
                    # break  # What else is there to do?
        print(f'Progress: {i - start + 1}/{stop - start}')
        exit()

    if args.temp_dir:
        if args.tar:
            output_name = output_image_dir_dest.name + f'_{args.shard}'
            if not output_name.endswith('.tar'):
                output_name += ".tar"
            i = 0
            base_name = output_name
            while (output_image_dir_dest / output_name).exists():
                output_name = base_name + f"_{i}"
            outp = output_image_dir_dest / output_name
            subprocess.check_call(
                ["/usr/bin/tar", "--no-auto-compress", "-cf", str(outp), "-C", str(output_base_path.parent),
                 str(output_base_path.name)], close_fds=True)
        else:
            import shutil
            shutil.copytree(output_base_path, output_image_dir_dest)


def render_scene(args,
                 num_objects,
                 output_path,
                 output_index=0,
                 ):
    scene = SceneWrapper(args.scene_blendfile)

    sampling_conf = SamplingConfig(
        args.properties_json,
        args.material_dir,
        args.shape_dir,
        same_material=args.same_material
    )

    # This will give ground-truth information about the scene and its objects
    metadata = {
        'num_objects': num_objects,
        'split': args.variant,
        'image_index': output_index,
        # 'image_filename': str(output_image.name),
    }

    # Use random as a single source of randomness
    jitter_array_fn = lambda s: (np.array([random.random(), random.random(), random.random()]) - 0.5) * 2.0 * s

    scene.camera.location = tuple(np.array(scene.camera.location) + jitter_array_fn(CAMERA_JITTER))
    scene.keylamp.location = tuple(np.array(scene.keylamp.location) + jitter_array_fn(KEY_LIGHT_JITTER))
    scene.backlamp.location = tuple(np.array(scene.backlamp.location) + jitter_array_fn(BACK_LIGHT_JITTER))
    scene.filllight.location = tuple(np.array(scene.filllight.location) + jitter_array_fn(FILL_LIGHT_JITTER))

    # Add ground material
    name, material_path = sampling_conf.sample_ground_mat()
    utils.add_material(scene.ground, material_path,
                       Color=(*[random.random() for _ in range(3)], 0),
                       Random=random.random(),
                       Displacement=0.35,
                       Scale=2.)
    metadata['ground_material'] = name

    metadata['directions'] = clevr_qa.compute_directions()

    # Populate scene geometry
    for i in range(3):
      scene.camera.location[i] *= 0.8

    objects, blender_objects = add_random_objects(args, metadata['directions'], num_objects, scene.camera,
                                                  sampling_conf)

    # todo: change path

    # scene.view_layer.use_pass_diffuse_color = True
    # output_albedo_path = utils.compositor_output(scene.output_layers_node.outputs['DiffCol'],
    #                                              scene.node_tree,
    #                                              str(output_image).replace('.png', '_albedo_'),
    #                                              set_bw=False)

    # scene.view_layer.use_pass_shadow = True
    # output_shadow_path = utils.compositor_output(scene.output_layers_node.outputs['Shadow'],
    #                                              scene.node_tree,
    #                                              str(output_image).replace('.png', '_shadow_'),
    #                                              set_bw=True)

    # scene.view_layer.use_pass_normal = True
    # output_normal_path = utils.compositor_output(scene.output_layers_node.outputs['Normal'],
    #                                              scene.node_tree,
    #                                              str(output_image).replace('.png', '_normal_'),
    #                                              set_bw=False)

    # TODO: any other useful passes?

    # obj_mask_imgs = []
    scene.view_layer.use_pass_object_index = True
    # add objects to meta
    for i, (obj, meta) in enumerate(zip(blender_objects, objects), start=1):
        meta['index'] = i
        obj.pass_index = i
        # obj_mask_path = utils.compositor_obj_mask_output(scene.output_layers_node.outputs['IndexOB'],
        #                                                  scene.node_tree,
        #                                                  i,
        #                                                  str(output_image).replace('.png', ''))

        # obj_mask_imgs.append(obj_mask_path)

        meta['color'], rgba = sampling_conf.sample_color_for_shape(meta['shape'])
        meta['material'], mat_path = sampling_conf.sample_material_for_shape(meta['shape'])
        utils.add_material(obj, mat_path, Color=rgba, Random=random.random())
        print(f'Set {obj.name} to {meta["material"]} from {mat_path}')
    metadata['objects'] = objects

    if args.missing_filepath_prefix is not None:
        missing_path_directory = str(args.missing_filepath_prefix)
        bpy.ops.file.find_missing_files(directory=missing_path_directory)



    # loop
    length = 0.
    for i in range(3):
      length+=scene.camera.location[i]**2
    length=math.sqrt(length)

    metadata['cam2world'] = tuple(tuple(row) for row in scene.camera.matrix_world)
    metadata['focal'] = scene.camera.data.lens
    metadata['relationships'] = clevr_qa.compute_all_relationships(metadata)
    # metadata['normal_filename'] = str(output_normal_path.name)
    # metadata['albedo_filename'] = str(output_albedo_path.name)
    # metadata['shadow_filename'] = str(output_shadow_path.name)

    output_image_dir = output_path / 'images'
    output_info_dir = output_path / 'info'
    output_depth_dir = output_path / 'depth'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_info_dir.mkdir(parents=True, exist_ok=True)
    output_depth_dir.mkdir(parents=True, exist_ok=True)
    # fix the output node, todo: check this
    output_node = scene.node_tree.nodes.new('CompositorNodeComposite')
    output_node.location = 200,0

    render_args = scene._ref.render
    render_args.image_settings.color_depth = '16'
    scene_args = scene._ref
    #setup the depthmap calculation using blender's mist function:
    # scene_args.render.layers['RenderLayer'].use_pass_mist = True
    #the depthmap can be calculated as the distance between objects and camera ('LINEAR'), or square/inverse square of the distance ('QUADRATIC'/'INVERSEQUADRATIC'):
    scene_args.world.mist_settings.falloff = 'LINEAR'
    #minimum depth:
    scene_args.world.mist_settings.intensity = 0.0
    #maximum depth (can be changed depending on the scene geometry to normalize the depth map whatever the camera orientation and position is):
    scene_args.world.mist_settings.depth = 30.
    for i in range(100):
        # file name
        output_image_path = Path(str(output_image_dir) + f'/{i:03}.png')
        output_depth_path_pre = Path(str(output_depth_dir) + f'/{i:03}.png')
        utils.configure_cycles(output_image_path,
                           args.width,
                           args.height,
                           args.render_tile_size,
                           args.render_num_samples,
                           args.render_min_bounces,
                           args.render_max_bounces,
                           use_gpu=args.gpu)

        # scene.view_layer.use_pass_mist = True
        # scene.node_tree.links.new(
        #     scene.output_layers_node.outputs['Mist'],
        #     output_node.inputs['Image'])
        # render_args = scene._ref.render
        # render_args.filepath = output_depth_path_pre
        # output_depth_path = utils.compositor_output(scene.output_layers_node.outputs['Mist'],
        #                                         scene.node_tree,
        #                                         str(output_depth_path_pre).replace('.png', '_depth_'),
        #                                         output_node,
        #                                         set_bw=True)
        # print(str(output_image_path).replace('.png', '_depth_'))
        metadata['depth_filename'] = str(output_depth_path_pre.name)
        metadata['image_filename'] = str(output_image_path.name)

        # camera
        if i!=0:
            alpha=(i%20) / 20. * 360.
            phi=80. - np.floor(i/5) / 20. * 80.

            location = pose_spherical(alpha,phi,length)

            for j in range(3):
              scene.camera.location[j] = location[j]#rand(args.camera_jitter)
            print(scene.camera.matrix_world)

            metadata['cam2world'] = tuple(tuple(row) for row in scene.camera.matrix_world)
            metadata['image_index'] = i
            # cam2w['cam2world'] = metadata['cam2world'] 
        
        # Render the scene

        scene.view_layer.use_pass_mist = True
        scene.node_tree.links.new(
            scene.output_layers_node.outputs['Mist'],
            output_node.inputs['Image'])
        render_args.filepath = str(output_depth_path_pre)
        bpy.ops.render.render(write_still=True)

        scene.node_tree.links.new(
            scene.output_layers_node.outputs['Image'],
            output_node.inputs['Image'])
        render_args.filepath = str(output_image_path)
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(e)
            raise

        # mask_out_path = Path(str(output_prefix_path) + '_flat.png')
        # masks = img_utils.collect_object_masks(obj_mask_imgs, args.max_objects, mask_out_path)

        # for omi in obj_mask_imgs:
        #     omi.unlink()

        # metadata['mask_filename'] = str(mask_out_path.name)

        with Path(str(output_info_dir) + f'/{i:03}.json').open('w') as f:
            json.dump(metadata, f, indent=2)
    print('hi')  
    exit()

    # if not np.all(masks[..., 1:num_objects + 1].astype(int).sum((0, 1)) > PIXELS_PER_OBJECT):
    #     print("Scene is too occluded")
    #     # Clean artifacts
    #     mask_out_path.unlink()
    #     output_normal_path.unlink()
    #     output_shadow_path.unlink()
    #     output_albedo_path.unlink()
    #     output_depth_path.unlink()
    #     output_image.unlink()

    #     # Raise error to retry
    #     raise TriesExceededError()


    cced_img_path = None
    if args.cc:
        cced_img_path = Path(str(output_prefix_path) + '_cc.jpg')
        img = img_utils.colorcorrect(output_image, cced_img_path)
    else:
        img = img_utils.load(output_image)

    if args.npz:
        npz_output_path = Path(str(output_prefix_path) + '.npz')
        vis = [True] + [True] * num_objects + [False] * (args.max_objects - num_objects)
        np.savez_compressed(npz_output_path, mask=masks, visibility=np.array(vis), image=img)

    # with Path(str(output_prefix_path) + '.json').open('w') as f:
    #     json.dump(metadata, f, indent=2)

    if args.blendfiles:
        bpy.ops.file.make_paths_absolute()
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(str(output_prefix_path) + '.blend')))


def sample_position(obj_name_out, r, positions, directions, args):
    for num_tries in range(MAX_RETRIES):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)

        within_min_dist = True
        within_margin = True

        # Intersection check
        for (other_x, other_y, other_r) in positions:

            dx = x - other_x
            dy = y - other_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist - r - other_r < args.min_dist:
                within_min_dist = False
                break

            if args.margin > 0 and len(directions):
                for direction_name, direction_vec in directions.items():
                    if direction_name in {'above', 'below'}:
                        continue

                    margin = dx * direction_vec[0] + dy * direction_vec[1]

                    if 0 < margin < args.margin:
                        print(f'BROKEN MARGIN! {margin}, {args.margin} {direction_name}')
                        within_margin = False
                        break
                if not within_margin:
                    break

        if within_min_dist and within_margin:
            break
    else:
        raise TriesExceededError()

    # TODO: should size adjustement be done to shapes
    if obj_name_out == 'cube':
        r /= math.sqrt(2)

    # Choose random orientation for the object.
    rotation = 360.0 * random.random()
    return (x, y, r), {
        'shape': obj_name_out,
        'rotation': rotation,
    }


def sample_random_objects(args, directions, num_objects, conf):
    positions = []
    objects = []
    paths = []
    for i in range(num_objects):
        # Choose a random size
        size_name, r = conf.sample_size()
        # obj_name_out, obj_path = conf.sample_shape()
        obj_name_out, obj_path = conf.shapes[i]

        pos, obj = sample_position(obj_name_out, r, positions, directions, args)
        positions.append(pos)
        obj['size'] = size_name
        objects.append(obj)
        paths.append(obj_path)

    return objects, positions, paths


def sample_test_objects(args, directions, conf):
    size_name, r = next(iter(conf.sizes))

    positions = []
    objects = []
    paths = []
    for obj_name_out, obj_path in conf.shapes:
        pos, obj = sample_position(obj_name_out, r, positions, directions, args)
        positions.append(pos)
        obj['size'] = size_name
        objects.append(obj)
        paths.append(obj_path)
    return objects, positions, paths


def add_random_objects(args, directions, num_objects, camera, conf):
    """
    Add random objects to the current blender scene
    """
    objects, positions, paths = sample_random_objects(args, directions, num_objects, conf)
    blender_objects = []
    for meta, (x, y, r), obj_path in zip(objects, positions, paths):
        utils.add_object_geometry(obj_path, r, (x, y), rotation=meta['rotation'])
        obj = bpy.context.object
        blender_objects.append(obj)

        meta['3d_coords'] = tuple(obj.location)
        meta['pixel_coords'] = utils.get_camera_coords(camera, obj.location)

    return objects, blender_objects


def test_scene(args,
               objects,
               positions,
               paths,
               output_prefix_path,
               output_index=0,
               ):
    """Prepare test scene"""
    scene = SceneWrapper(args.scene_blendfile)
    output_image = Path(str(output_prefix_path) + '.png')
    utils.configure_cycles(output_image,
                           args.width,
                           args.height,
                           args.render_tile_size,
                           args.render_num_samples,
                           args.render_min_bounces,
                           args.render_max_bounces,
                           use_gpu=bool(args.gpu))
    # This will give ground-truth information about the scene and its objects
    metadata = {
        'num_objects': -1,
        'split': args.variant,
        'image_index': output_index,
        'image_filename': '',
        'directions': clevr_qa.compute_directions()
    }
    # Populate scene geometry
    blender_objects = []
    for meta, (x, y, r), obj_path in zip(objects, positions, paths):
        utils.add_object_geometry(obj_path, r, (x, y), rotation=meta['rotation'])
        obj = bpy.context.object
        blender_objects.append(obj)

        meta['3d_coords'] = tuple(obj.location)
        meta['pixel_coords'] = utils.get_camera_coords(scene.camera, obj.location)

    scene.view_layer.use_pass_mist = True
    output_depth_path = utils.compositor_output(scene.output_layers_node.outputs['Mist'],
                                                scene.node_tree,
                                                str(output_image).replace('.png', '_depth_'),
                                                set_bw=True)

    scene.view_layer.use_pass_diffuse_color = True
    output_albedo_path = utils.compositor_output(scene.output_layers_node.outputs['DiffCol'],
                                                 scene.node_tree,
                                                 str(output_image).replace('.png', '_albedo_'),
                                                 set_bw=False)

    scene.view_layer.use_pass_shadow = True
    output_shadow_path = utils.compositor_output(scene.output_layers_node.outputs['Shadow'],
                                                 scene.node_tree,
                                                 str(output_image).replace('.png', '_shadow_'),
                                                 set_bw=True)

    scene.view_layer.use_pass_normal = True
    output_normal_path = utils.compositor_output(scene.output_layers_node.outputs['Normal'],
                                                 scene.node_tree,
                                                 str(output_image).replace('.png', '_normal_'),
                                                 set_bw=False)

    obj_mask_imgs = []
    scene.view_layer.use_pass_object_index = True
    for i, (obj, meta) in enumerate(zip(blender_objects, objects), start=1):
        meta['index'] = i
        obj.pass_index = i
        obj_mask_path = utils.compositor_obj_mask_output(scene.output_layers_node.outputs['IndexOB'],
                                                         scene.node_tree,
                                                         i,
                                                         str(output_image).replace('.png', ''))

        obj_mask_imgs.append(obj_mask_path)
    metadata['objects'] = objects

    metadata['normal_filename'] = str(output_normal_path.name)
    metadata['albedo_filename'] = str(output_albedo_path.name)
    metadata['depth_filename'] = str(output_depth_path.name)
    metadata['shadow_filename'] = str(output_shadow_path.name)

    return scene, blender_objects, metadata, obj_mask_imgs


def test(args):
    output_image_dir_dest = Path(args.output_dir) / f'{args.filename_prefix}_{args.variant}'.lower()
    output_image_dir_dest.mkdir(parents=True, exist_ok=True)
    output_base_path = output_image_dir_dest

    if args.temp_dir is not None:
        # Redirect to temp dir; useful e.g. for fast IO to local storage
        tmp_dir = Path(args.temp_dir) / output_image_dir_dest.name
        tmp_dir.mkdir(parents=True)
        output_base_path = tmp_dir

    sampling_conf = SamplingConfig(
        args.properties_json,
        args.material_dir,
        args.shape_dir,
        same_material=True
    )

    TRIES = 0
    while True:
        try:
            scene = SceneWrapper(args.scene_blendfile)
            directions = clevr_qa.compute_directions()
            objects, positions, paths = sample_test_objects(args, directions, sampling_conf)
            for i, (name, mat_path) in enumerate(sampling_conf.mats):
                TRIES = 0
                idx = i + args.start_idx
                chunk = '0'
                basename_prefix = f'{args.filename_prefix}_{args.variant}_{idx:0>6d}'
                output_path = output_base_path / chunk
                output_path.mkdir(exist_ok=True)
                output_prefix_path = output_path / basename_prefix
                scene, blender_objects, metadata, obj_mask_imgs = test_scene(args, objects, positions, paths,
                                                                             output_prefix_path)
                utils.add_material(scene.ground,
                                   mat_path,
                                   Color=sampling_conf._color[-1],
                                   Random=0.5,
                                   Displacement=0.35,
                                   Scale=2.)

                bpy.ops.file.report_missing_files()
                if args.missing_filepath_prefix is not None:
                    missing_path_directory = str(args.missing_filepath_prefix)
                    bpy.ops.file.find_missing_files(directory=missing_path_directory)

                metadata['ground_material'] = name
                for meta, obj in zip(metadata['objects'], blender_objects):
                    meta['color'], rgba = sampling_conf._color
                    meta['material'] = name
                    utils.add_material(obj, mat_path, Color=rgba, Random=0.5)
                    print(f'Set {obj.name} to {meta["material"]} from {mat_path}')

                # Render the scene
                while True:
                    try:
                        bpy.ops.render.render(write_still=True)
                        break
                    except Exception as e:
                        print(e)
                        raise

                mask_out_path = Path(str(output_prefix_path) + '_flat.png')
                output_image = Path(str(output_prefix_path) + '.png')
                masks = img_utils.collect_object_masks(obj_mask_imgs, args.max_objects, mask_out_path)

                for omi in obj_mask_imgs:
                    omi.unlink()

                if not np.all(masks[..., 1:len(objects) + 1].astype(int).sum((0, 1)) > PIXELS_PER_OBJECT):
                    print("Scene is too occluded")
                    # Clean artifacts
                    mask_out_path.unlink()
                    # Raise error to retry
                    raise TriesExceededError()

                metadata['relationships'] = clevr_qa.compute_all_relationships(metadata)

                metadata['mask_filename'] = str(mask_out_path.name)

                cced_img_path = None
                if args.cc:
                    cced_img_path = Path(str(output_prefix_path) + '_cc.jpg')
                    img = img_utils.colorcorrect(output_image, cced_img_path)
                else:
                    img = img_utils.load(output_image)

                npz_output_path = Path(str(output_prefix_path) + '.npz')
                vis = [True] + [True] * len(objects) + [False] * (args.max_objects - len(objects))
                np.savez_compressed(npz_output_path, mask=masks, visibility=np.array(vis), image=img)

                with Path(str(output_prefix_path) + '.json').open('w') as f:
                    json.dump(metadata, f, indent=2)

                if args.blendfiles:
                    bpy.ops.file.make_paths_absolute()
                    bpy.ops.wm.save_as_mainfile(filepath=str(Path(str(output_prefix_path) + '.blend')))
            break
        except TriesExceededError:
            if TRIES > 10000:
                print(f"Too many retries on a single scene: {TRIES}")
                # break

    if args.temp_dir:
        if args.tar:
            output_name = output_image_dir_dest.name + f'_{args.shard:0>2d}'
            if not output_name.endswith('.tar'):
                output_name += ".tar"
            i = 0
            base_name = output_name
            while (output_image_dir_dest / output_name).exists():
                output_name = base_name + f"_{i}"
            outp = output_image_dir_dest / output_name
            subprocess.check_call(
                ["/usr/bin/tar", "--no-auto-compress", "-cf", str(outp), "-C", str(output_base_path.parent),
                 str(output_base_path.name)], close_fds=True)
        else:
            import shutil
            shutil.copytree(output_base_path, output_image_dir_dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Generation options
    parser.add_argument('--scene_blendfile', default='data/scene.blend',
                        help="Base blendfile which describes scene and camera and colour mapping settings.")
    parser.add_argument('--properties_json', default='data/full.json',
                        help="JSON which describes the sampling configuration options for files")
    parser.add_argument('--shape_dir', default='data/shapes',
                        help="Directory for <Shape name>.blend files. Sampling conf might scan this.")
    parser.add_argument('--material_dir', default='data/materials',
                        help="Directory for <material name>.blend files can be found. Sampling conf might scan this")

    # Settings for objects
    parser.add_argument('--min_objects', default=3, type=int)
    parser.add_argument('--max_objects', default=10, type=int)

    # Settings for CLEVR-like placement control:
    parser.add_argument('--min_dist', default=0.25, type=float,
                        help="The minimum allowed distance between object centers")
    parser.add_argument('--margin', default=0.4, type=float,
                        help="Along all cardinal directions (left, right, front, back), all " +
                             "objects will be at least this distance apart. This makes resolving " +
                             "spatial relationships slightly less ambiguous.")

    # CAMO dataset?
    parser.add_argument('--same_material', default=False, action='store_true',
                        help='Use the same material for ground and all objects')

    # Rendering fidelity options; these tradeoff visuals for performance
    parser.add_argument('--width', default=400, type=int)
    parser.add_argument('--height', default=400, type=int)
    parser.add_argument('--render_num_samples', default=512, type=int,
                        help="The number of samples to use when rendering. Larger values will " +
                             "result in nicer images but will cause rendering to take longer.")
    parser.add_argument('--render_min_bounces', default=8, type=int,
                        help="Minimum number of times to bounce the ray around the scene. Larger values should simulate"
                             "flood lighting better at the cost of performance.")
    parser.add_argument('--render_max_bounces', default=32, type=int,
                        help="Maximum number of times to bounce the ray around the scene. Larger values should lead to"
                             "more detailed reflections (inside reflections, inside reflections...).")
    parser.add_argument('--cc', default=False, action='store_true',
                        help="Apply ACE colour correction algorithm as a post proceesing step IN ADDITION TO BLENDER")

    # Output settings
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--num_images', default=5, type=int)

    # Helper options for multi-node generation such as SLURM
    parser.add_argument('--shard', default=os.environ.get('SLURM_ARRAY_TASK_ID', 0), type=int,
                        help='Current id for shard number')
    parser.add_argument('--num_shards', default=os.environ.get('SLURM_ARRAY_TASK_COUNT', 1), type=int,
                        help='Number of shards participacing in current generation')

    # Output filename options
    # File will have f"{output_dir}/{index//1000}/{filename_prefix}_{variant}_{index}.png" as name
    parser.add_argument('--filename_prefix', default='CLEVRTEX',
                        help="This prefix will be prepended to the rendered images and JSON scenes")
    parser.add_argument('--variant', default='debug',
                        help="Name of the dataset variant to use in the output")
    parser.add_argument('--output_dir', default='../output/images/',
                        help="The directory where output will be stored. It will be " +
                             "created if it does not exist.")

    parser.add_argument('--temp_dir', default=None,
                        help='Store intermediate output to temp directory before moving to output_dir')
    parser.add_argument('--tar', default=False, action='store_true',
                        help='Package files into non-compressed tar before moving to final location, '
                             'useful to prevent dumping lots of small files on a filesystem.')

    # Renderer options
    parser.add_argument('--gpu', default=False, action='store_true',
                        help="Use gpu for rendering. Right now only CUDA devices will be configured by this code.")
    parser.add_argument('--render_tile_size', default=512, type=int,
                        help="Size of a render tile. We found large tiles (around 512) to be good for GPU and used"
                             "64 for CPU based rendering. It is worthwhile to benchmark your hardware before generating.")

    # Debugging options
    parser.add_argument('--test_scan', default=False, action='store_true',
                        help='Rather than sampling, generate a series of test scenes'
                             'by finding a object placement where all different objects are visible'
                             'and sweeping through all materials.')
    parser.add_argument('--blendfiles', default=False, action='store_true',
                        help="Whether the scene blendfiles should be saved. Each might be 100-200Mb. "
                             "Only useful for debugging")

    parser.add_argument('--npz', default=False, action='store_true',
                        help="Place RGB images and masks into convenient npz archive for loading")

    parser.add_argument('--missing_filepath_prefix', default=None,
                        help="Path for remapping resource (texture) locations")

    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    args = parser.parse_args(argv)
    if args.test_scan:
        test(args)
    else:
        main(args)
