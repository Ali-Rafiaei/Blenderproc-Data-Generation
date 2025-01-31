import blenderproc as bproc
import argparse
import os
import numpy as np
import h5py as h5
import time


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    # print(np.random.uniform(min, max))
    # print(np.random.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6]))
    # obj.set_location(np.random.uniform(min, max))
    obj.set_location(np.random.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def render(args):
    print("Starting the data generation process...")

    cad_path = args.cad_file
    if args.intrinsics:
        intrinsics = args.intrinsics
    else:
        intrinsics = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

    if args.img_size:
        img_size = args.img_size
    else:
        img_size = [640, 480]
    bproc.init()

    # Load the object
    print("-----------------\n-----Loading Object-----\n--------------------")
    interested_obj = bproc.loader.load_obj(cad_path)[0]
    # scaling the object from millimeter to meter
    interested_obj.set_scale([0.001, 0.001, 0.001])
    interested_obj.set_cp("category_id", 1)

    # set shading and physics properties and randomize PBR materials
    interested_obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    interested_obj.set_shading_mode('auto')

    # create room
    print("-----------------\n-----Creating Room-----\n--------------------")
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2],
                                                 rotation=[-1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2],
                                                 rotation=[1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2],
                                                 rotation=[0, -1.570796, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2],
                                                 rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

    print("-----------------\n-----generating the lights-----\n--------------------")
    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                       emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0],
                                                                        [1.0, 1.0, 1.0, 1.0]))
    light_plane.replace_materials(light_plane_material)

    # sample a random point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(300)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))


    # sample CC Texture and assign to room planes
    print("-----------------\n-----Loading CC Textures-----\n--------------------")
    if args.cc_textures_to_use:
        cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path, used_assets=args.cc_textures_to_use)
    else:
        cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

    # Setting the rendering settings
    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # activate segmentation rendering
    bproc.renderer.enable_segmentation_output(map_by="category_id", default_values={"category_id": 0})
    # activate normal rendering
    bproc.renderer.enable_distance_output(True)
    # Enable transparency (alpha channel)
    bproc.renderer.set_output_format(enable_transparency=True)

    bproc.renderer.set_max_amount_of_samples(50)

    # Set the camera intrinsics
    bproc.camera.set_intrinsics_from_K_matrix(intrinsics, img_size[0], img_size[1])

    # Check if the output directory exists and if so ask if the user wants to append to it
    append_condition = False
    while os.path.exists(args.output_dir):
        answer = input("The output directory already exists, would you like to overwrite or append (o/a) to it? ")
        if answer == "a":
            append_condition = True
            break

        elif answer == "o":
            os.system(f"rm -r {args.output_dir}")
            break
        else:
            print("Please enter either 'o' or 'a' ")

    for scene in range(args.num_scenes):
        print("-----------------\n-----Assigning Random Texture to Room Planes-----\n--------------------")
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        light_location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5, elevation_min=5,
                                             elevation_max=89,
                                             uniform_volume=True)
        light_point.set_location(light_location)


        if args.randomize_obj_materials:
            mat = interested_obj.get_materials()[0]

            grey_col = np.random.uniform(0.1, 0.2)
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))


        # Sample object poses and check collisions
        print("-----------------\n-----Sampling Object Poses-----\n--------------------")
        bproc.object.sample_poses(objects_to_sample=[interested_obj],
                                  sample_pose_func=sample_pose_func,
                                  max_tries=1000)

        # Physics Positioning
        print("-----------------\n-----Simulating Physics-----\n--------------------")
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                          max_simulation_time=10,
                                                          check_object_interval=1,
                                                          substeps_per_frame=20,
                                                          solver_iters=25)

        # BVH tree used for camera obstacle checks
        print("-----------------\n-----Creating BVH Tree-----\n--------------------")
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects([interested_obj])

        # CAMERA SAMPLER:
        print("-----------------\n-----Sampling Camera Poses-----\n--------------------")
        poses = 0

        while poses < 1:

            # Sample location
            if args.camera_location:
                camera_location = args.camera_location
            else:
                camera_location = bproc.sampler.shell(center=[0, 0, 0],
                                                      radius_min=0.5,
                                                      radius_max=1.0,
                                                      elevation_min=5,
                                                      elevation_max=89,
                                                      uniform_volume=True)

            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            if args.object_always_at_center:
                poi = bproc.object.compute_poi([interested_obj])
            else:
                poi = bproc.object.compute_poi([interested_obj]) + np.random.uniform([-0.2, -0.1, 0.0], [0.2, 0.1, 0.0])

            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - camera_location,
                                                                     inplane_rot=np.random.uniform(-0.7854, 0.7854))

            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            obstacle_condition = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3},
                                                                             bop_bvh_tree)

            # first render the scene with this specific camera pose and if object visible add the pose to a list of poses, after loop is done render the whole list
            if obstacle_condition:
                bproc.camera.add_camera_pose(cam2world_matrix, poses)
                poses += 1


        data = bproc.renderer.render()


        if scene > 0:
            append_condition = True


        bproc.writer.write_bop(args.output_dir,
                               target_objects=[interested_obj],
                               depths=data["depth"],
                               colors=data["colors"],
                               color_file_format="JPEG",
                               ignore_dist_thres=10,
                               annotation_unit="mm",
                               append_to_existing_output=append_condition,
                               frames_per_chunk=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'cad_file',
        nargs='?',
        help="Main CAD File",
        default='/media/ali/SecondSSD1/MyResearch/Datasets/BOP/lm/models/obj_000001.ply')

    parser.add_argument(
        'cc_textures_path',
        nargs='?',
        default="/media/ali/SecondSSD1/WorkingDircetory/venvs/data_sim/Lib/site-packages/blenderproc/resources/cctextures",
        help="Path to downloaded cc textures")

    parser.add_argument(
        'output_dir',
        nargs='?',
        help="Path to where the final files will be saved ",
        default="./output/lm_ape")

    parser.add_argument('--num_scenes',
                        type=int,
                        default=15,
                        help="How many scenes with f number of images each to generate")

    parser.add_argument('--randomize_obj_materials',
                        type=bool,
                        default=False,
                        help="Randomize object materials")

    args = parser.parse_args()

    # Define the camera intrinsics else Linemod intrinsics will be used:
    args.intrinsics = None

    # Define the image size else 640x480 (Linemod) will be used:
    args.img_size = [640, 480]

    # Define the cc textures assets to use else all will be used:
    args.cc_textures_to_use = ["Ground", "Brick", "Metal"]

    # Define if the object should always be at the center of the image
    args.object_always_at_center = False

    # Define the camera location else it will be sampled
    args.camera_location = [0, 0, 0.5]

    render(args)
