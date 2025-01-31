import blenderproc as bproc
import argparse
import os
import numpy as np
import h5py as h5
import time

print("Inside My Code")

parser = argparse.ArgumentParser()
parser.add_argument(
    'cad_file',
    nargs='?',
    help="Main CAD File",
    default='test_bw.fbx')

parser.add_argument(
    'cc_textures_path',
    nargs='?',
    default="/media/ali/SecondSSD/WorkingDircetory/venvs/data_sim/Lib/site-packages/blenderproc/resources/cctextures",
    help="Path to downloaded cc textures")

parser.add_argument(
    'output_dir',
    nargs='?',
    help="Path to where the final files will be saved ",
    default="./output")

parser.add_argument('--num_scenes',
                    type=int,
                    default=1000,
                    help="How many scenes with f number of images each to generate")

args = parser.parse_args()

bproc.init()

# Setting the interinsics of the camera:
# interinsics = np.array([711.1112738715278, 0.0, 255.5, 0.0, 711.1112738715278, 255.5, 0.0, 0.0, 1.0]).reshape(3,3)
# interinsics = np.array(
#     [2665.113525390625, 0.0, 948.5543842315674, 0.0, 2663.823627045982, 599.569589296374, 0.0, 0.0, 1.0]).reshape(3, 3)

img_size = [1920, 1200]
# img_size = [512, 512]

# bproc.camera.set_intrinsics_from_K_matrix(interinsics, img_size[0], img_size[1])


# Setting the render options
# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# activate segmentation rendering
bproc.renderer.enable_segmentation_output(map_by="category_id", default_values={"category_id": 0})

# activate normal rendering
bproc.renderer.enable_distance_output(True)

# Enable transparency (alpha channel)
bproc.renderer.set_output_format(enable_transparency=True)

bproc.renderer.set_max_amount_of_samples(50)

# create room
print("-----------------\n-----Creating Room-----\n--------------------")

# Creating the floor
floor = bproc.object.create_primitive('PLANE', scale=[1, 1, 1])
floor.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

# create bin to hold objects
bin_planes = [bproc.object.create_primitive('PLANE', scale=[0.31, 0.15, 1], location=[0, -0.20, 0.15], rotation=[-1.570796, 0, 0]),
              bproc.object.create_primitive('PLANE', scale=[0.31, 0.15, 1], location=[0, 0.20, 0.15], rotation=[1.570796, 0, 0]),
              bproc.object.create_primitive('PLANE', scale=[0.15, 0.20, 1], location=[0.31, 0, 0.15], rotation=[0, -1.570796, 0]),
              bproc.object.create_primitive('PLANE', scale=[0.15, 0.20, 1], location=[-0.31, 0, 0.15], rotation=[0, 1.570796, 0])]

for plane in bin_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)


## sample CC Texture and assign to room planes
# print("-----------------\n-----Loading CC Textures-----\n--------------------")
## cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path, used_assets=['Ground', 'Brick', 'Chainmail', 'Concrete', 'Fabric', 'Gravel', 'Leather', 'Rock', 'WoodFloor'])


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    # print(np.random.uniform(min, max))
    # print(np.random.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.6]))
    # obj.set_location(np.random.uniform(min, max))
    obj.set_location(np.random.uniform([-0.22, -0.11, 0.1], [0.22, 0.11, 0.3]))
    #    obj.set_location(np.random.uniform([-0.0, -0.0, 0.1], [0.0, 0.0, 0.2]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# sample plane light from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 5])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
light_plane.replace_materials(light_plane_material)

# sample a random point light on shell
light_point = bproc.types.Light()
light_point.set_energy(300)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))

list_of_objs = []
num_objs = 60
for obj_idx in range(num_objs):
    bw_obj = bproc.loader.load_obj(args.cad_file)[0]
    bw_obj.set_scale([0.001, 0.001, 0.001])
    bw_obj.set_cp("category_id", obj_idx + 1)
    bw_obj.enable_rigidbody(False, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    bw_obj.set_shading_mode('auto')
    bw_obj.set_location([0, 0, 8])
    list_of_objs.append(bw_obj)

# Scene Loop
for scene in range(args.num_scenes):
    # for scene in range(10):
    print("-----------------\n-----Assigning Random Texture to Floor and Bin Planes-----\n--------------------")

    random_cc_texture = np.random.choice(cc_textures)
    floor.replace_materials(random_cc_texture)

    for plane in bin_planes:
        random_cc_texture = np.random.choice(cc_textures)
        plane.replace_materials(random_cc_texture)

    print("Sampling the point light source location")
    light_location = bproc.sampler.shell(center=[0, 0, 0], radius_min=.5, radius_max=.8, elevation_min=7,
                                         elevation_max=89,
                                         uniform_volume=True)
    light_point.set_location(light_location)

    #    print("Setting random material properties for each obj")
    #    for obj in list_of_objs:
    #        mat = obj.get_materials()[0]
    #        print(type(obj))
    #        print(type(mat))
    #        grey_col = np.random.uniform(0.1, .15)
    #        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    #        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    #        mat.set_principled_shader_value("Specular", np.random.uniform(0.5, 1.0))

    # Sample object poses and check collisions
    print("-----------------\nFirst Simulation Phase\n--------------------")
    print("---------------------Sampling Object Poses------------------------")
    bproc.object.sample_poses(objects_to_sample=list_of_objs[:15],
                              sample_pose_func=sample_pose_func,
                              max_tries=1000)

    # Physics Positioning
    print("--------------------Simulating Physics-------------------------")
    for obj in list_of_objs[:15]:
        obj.enable_rigidbody(True)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=20,
                                                      check_object_interval=1,
                                                      object_stopped_location_threshold=0.05,
                                                      object_stopped_rotation_threshold=0.5,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)
    # for obj in list_of_objs[:15]:
    #     obj.enable_rigidbody(False)

    print("----------------------------------------------------------------------")

    print("-----------------\nSecond Simulation Phase\n--------------------")
    print("---------------------Sampling Object Poses------------------------")
    bproc.object.sample_poses(objects_to_sample=list_of_objs[15:30],
                              sample_pose_func=sample_pose_func,
                              max_tries=1000)

    print("--------------------Simulating Physics-------------------------")
    for obj in list_of_objs[15:30]:
        obj.enable_rigidbody(True)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=20,
                                                      check_object_interval=1,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)

    # for obj in list_of_objs[15:30]:
    #     obj.enable_rigidbody(False)

    print("----------------------------------------------------------------------")

    print("-----------------\nThird Simulation Phase\n--------------------")
    print("---------------------Sampling Object Poses------------------------")
    bproc.object.sample_poses(objects_to_sample=list_of_objs[30:45],
                              sample_pose_func=sample_pose_func,
                              max_tries=1000)
    
    # Physics Positioning
    print("--------------------Simulating Physics-------------------------")
    for obj in list_of_objs[30:45]:
        obj.enable_rigidbody(True)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=20,
                                                      check_object_interval=1,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)
    
    # for obj in list_of_objs[30:45]:
    #     obj.enable_rigidbody(False)
    print("----------------------------------------------------------------------")

    print("-----------------\nForth Simulation Phase\n--------------------")
    print("---------------------Sampling Object Poses------------------------")
    bproc.object.sample_poses(objects_to_sample=list_of_objs[45:60],
                              sample_pose_func=sample_pose_func,
                              max_tries=1000)
    
    # Physics Positioning
    print("--------------------Simulating Physics-------------------------")
    for obj in list_of_objs[45:60]:
        obj.enable_rigidbody(True)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=20,
                                                      check_object_interval=1,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)
    
    # for obj in list_of_objs[45:60]:
    #     obj.enable_rigidbody(False)
    print("----------------------------------------------------------------------")

    # BVH tree used for camera obstacle checks
    print("-----------------\n-----Creating BVH Tree-----\n--------------------")
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(list_of_objs)

    # CAMERA SAMPLER:
    print("-----------------\n-----Sampling Camera Poses-----\n--------------------")
    poses = 0
    bproc.camera.set_resolution(img_size[0], img_size[1])

    # Only changing the camera z location in each frame
    while poses < 5:

        # Sample location
        camera_location = np.random.uniform([0, 0, 1.2], [0, 0, 1.4])

        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(list_of_objs)

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, 0] - camera_location,
                                                                 inplane_rot=np.random.uniform(-0.1, 0.1))

        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        obstacle_condition = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree)

        # first render the scene with this specific camera pose and if object visible add the pose to a list of poses, after loop is done render the whole list
        if obstacle_condition:
            bproc.camera.add_camera_pose(cam2world_matrix, poses)
            poses += 1


##        # if visib_check:
##        #     bproc.camera.add_camera_pose(cam2world_matrix)
##        #

##    # render the whole pipeline
##    # bproc.camera.get_camera_pose(1)
##    # exit()
##    #    bproc.camera.set_intrinsics_from_K_matrix(interinsics)
    bproc.renderer.set_max_amount_of_samples(5)
    data = bproc.renderer.render()

   # print(np.array(data["instance_attribute_maps"]).shape)
   # Check if object is in view:
   # tmp_segmaps = np.array(data["category_id_segmaps"])
   #
   # for i, segmap in enumerate(tmp_segmaps):
   #     visible_pixels_count = len(np.where(segmap != 0)[0])
   #     if visible_pixels_count < 1000:
   #         print("Object not in view")
   #         data["category_id_segmaps"].pop(i)
   #         data["depth"].pop(i)
   #         data["colors"].pop(i)
   #         data["distance"].pop(i)
   #         data["instance_attribute_maps"].pop(i)
    append_condition = True

    #
    # Write data in bop format
    bproc.writer.write_bop('output/bin_picking_final/',
                           target_objects=list_of_objs,
                           depths=data["depth"],
                           colors=data["colors"],
                           color_file_format="JPEG",
                           ignore_dist_thres=10,
                           annotation_unit="mm",
                           append_to_existing_output=append_condition,
                           frames_per_chunk=20)
