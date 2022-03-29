import os
import numpy as np
import imageio 
import json

root='output/clevrtex_animal2/0'
cam2w=[]
N_frames = 98
for i in range(N_frames):
    # s=str(i).zfill(6)
    file_name=os.path.join(root, 'info', f'{i:03}.json')
    
    with open(file_name, 'r') as fp:
        scene_struct = json.load(fp)
        cur_cam=scene_struct['cam2world']
    print(cur_cam)
    cam2w.append(cur_cam)

target_file1=os.path.join(root, 'transforms_train.json')
target_file2=os.path.join(root, 'transforms_val.json')
target_file3=os.path.join(root, 'transforms_test.json')
target_files = [target_file1, target_file2, target_file3]
for target_file in target_files:
    with open(target_file, "w") as t:
        t.write('{\n')
        t.write('    \"camera_angle_x\": 0.8575560450553894,\n')
        t.write('    \"frames": [\n')
        for i in range(N_frames):
            t.write('        {\n')
            t.write('            \"file_path\": "./train/r_'+str(i)+'\",\n')
            t.write('            \"rotation\": 0.012566370614359171,\n')
            t.write('            \"transform_matrix\": [\n')

            for j in range(4):
                t.write('                [\n')
                for k in range(4):
                    t.write('                    '+str(cam2w[i][j][k]))
                    if k!=3:
                        t.write(',\n')
                    else:
                        t.write('\n')
                if j!=3:
                    t.write('                ],\n')
                else:
                    t.write('                ]\n')
            t.write('            ]\n')
            if i!=N_frames-1:
                t.write('        },\n')
            else:
                t.write('        }\n')
        t.write('    ]\n')
        t.write('}\n')


