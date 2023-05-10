import numpy as np

def get_pc_rgb(rgb, distance, semantic, sem_uniq, height, width, f_w, f_h):
    ids_3d_points = {}
    r_values = {}
    g_values = {}
    b_values = {}

    pcd = []

    x0 = width//2
    y0 = height//2

    depth = convert_distance_mtrs_depth(distance, int(height), int(width))

    for id in sem_uniq:
        ids_3d_points[id] = []
        r_values[id] = []
        g_values[id] = []
        b_values[id] = []

    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = (j - x0) * z / f_w
            y = (i - y0) * z / f_h
            pcd.append([x, y, z])

            r = rgb[i][j][0]
            g = rgb[i][j][1]
            b = rgb[i][j][2]

            sem_id = semantic[i][j]
            if sem_id != -1:
                ids_3d_points[sem_id].append([x,y,z])
                r_values[sem_id].append(r)
                g_values[sem_id].append(g)
                b_values[sem_id].append(b)

    return ids_3d_points, r_values, g_values, b_values, pcd

def convert_distance_mtrs_depth(distance, height, width):
    # hypersim bug fixed: They called depth distance! So we need to convert distance to get actual depth
    focal = 886.81

    plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    plane_z = np.full([height, width, 1], focal, np.float32)
    img_plane = np.concatenate([plane_x, plane_y, plane_z], 2)

    npyDepth = distance / np.linalg.norm(img_plane, 2, 2) * focal

    return npyDepth
