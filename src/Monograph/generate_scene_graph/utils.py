def get_pc_rgb(rgb, depth, semantic, sem_uniq, height, width, f_w, f_h):
    ids_3d_points = {}
    r_values = {}
    g_values = {}
    b_values = {}

    pcd = []

    x0 = width//2
    y0 = height//2

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

