import numpy as np
from simplification.cutil import simplify_coords
import pandas as pd

if __name__ == '__main__':
    w, h = 800, 600
    strokes = [[[510, 154], [502, 157], [491, 161], [478, 166], [463, 171], [443, 178], [424, 183], [402, 189], [382, 196], [362, 205], [344, 213], [332, 217], [325, 219], [319, 220], [316, 221], [314, 221], [313, 218], [312, 215], [312, 211]], [[351, 170], [360, 176], [371, 182], [382, 189], [395, 196], [408, 205], [419, 211], [432, 218], [443, 225], [453, 230], [462, 236], [468, 240], [471, 241], [472, 242], [473, 243]]]
    x_min, x_max = 800, 0
    y_min, y_max = 600, 0
    for stk in strokes:
        p = np.split(np.array(stk), [-1], axis=1)
        # print(p)
        p = np.array(p)
        # print(np.min(p[0]), np.max(p[0]))
        # print(np.min(p[1]), np.max(p[1]))
        x_min = min(x_min, np.min(p[0]))
        x_max = max(x_max, np.max(p[0]))
        y_min = min(y_min, np.min(p[1]))
        y_max = max(y_max, np.max(p[1]))

        # breakpoint()
    # print(x)
    # print(np.array(x).flatten())
    # breakpoint()
    # print(x_min, x_max, y_min, y_max)
    # breakpoint()
    stks = []
    mx = 0
    for stk in strokes:
        # print(stk)
        # print(p.shape)
        p = np.split(np.array(stk), [-1], axis=1)
        p = np.array(p).astype(np.float32).squeeze(-1)

        p[0] -= x_min
        p[1] -= y_min
        p[0] *= 255.0 / (x_max - x_min)
        p[1] *= 255.0 / (y_max - y_min)
        # print(p)
        # print(p.shape)
        p = p.astype(np.uint8)
        # print(p)
        # breakpoint()
        # print(p.shape)
        a = np.array(p[0])
        b = np.array(p[1])
        print(a, b)
        p = np.stack((a, b), axis=-1)
        print(p)
        print(p.shape)
        # breakpoint()

        p = simplify_coords(p, 2.0)
        p = np.split(p, [-1], axis=1)
        print(p)



        p = np.array(p).squeeze(-1).astype(np.uint8)
        stks.append(p.tolist())


    data = pd.DataFrame([[9000003627287624, 'UA', stks.__str__()]], columns=['key_id', 'countrycode', 'drawing'])
    data.to_csv('dataset/test_simplified/tmp.csv')