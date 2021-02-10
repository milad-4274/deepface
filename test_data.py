from deepface import DeepFace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os


df = pd.read_excel('second/sample_identity_gallery_probe.xlsx')
galleries = df[df['proposed Gallery/Probe'] == 'G']
probes = df[df['proposed Gallery/Probe'] == 'P']


def get_file_name(df, id):
    name = df[df['id'] == id]['file name']
    return name.tolist()


print(get_file_name(galleries, 3))
print(get_file_name(probes, 3))
print(get_file_name(probes, 3))

# should be general
here = os.path.dirname(os.path.realpath(__file__))
path = here + "/second/images"

for i in range(1,121):
    gallery = get_file_name(galleries, i)[0]
    for img in get_file_name(probes, i):
        gpath = osp.join(path, gallery)
        ppath = osp.join(path, img)
        # print(gpath,ppath,"pgpath")
        res = DeepFace.verify(gpath, ppath,enforce_detection=False)
        print(img, res)



