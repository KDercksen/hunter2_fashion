import json
import os
import sys
import io

#remove images with imageID in missing/json
dataset = './data/missing.json'
outdir = './data/train'
f = open(dataset, 'r')
data = json.load(f)
for image in data['images']:
	fname = os.path.join(outdir, f'{image["imageId"]}.jpg')	
	os.remove(fname)

