import json
import os
import sys
import io

#remove images with imageID in missing/json
trainImagesFile = './data/train.json'
dataset = './data/missing.json'
outdir = './data/train'
f = open(dataset, 'r')

trainFile = open(trainImagesFile, 'r')
listOfImagesJson  = json.load(trainFile)

countLastI = 0
data = json.load(f)
for image in data['images']:
	#Uncomment the next two lines if you have downloaded images with the old 'train.json' file.
	#fname = os.path.join(outdir, f'{image["imageId"]}.jpg')	
	#os.remove(fname)
	for i in range(countLastI,len(listOfImagesJson['images'])):

		if listOfImagesJson['images'][i]["imageId"] == image["imageId"]:
			if (i>=963824):
				print('You are done deleting broken images! Writing to file now...')
			listOfImagesJson['images'].pop(i)
			listOfImagesJson['annotations'].pop(i)
			countLastI=i
			break
	

open(trainImagesFile, "w").write(
    json.dumps(listOfImagesJson, indent=4, separators=(',', ': '))
)

