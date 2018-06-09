import sys
sys.path.insert(0, 'object_detection/utils')

import visualization_utils as vis_util


import label_map_util
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from collections import defaultdict
from io import StringIO
from PIL import Image

folder = 'train'
LabelCounter = np.zeros(545, dtype=np.int64)
row_counter = 0 
startIndex = 1
flag = 0
if os.path.isfile('./data/boundariesCroppedImages'+folder+'.csv'):
  print('Recovering previous csv file...')
  flag = 1
  old_df = pd.read_csv('./data/boundariesCroppedImages'+folder+'.csv', index_col = 0)
  old_df = old_df[(old_df.T != 0).any()]
  row_counter = old_df[(old_df.T != 0).any()].shape[0]
  #sets start index to last imageId + 1
  startIndex = int((old_df.iloc[(row_counter-1)]['imageId' ]) +1)
  print('Csv file was successfully recovered.')
  print(old_df)
print(startIndex)

labelsCounterTxt = ''

whiteListLabels = [1,2,3,4, 7, 11, 12, 21, 22, 24, 25, 28, 29, 33, 39, 56, 60, 68, 72, 73, 90, 95, 96, 99, 121, 28, 130, 133, 145, 168, 179, 183, 186, 203, 207, 212, 224, 230, 254, 284, 319, 322, 327, 364, 438, 440, 530]

def load_image_into_numpy_array (image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

categories, probabilities = [], []
PATH_TO_CKPT = './faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = './object_detection/oid_bbox_trainable_label_map.pbtxt'
#PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#print(label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=545, use_display_name=True)
category_index = label_map_util.create_category_index(categories)  


PATH_TO_TEST_IMAGES_DIR = './data/'+folder
names = np.array([name.split('.jpg')[0] for name in os.listdir(PATH_TO_TEST_IMAGES_DIR) if os.path.isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, name))])
for i in range(30, 40):
  #print((str(i)+'.jpg') in names)
  print( str(names[i])+'.jpg')
names = np.sort(names.astype(int))
for i in range(30, 40):
  #print((str(i)+'.jpg') in names)
  print( str(names[i])+'.jpg')


#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1,len([name for name in os.listdir(PATH_TO_TEST_IMAGES_DIR) if os.path.isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, name))])+1) ]

start, = np.where(names == startIndex)
while(len(start) == 0):
  print('Searching for valid image name index: '+str(startIndex)+'...')
  startIndex += 1
  start, = np.where(names == startIndex)
print(start)
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(name)) for name in names[start[0]:] ]

'''for i in range (0, len(TEST_IMAGE_PATHS)):
    print(TEST_IMAGE_PATHS[i])
'''
# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

def writeCountedLabelsToFile():
    global labelsCounterTxt
    my_text_file = open('./data/countedLabels.txt', 'w')
    for i in range(0, len(LabelCounter)):
        if(LabelCounter[i] > 0):
            labelsCounterTxt += str(categories[i]['name']) + '('+ str(categories[i]['id'])+ '): ' + str(LabelCounter[i]) + '\n'
    my_text_file.write(labelsCounterTxt)
    my_text_file.close()



def filterDictionary(output_dict):
    new_num_detec = output_dict['num_detections']
    for i in range(0, output_dict['num_detections']):
        
        if( not (int(output_dict['detection_classes'][i]) in whiteListLabels)):
            output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], i)
            output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], (i))
            output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'],(i),0)
            new_num_detec -= 1
    output_dict['num_detections'] = new_num_detec;
    return output_dict

def detect_object(image):

    def img2array(img):
        (img_width, img_height) = img.size
        return np.array(img.getdata()).reshape((img_width, img_height, 3)).astype(np.uint8)

    
    with detection_graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

imageIndex = np.zeros(shape =len(TEST_IMAGE_PATHS))
boundaries = np.zeros(shape = (len(TEST_IMAGE_PATHS), 4))

counter = 0


# else:
#   df = pd.DataFrame(boundaries, columns=['yMin', 'xMin', 'yMax  ', 'xMax'])
#   print(df)
#   dfImageLabel = pd.DataFrame(imageIndex[(startIndex-1):], columns=['imageId'])
#   print(dfImageLabel)
#   result = pd.concat([dfImageLabel,df], axis=1)
print('checked')
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = detect_object(image_np)
    #Takes out labels that are not in whitelist.
    output_dict = filterDictionary(output_dict)
    
    relevantOutPutBoxCoordinates = [coordinates for coordinates in output_dict['detection_boxes'] if (np.count_nonzero(coordinates) > 0)]
    minMaxCoordinates = np.zeros(4)
    if(len(relevantOutPutBoxCoordinates)>0):
      for i in range(0, 4):
          if(i < 2):
              minMaxCoordinates[i] = np.min(np.array(relevantOutPutBoxCoordinates)[:,i])
          else:
              minMaxCoordinates[i] = np.max(np.array(relevantOutPutBoxCoordinates)[:,i])
    # Visualization of the results of a detection.
    '''vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)'''
    cutImage = (load_image_into_numpy_array(image))
    if(np.count_nonzero(minMaxCoordinates) > 0):
      cutImage = cutImage[int(minMaxCoordinates[0]*cutImage.shape[0]):int(minMaxCoordinates[2]*cutImage.shape[0]), int(minMaxCoordinates[1]*cutImage.shape[1]):int(minMaxCoordinates[3]*cutImage.shape[1]), :]
      imageBoundaries = [int(minMaxCoordinates[0]*image_np.shape[0]), int(minMaxCoordinates[1]*image_np.shape[1]), int(minMaxCoordinates[2]*image_np.shape[0]), int(minMaxCoordinates[3]*image_np.shape[1])]
    else:
      imageBoundaries = [0, 0, image_np.shape[1],image_np.shape[0]]
    print(output_dict['detection_boxes'])
    print(image_np.shape)
    print(imageBoundaries)
    print(minMaxCoordinates)
    result = Image.fromarray(cutImage)
    result.save('./data/detected'+image_path.split('./data/'+folder)[1])
    print('saved'+ ('./data/detected'+image_path.split('./data/'+folder)[1]))
    print((image_path.split('./data/'+folder)[1]), ((image_path.split('./data/'+folder)[0])))
    imageId = ((image_path.split('./data/'+folder)[1]).replace("\\", ""))
    imageId = (imageId.replace(".jpg", ""))
    print(imageId)
    imageIndex[counter] = int(imageId)
    boundaries[counter] = imageBoundaries
    ClassPercentage = ''
    currentCounter = np.zeros(545, dtype=np.int64)
    print((len(output_dict['detection_scores'])))
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] >= 0.1:
            ClassPercentage += (str(categories[output_dict['detection_classes'][i]-1]['name']) + 
                            '('+str(categories[output_dict['detection_classes'][i]-1]['id'])+')'+ 
                            ': '+ str(output_dict['detection_scores'][i]) + '\n')
            if(currentCounter[categories[output_dict['detection_classes'][i]-1]['id']] == 0):
                LabelCounter[categories[output_dict['detection_classes'][i]-1]['id']] += 1 
                currentCounter[categories[output_dict['detection_classes'][i]-1]['id']] = 1
    print(ClassPercentage)
    
    
    text_file = open( ( './data/detected'+image_path.split('./data/'+folder)[1]).split('.jpg')[0]+'.txt', "w")
    text_file.write(ClassPercentage)
    text_file.close()
    writeCountedLabelsToFile()
    '''if(flag == 1):
      print(boundaries)
      print(boundaries[counter])
      df = pd.DataFrame(boundaries[counter].reshape(-1, len(boundaries[counter])), columns=['yMin', 'xMin', 'yMax  ', 'xMax'])
      print(df)
    else:'''
    df = pd.DataFrame(boundaries, columns=['yMin', 'xMin', 'yMax  ', 'xMax'])
    print (df)

    dfImageLabel = pd.DataFrame(imageIndex, columns=['imageId'])

    print(dfImageLabel)

    current = pd.concat([dfImageLabel,df], axis=1)
    if(flag == 1):
      old_df_updated = old_df.append(current, ignore_index=True)
      print(old_df_updated)
      old_df_updated.to_csv(os.path.join('./data', 'boundariesCroppedImages'+folder+'.csv'))
      
    else: 
      current.to_csv(os.path.join('./data', 'boundariesCroppedImages'+folder+'.csv'))
      print(current)
    counter += 1
    

    

print(imageIndex)
print(boundaries)
#d = {'xMin' : pd.Series(boundaries[:][0], index=imageIndex)}
#test_merge = pd.DataFrame(test['images'])






#print(df)




