#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.applications import (xception,
                                inception_v3,
                                resnet50,
                                inception_resnet_v2,
                                densenet)
from keras.layers import Dense
from argparse import ArgumentParser
import sys
from fashion_code.callbacks import F1Utility
from fashion_code.constants import num_classes, paths, GCP_paths
from fashion_code.generators import SequenceFromDisk, SequenceFromGCP
from keras.models import Model, load_model
from keras.optimizers import Adam
from os.path import join


# Maps network names to module, constructor, input size
networks = {
    'xception': (xception, xception.Xception, 299),
    'resnet50': (resnet50, resnet50.ResNet50, 224),
    'inceptionv3': (inception_v3, inception_v3.InceptionV3, 299),
    'inceptionresnetv2': (inception_resnet_v2, inception_resnet_v2.InceptionResNetV2, 299),
    'densenet201': (densenet, densenet.DenseNet201, 224),
}


#Right now these models are all pretrained and not finetuned.
#For this matter, they all even have the same top layers. 
def build_model(NetworkConstr, num_classes):
    base_model = NetworkConstr(weights='imagenet', pooling='avg', include_top=False)
    x = base_model.output
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(num_classes, activation='sigmoid',
                        kernel_initializer='he_normal')(x)
    model = Model(base_model.inputs, predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model 
    
def train_model(NetworkConstr,preprocess_fun ,args):
      
    # Some variables
    batch_size = args.batch_size
    chpt = args.continue_from_chpt
    epochs = args.epochs
    img_size = (299, 299)
    loss = 'binary_crossentropy'
    optimizer = Adam(decay=1e-6)
    use_multiprocessing = not args.windows
    workers = 0 if args.windows else 8
    if args.gcp:
        path_dict = GCP_paths
    else:
        path_dict = paths

    # Create and compile model
    if chpt:
        model = load_model(join(paths['models'], chpt))
    else:
        model = build_model(NetworkConstr, num_classes)
        model.compile(optimizer=optimizer, loss=loss)

    # Create data generators
    if args.gcp:
        train_gen = SequenceFromGCP('train', batch_size, img_size,
                                    preprocessfunc=preprocess_fun)
        valid_gen = SequenceFromGCP('validation', batch_size, img_size,
                                    preprocessfunc=preprocess_fun)
        if args.create_submission:
            test_gen = SequenceFromGCP('test', batch_size, img_size,
                                       preprocessfunc=preprocess_fun)
        else:
            test_gen = None
    else:
        train_gen = SequenceFromDisk('train', batch_size, img_size,
                                     preprocessfunc=preprocess_fun)
        valid_gen = SequenceFromDisk('validation', batch_size, img_size,
                                     preprocessfunc=preprocess_fun)
        if args.create_submission:
            test_gen = SequenceFromDisk('test', batch_size, img_size,
                                        preprocessfunc=preprocess_fun)
        else:
            test_gen = None

    # Fit model
    pm = F1Utility(valid_gen, test_generator=test_gen,
                   save_path=path_dict['models/stacks'], save_fname=args.save_filename)

    train_steps = args.train_steps or len(train_gen)

    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_steps,
                        use_multiprocessing=use_multiprocessing,
                        workers=workers,
                        # This callback does validation, checkpointing and
                        # submission creation
                        callbacks=[pm],
                        verbose=1)


if __name__ == '__main__':
    p = ArgumentParser('Simple stacking script.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size of images to read into memory')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of epochs to train')
    p.add_argument('--train-steps', type=int, help='Number of batches to '
                                                   'train on each epoch; if '
                                                   'not given, train on full '
                                                   'training set')
    p.add_argument('--save-filename', type=str,
                   help='Filename for saved model (will be appended to models '
                        'directory)')
    p.add_argument('--continue-from-chpt', type=str,
                   help='Continue training from h5 checkpoint file')
    p.add_argument('--create-submission', action='store_true',
                   help='If given, create a submission after training')
    p.add_argument('--windows', action='store_true',
                   help='Disable multiprocessing in order to run on Windows')
    p.add_argument('--gcp', action='store_true',
                   help='Change file loading for Google Cloud Platform')
    p.add_argument('--job-dir', type=str, help='Location of the job directory '
                                               'for the current GCP job')
    args = p.parse_args()
    
    
    for net in networks.keys():
        NetworkConstr = networks[f'{net}'][1]
        preprocess_fun = networks[f'{net}'][0].preprocess_input
        args.save_filename = f'{net}'
        print(args.save_filename)
        train_model(NetworkConstr,preprocess_fun,args)

    sys.exit(0)    

    
# KEEP: possibly comes in handy when top layers are different among networks

#class network_build(object):
#    def build_model(self, argument):
#        """network handler
#            self:: 
#            argument:: the name of the network (see list below)
#        """
#        method_name = 'build_' + str(argument)
#        # Get the method from 'self'. Default to a lambda.
#        method = getattr(self, method_name, lambda: "Invalid model")
#        # Call the method as we return it
#        return method()
# 
#    def build_Xception(self):
#        self.base_model = Xception(weights='imagenet', pooling='avg', include_top=False)
#        return self.base_model
# 
#    def build_InceptionV3(self):
#        self.base_model = InceptionV3(weights='imagenet', pooling='avg', include_top=False)
#        return self.base_model
#
#    def build_resnet50(self):
#        self.base_model = ResNet50(weights='imagenet', pooling='avg', include_top=False)
#        return self.base_model
#    
#    def build_InceptionResNetV2(self):
#        self.base_model = InceptionResNetV2(weights='imagenet', pooling='avg', include_top=False)
#        return self.base_model
#    
#    def build_densenet201(self):
#        self.base_model = DenseNet201(weights='imagenet', pooling='avg', include_top=False)
#        return self.base_model
#
##For now, just use this architecture for all of the pretrained networks without
## any diversifciation between networks. Change later on
#def add_start_layers(base_model,num_classes):
#    x = base_model.output
#    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
#    predictions = Dense(num_classes, activation='sigmoid',
#                        kernel_initializer='he_normal')(x)
#    model = Model(base_model.inputs, predictions)
#    
#    for layer in base_model.layers:
#        layer.trainable = False
#    
#    return model
#        
#        
#print('attempting to load model')
#x = Switcher()
#x = x.build_model('Xception')
#
#model = add_start_layers(x,228)
#x.summary()
#model.summary()