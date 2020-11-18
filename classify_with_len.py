#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os
import cv2
import numpy
import string
import random
import argparse
#import tensorflow as tf
#import tensorflow.keras as keras
import tflite_runtime.interpreter as tflite
import itertools

def decode(characters, y, len_y):
    y_idx = numpy.argmax(numpy.array(y), axis=1)
    y_pred = numpy.max(numpy.array(y), axis=1)
     
    cap_len = numpy.argmax(numpy.array(len_y)) + 1

    y_chars = numpy.argsort(y_pred)[-cap_len:]
    
    res = ''.join([characters[x] for i,x in enumerate(y_idx) if i in y_chars])
    return res

def decode_fix(characters, y):
    y_idx = numpy.argmax(numpy.array(y), axis=1)
    y_pred = numpy.max(numpy.array(y), axis=1)
    #res = ''.join([characters[x] for x in y_idx])
    res = ''.join([characters[x] for i,x in enumerate(y_idx) if y_pred[i]>0.65])
    return res

def decode_mix(characters, y):
    y_idx = numpy.argmax(numpy.array(y), axis=1)
    sym_len = len(characters)
    res = ''.join([characters[x] for i,x in enumerate(y_idx) if x < sym_len])
    return res
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--len-model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)
    
    if args.len_model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    #with tf.device('/cpu:0'):
    if True:
        with open(args.output, 'w') as output_file:
            # char length tf model
            # len_json_file = open(args.len_model_name+'/model.json', 'r')
            # len_loaded_model_json = len_json_file.read()
            # len_json_file.close()
            # len_model = keras.models.model_from_json(len_loaded_model_json)
            # len_model.load_weights(args.len_model_name+'/model_checkpoint.h5')
            # len_model.compile(loss='categorical_crossentropy',
            #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            #               metrics=['accuracy'])

            # char pred tf model
            # json_file = open(args.model_name+'/model.json', 'r')
            # loaded_model_json = json_file.read()
            # json_file.close()
            # model = keras.models.model_from_json(loaded_model_json)
            # model.load_weights(args.model_name+'/model_checkpoint.h5')
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            #               metrics=['accuracy'])

            #char length tflite model
            len_interpreter = tflite.Interpreter(args.len_model_name+'/model_len.tflite')
            len_interpreter.allocate_tensors()

            len_input_d = len_interpreter.get_input_details()
            len_output_d = len_interpreter.get_output_details()

            
            # char pref tflite model
            char_interpreter = tflite.Interpreter(args.model_name+'/model_char.tflite')
            char_interpreter.allocate_tensors()

            char_input_d = char_interpreter.get_input_details()
            char_output_d = char_interpreter.get_output_details()

            

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data, dtype=numpy.float32) / 255.0
                (c, h, w) = image.shape
                # assuming that input will have same size as of trained image
                image = image.reshape([-1, c, h, w])

                #length part
                len_interpreter.set_tensor(len_input_d[0]['index'], image)
                len_interpreter.invoke()

                len_prediction = len_interpreter.get_tensor(len_output_d[0]['index'])

                
                # prediction = model.predict(image)
                # len_prediction = len_model.predict(image)

                #predict from char-model
                char_interpreter.set_tensor(char_input_d[0]['index'], image)
                char_interpreter.invoke()
                prediction = []
                for output_node in char_output_d:
                    prediction.append(char_interpreter.get_tensor(output_node['index']))
                
                prediction = numpy.reshape(prediction, (len(char_output_d),-1))

                # res = decode(captcha_symbols, prediction,len_prediction)
                # res = decode_fix(captcha_symbols, prediction)
                res = decode_mix(captcha_symbols, prediction)
                output_file.write(x + "," + res + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
