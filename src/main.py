import cv2
import os
import pickle
import sys

import numpy

import data
import network as n
import preprocessing as pp
import linesegment as ls

def list_files(directory):
    if not os.path.isdir(directory):
        print("{} is not a valid directory")
        sys.exit(-2)
    
    return os.listdir(directory)

if __name__ == "__main__2":
    dat = n.list_characters(testnames, easy_pkl)
    
    for name, chars in dat.items():
        print("File: {}\n    ".format(name), end="")
        for c in chars:
            print("{} ".format(c.name), end="")
        print()
        

if __name__ == "__main__":
    if sys.version_info[0] < 3:
        print("Please use python 3...")
        print("    Example usage: python3 path/to/directory/")
        sys.exit(-1)

    if not len(sys.argv) == 2:
        print("Incorrect ammount of arguments, required: 1")
        print("    Example usage: python3 path/to/directory/")
        sys.exit(-1)

    directory = sys.argv[1]
    files = list_files(directory)
    images = []

    for f in files:
        if(f.endswith(".txt")): continue
        images.append(data.Image(directory, f))


    for img in images:
        img.load_processed()
        img.segment_lines()

    namelist = n.write_files(images)
    print(namelist)
    n.run_network()
    output = n.list_characters(namelist, "network/Data/RBA/Outputs/TEST_boxes_classes.pkl")

    print("Writing detection output to files...", end="")

    for img in images:
        img.output_annotation(output)

    print("done!")
    print("All tasks complete, exiting...")
