#! /usr/bin/env python

import argparse
import os
import cv2
# from tqdm import tqdm
import keras.utils
from .utils import draw_boxes
from .frontend import YOLO
import json
import threading
from queue import Queue
import numpy as np

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def main():
    args = argparser.parse_args()
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'],
                weight_path=config['model']['weight_path'])

    ###############################
    #   Load trained weights
    ###############################

    print("weights", weights_path)
    print("cwd", os.getcwd())
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)
        meta_file = image_path[:-4] + '.json'

        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        sequence = VideoFrameSequence(video_reader, 150)
        results = yolo.predit_generator(sequence,
                                        use_multiprocessing=False,
                                        steps=None,
                                        workers=6)
        print("Results gathered")
        video_writer = cv2.VideoWriter(
            video_out,
            cv2.VideoWriter_fourcc(*'MPEG'),
            50.0, (frame_w, frame_h))

        sequence.stream.stop()
        video_reader.release()
        video_reader = cv2.VideoCapture(image_path)
        sequence = VideoFrameSequence(video_reader, 150)
        images = (sequence[i] for i in range(len(sequence)))
        metadata = {"frames": []}
        i = 0
        for image, boxes in zip(images, results):
            found_boxes = []
            for b in boxes:
                label = b.get_label()
                score = b.get_score()

                found_boxes.append(
                    {
                        "label": config['model']['labels'][label],
                        "score": float(score),
                        "xmin": float(b.xmin),
                        "xmax": float(b.xmax),
                        "ymin": float(b.ymin),
                        "ymax": float(b.ymax),
                    })

                metadata["frames"].append({"number": i, "boxes": found_boxes})
                i += 1

            image = draw_boxes(
                image, boxes, config['model']['labels'], threshold=0.35)
            video_writer.write(np.uint8(image))
        print(type(sequence.max_frames))
        metadata["total_frames"] = int(sequence.max_frames)
        video_writer.release()
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        for b in boxes:
            label = b.get_label()
            score = b.get_score()
            print(f"Box {label} {score}")

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


class VideoFrameSequence(keras.utils.Sequence):
    """docstring for VideoSequence"""

    def __init__(self, video_reader, subsampling=1):
        super(VideoFrameSequence, self).__init__()
        self.subsampling = subsampling
        self.max_frames = int(video_reader.get(
            cv2.CAP_PROP_FRAME_COUNT)//subsampling)
        self.video_reader = video_reader
        self.stream = FileVideoStream(
            SubsamplingReader(self.video_reader, subsampling), self.max_frames)
        self.stream.start()
        print("Total frames", len(self))

    def __len__(self):
        return int(self.max_frames)

    def __getitem__(self, index):
        frame = self.stream.read()
        return frame


class SubsamplingReader(object):
    """docstring for SubsamplingReader"""

    def __init__(self, video_reader, subsampling):
        super(SubsamplingReader, self).__init__()
        self.video_reader = video_reader
        self.subsampling = subsampling
        self.position = 0
        self.lock = threading.Lock()

    def read(self, index=None):
        if index is None:
            index = self.position
        self.lock.acquire()
        if index != self.position or self.subsampling != 1:
            self.video_reader.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.position = index

        grabbed, frame = self.video_reader.read()
        self.position += self.subsampling
        # TODO have a producer thread to improve this
        self.lock.release()
        return grabbed, frame


class FileVideoStream:
    def __init__(self, video_reader, max_frames, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = video_reader
        self.stopped = False
        self.max_frames = max_frames

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self):
        for i in range(self.max_frames):
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            # if not self.Q.full():
            # read the next frame from the file
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                print("not grabbed")
                return

            # add the frame to the queue
            self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    main()
