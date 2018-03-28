# Copyright 2018 Brendan Duke.
#
# This file is part of Lintel.
#
# Lintel is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Lintel is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Lintel. If not, see <http://www.gnu.org/licenses/>.

"""Unit test for loadvid."""
import random
import time

import click
import numpy as np
import matplotlib.pyplot as plt

import lintel
import ipdb
import argparse


def _loadvid_test_vanilla(filename, width, height):
    """Tests the usual loadvid call, with a default FPS cap.

    The input file, an encoded video corresponding to `filename`, is repeatedly
    decoded (with a random seek). The first and last of the returned frames are
    plotted using `matplotlib.pyplot`.
    """
    with open(filename, 'rb') as f:
        encoded_video = f.read()

    num_frames = 32
    for _ in range(10):
        start = time.perf_counter()
        decoded_frames, _ = lintel.loadvid(encoded_video,
                                           should_random_seek=True,
                                           width=width,
                                           height=height,
                                           num_frames=num_frames)
        decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
        decoded_frames = np.reshape(decoded_frames,
                                    newshape=(num_frames, height, width, 3))
        end = time.perf_counter()

        print('time: {}'.format(end - start))
        plt.imshow(decoded_frames[0, ...])
        plt.show()
        plt.imshow(decoded_frames[-1, ...])
        plt.show()


def _loadvid_test_frame_nums(filename, width, height):
    """Tests loadvid_frame_nums Python extension.

    `loadvid_frame_nums` takes a list of (strictly increasing, and not
    repeated) frame indices to decode from the encoded video corresponding to
    `filename`.

    This function randomly selects frames to decode, in a loop, decodes the
    chosen frames with `loadvid_frame_nums`, and visualizes the resulting
    frames (all of them) using `matplotlib.pyplot`.
    """
    with open(filename, 'rb') as f:
        encoded_video = f.read()

    num_frames = 2
    for _ in range(1):
        start = time.perf_counter()

        i = 0
        frame_nums = []
        for _ in range(num_frames):
            i += int(random.uniform(1, 4))
            frame_nums.append(i)

        decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                                   frame_nums=frame_nums,
                                                   width=width,
                                                   height=height)
        decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
        decoded_frames = np.reshape(decoded_frames,
                                    newshape=(num_frames, height, width, 3))
        end = time.perf_counter()

        print('time: {}'.format(end - start))
        for i in range(num_frames):
            plt.imshow(decoded_frames[i, ...])
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--filename',
                        default='/Users/fabien/Datasets/VLOG/avi/0/0/I/v_LdhYI_Pq00I/003/clip.mp4',
                        type=str,
                        help='Name of the input video.')
    parser.add_argument('--height',
                        default=368,
                        type=int,
                        help='The _exact_ height of the input video.')
    parser.add_argument('--width',
                        default=654,
                        type=int,
                        help='The _exact_ width of the input video.')
    args = parser.parse_args()

    # _loadvid_test_vanilla(args.filename, args.width, args.height)
    _loadvid_test_frame_nums(args.filename, args.width, args.height)
