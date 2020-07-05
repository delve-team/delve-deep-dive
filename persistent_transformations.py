# Original work Copyright (c) 2019 Christoph Hofer
# Modified work Copyright (c) 2019 Wolf Byttner
#
# This file is part of the code implementing the thesis
# "Classifying RGB Images with multi-colour Persistent Homology".
#
#     This file is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published
#     by the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This file is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this file.  If not, see <https://www.gnu.org/licenses/>.

import scipy
import skimage.morphology

import multiprocessing
import os
import sys
import numpy as np

import torch

sys.path.append(os.path.join(os.getcwd(), "tda-toolkit"))

from pershombox import calculate_discrete_NPHT_2d


def threshold_diagram(persistence_diagram):
    t = 0.01
    return [pdgm for pdgm in persistence_diagram if pdgm[1] - pdgm[0] > t]


def calculate_npht_2d(monochrome_image, directions):
    label_map, count = skimage.morphology.label(np.squeeze(monochrome_image),
                                                connectivity=1,
                                                background=0,
                                                return_num=True)
    volumes = [np.count_nonzero(label_map == (i + 1)) for i in range(count)]
    arg_max = np.argmax(volumes)
    label_image = (label_map == (arg_max + 1))
    label_image = np.ndarray.astype(label_image, bool)
    return calculate_discrete_NPHT_2d(label_image, directions)


def calculate_rotated_diagrams(monochrome_image, directions, homology_dimensions):
    diagrams = list()
    error = None
    npht = calculate_npht_2d(monochrome_image, directions)
    for diagram in npht:
        for dimension in homology_dimensions:
            try:
                diagrams.append(list(threshold_diagram(diagram[dimension])[0]))
            except Exception as e:
                # TODO fix diagram length
                diagrams.append([0, 0])
                print(e)

    if len(diagrams) != directions * len(homology_dimensions):
        raise ValueError('Diagram is genenerate. Transformation failed')
    return torch.tensor(diagrams)


class ToRotatedPersistenceDiagrams(object):
    """Convert a monochrome image to a set of rotated persistence diagrams.

    Args:
        directions (int): Number of directions to rotate the persistence diagrams.
        homology_dimension (sequence): Dimensions of the persistent homology to compute.

    """
    def __init__(self, directions=32, homology_dimensions=[0]):
        self.directions = directions
        self.homology_dimensions=homology_dimensions

    def __call__(self, img):
        return calculate_rotated_diagrams(img, self.directions, self.homology_dimensions)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.directions, self.homology_dimensions)
