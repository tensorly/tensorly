# -*- coding: utf-8 -*-
r"""
Sorters for Sphinx-Gallery subsections
======================================

Sorting key functions for gallery subsection folders
"""
# Created: Sun May 21 20:38:59 2017
# Author: Óscar Nájera
# License: 3-clause BSD

from __future__ import division, absolute_import, print_function
import os
import types


class ExplicitOrder(object):
    """Sorting key for all galleries subsections

    This requires all folders to be listed otherwise an exception is raised

    Parameters
    ----------
    ordered_list : list, tuple, types.GeneratorType
        Hold the paths of each galleries' subsections

    Raises
    ------
    ValueError
        Wrong input type or Subgallery path missing
    """

    def __init__(self, ordered_list):
        if not isinstance(ordered_list, (list, tuple, types.GeneratorType)):
            raise ValueError("ExplicitOrder sorting key takes a list, "
                             "tuple or Generator, which hold"
                             "the paths of each gallery subfolder")

        self.ordered_list = list(os.path.normpath(path)
                                 for path in ordered_list)

    def __call__(self, item):
        if item in self.ordered_list:
            return self.ordered_list.index(item)
        else:
            raise ValueError('If you use an explicit folder ordering, you '
                             'must specify all folders. Explicit order not '
                             'found for {}'.format(item))
