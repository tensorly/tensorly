# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Testing the rst files generator
"""
from __future__ import division, absolute_import, print_function
import os
import sphinx_gallery.docs_resolv as sg
import tempfile
import sys


def test_shelve():
    """Test if shelve can be caches information
    retrieved after file is deleted"""
    test_string = 'test information'
    tmp_cache = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(test_string)
    try:
        # recovers data from temporary file and caches it in the shelve
        file_data = sg.get_data(f.name, tmp_cache)
    finally:
        os.remove(f.name)
    # tests recovered data matches
    assert file_data == test_string

    # test if cached data is available after temporary file has vanished
    assert sg.get_data(f.name, tmp_cache) == test_string

    # shelve keys need to be str in python 2, deal with unicode input
    if sys.version_info[0] == 2:
        unicode_name = unicode(f.name)
        assert sg.get_data(unicode_name, tmp_cache) == test_string
