# -*- coding: utf-8 -*-
r"""
Parser for python source files
==============================
"""
# Created Sun Nov 27 14:03:07 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import ast
import re
from textwrap import dedent

SYNTAX_ERROR_DOCSTRING = """
SyntaxError
===========

Example script with invalid Python syntax
"""


def get_docstring_and_rest(filename):
    """Separate `filename` content between docstring and the rest

    Strongly inspired from ast.get_docstring.

    Returns
    -------
    docstring: str
        docstring of `filename`
    rest: str
        `filename` content without the docstring
    """
    # can't use codecs.open(filename, 'r', 'utf-8') here b/c ast doesn't
    # seem to work with unicode strings in Python2.7
    # "SyntaxError: encoding declaration in Unicode string"
    with open(filename, 'rb') as fid:
        content = fid.read()
    # change from Windows format to UNIX for uniformity
    content = content.replace(b'\r\n', b'\n')

    try:
        node = ast.parse(content)
    except SyntaxError:
        return SYNTAX_ERROR_DOCSTRING, content.decode('utf-8'), 1

    if not isinstance(node, ast.Module):
        raise TypeError("This function only supports modules. "
                        "You provided {0}".format(node.__class__.__name__))
    if node.body and isinstance(node.body[0], ast.Expr) and \
       isinstance(node.body[0].value, ast.Str):
        docstring_node = node.body[0]
        docstring = docstring_node.value.s
        if hasattr(docstring, 'decode'):  # python2.7
            docstring = docstring.decode('utf-8')
        lineno = docstring_node.lineno  # The last line of the string.
        # This get the content of the file after the docstring last line
        # Note: 'maxsplit' argument is not a keyword argument in python2
        rest = content.decode('utf-8').split('\n', lineno)[-1]
        return docstring, rest, lineno + 1
    else:
        raise ValueError(('Could not find docstring in file "{0}". '
                          'A docstring is required by sphinx-gallery')
                         .format(filename))


def extract_file_config(content):
    """
    Pull out the file-specific config specified in the docstring.
    """

    prop_pat = re.compile(
        r"^\s*#\s*sphinx_gallery_([A-Za-z0-9_]+)\s*=\s*(.+)\s*$",
        re.MULTILINE)
    file_conf = {}
    for match in re.finditer(prop_pat, content):
        name = match.group(1)
        value = match.group(2)
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            logger.warning(
                'Sphinx-gallery option %s was passed invalid value %s',
                name, value)
        else:
            file_conf[name] = value
    return file_conf


def split_code_and_text_blocks(source_file):
    """Return list with source file separated into code and text blocks.

    Returns
    -------
    file_conf : dict
        File-specific settings given in comments as:
        # sphinx_gallery_<name> = <value>
    blocks : list of (label, content)
        List where each element is a tuple with the label ('text' or 'code'),
        and content string of block.
    """
    docstring, rest_of_content, lineno = get_docstring_and_rest(source_file)
    blocks = [('text', docstring, 1)]

    file_conf = extract_file_config(rest_of_content)

    pattern = re.compile(
        r'(?P<header_line>^#{20,}.*)\s(?P<text_content>(?:^#.*\s)*)',
        flags=re.M)
    sub_pat = re.compile('^#', flags=re.M)

    pos_so_far = 0
    for match in re.finditer(pattern, rest_of_content):
        code_block_content = rest_of_content[pos_so_far:match.start()]
        if code_block_content.strip():
            blocks.append(('code', code_block_content, lineno))
        lineno += code_block_content.count('\n')

        lineno += 1  # Ignored header line of hashes.
        text_content = match.group('text_content')
        text_block_content = dedent(re.sub(sub_pat, '', text_content)).lstrip()
        if text_block_content.strip():
            blocks.append(('text', text_block_content, lineno))
        lineno += text_content.count('\n')

        pos_so_far = match.end()

    remaining_content = rest_of_content[pos_so_far:]
    if remaining_content.strip():
        blocks.append(('code', remaining_content, lineno))

    return file_conf, blocks
