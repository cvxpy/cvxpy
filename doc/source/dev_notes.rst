.. _dev_notes:

Developer Notes
===============

`autodoc`
---------
When using autodoc with latex in docstrings, make sure to use raw strings, such as

.. code-block:: python

    r""" This is a docstring.

    With more details
    """

Otherwise, you need to use double backslashes for latex commands.
