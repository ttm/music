"""Warnings that name the line that called into the package.

Python shows a warning with the line it names, and ``stacklevel`` names a
line by counting frames, which is right for one route in only.
`convert_to_stereo` is called by its users and by `pan_transitions`,
`initialize_hunts` by its users and by `PlainChanges`, and `read_audio` by
its users and, under its old name, by `read_wav`. Whatever count each
chose named a line inside the package for some route, and since Python
shows a warning once for each line it names, the first call by such a
route hid the warning from every caller after it.

So the package warns through :func:`warn`, which names the first line
outside the package however deep the call. Python 3.12 can do this itself
(``skip_file_prefixes``); the package supports 3.10.
"""

from __future__ import annotations

import os
import sys
import warnings

#: Where the frames that are not the caller's come from. The mutation
#: audit adds the file of the trampolines it puts in front of each function
#: it mutates, which are frames outside the package.
_SKIPPED = (os.path.dirname(__file__) + os.sep,)


def warn(message: str, category: type[Warning] = UserWarning) -> None:
    """Warn, naming the first line outside the package that led here."""
    frame, level = sys._getframe(1), 2
    while (frame.f_back is not None
           and frame.f_code.co_filename.startswith(_SKIPPED)):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
