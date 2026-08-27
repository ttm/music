# -*- coding: utf-8 -*-
"""Utilities to synthesize singing from text using eCantorix."""

import re
import logging
import shutil
import subprocess
from scipy.io import wavfile
from music.core import normalize_mono
from .paths import (ENGINE_MARKER, cache_dir, engine_dir, is_engine,
                    require_system_dependencies)


# def sing(text="ba-na-nin-ha pra vo-cê",
def sing(text="Mar-ry had a litt-le lamb",
         notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2),
         M='4/4', L='1/4', Q=120, K='C', reference=60,
         lang='en', transpose=-36, effect=None):
    #         lang='pt', transpose=-36, effect=None):
    # write abc file, write make file, convert to midi, sing it out
    # reference -= 24
    engine = engine_dir()
    cache = cache_dir()
    if not is_engine(engine):
        detail = (f"the directory exists but has no {ENGINE_MARKER}"
                  if engine.is_dir() else "nothing is there")
        raise RuntimeError(
            f"no usable eCantorix engine at {engine}: {detail}. "
            "Run music.singing.setup_engine() to install it."
        )
    require_system_dependencies()
    cache.mkdir(parents=True, exist_ok=True)

    write_abc(text, notes, durs, M=M, L=L, Q=Q, K=K, reference=reference)
    conf_text = '$ESPEAK_VOICE = "{}";\n'.format(lang)
    conf_text += '$ESPEAK_TRANSPOSE = {};'.format(transpose)
    if effect == 'flint':
        conf_text += "\ndo 'extravoices/flite.inc';"
    elif effect == 'tremolo':
        conf_text += "\ndo 'extravoices/tremolo.inc';"
    elif effect == 'melt':
        conf_text += "\ndo 'extravoices/melt.inc';"
    elif effect:
        raise ValueError('effect not understood')
    with open(cache / 'achant.conf', 'w') as f:
        f.write(conf_text)
    try:
        shutil.copy(engine / 'Makefile', cache / 'Makefile')
    except OSError as exc:
        raise RuntimeError(f'Failed to prepare singing cache: {exc}') from exc
    try:
        subprocess.run(['make', '-C', str(cache)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Failed to build singing cache: {exc}') from exc

    sample_rate, samples = wavfile.read(cache / 'achant.wav')
    if sample_rate != 44100:
        raise RuntimeError(
            f'expected the engine to render at 44100 Hz, got {sample_rate}'
        )
    return normalize_mono(samples)


def write_abc(text, notes, durs, M='4/4', L='1/4', Q=120, K='C', reference=60):
    text_ = 'X:1\n'
    text_ += 'T:Some chanting for music python package\n'
    text_ += 'M:{}\n'.format(M)
    text_ += 'L:{}\n'.format(L)
    text_ += 'Q:{}\n'.format(Q)
    text_ += 'V:1\n'
    text_ += 'K:{}\n'.format(K)
    notes = translate_to_abc(notes, durs, reference)
    text_ += notes + "\nw: " + text
    fname = cache_dir() / "achant.abc"
    with open(fname, 'w') as f:
        f.write(text_)


def translate_to_abc(notes, durs, reference):
    durs = [str(i).replace('-', '/') for i in durs]
    durs = [i if i != '1' else '' for i in durs]
    notes = converter.convert(notes, reference)
    return ''.join([i + j for i, j in zip(notes, durs)])


class Notes:
    def __init__(self):
        self.notes_dict = None
        self.make_dict()

    def make_dict(self):
        notes = re.findall(r'[\^=]?[a-g]', '=c^c=d^de=f^f=g^g=a^ab')
        # notes=re.findall(r'[\^]{0,1}[a-g]{1}','a^abc^cd^def^fg^g')
        notes_ = [note.upper() for note in notes]
        notes__ = [note + "," for note in notes_]
        notes___ = [note + "," for note in notes__]
        notes____ = [note + "," for note in notes___]
        notes_u = [note + "'" for note in notes]
        notes__u = [note + "'" for note in notes_u]
        notes___u = [note + "'" for note in notes__u]
        notes_all = notes____ + notes___ + notes__ + notes_ + notes + \
            notes_u + notes__u + notes___u
        self.notes_dict = dict([(i, j) for i, j in zip(range(12, 97),
                                                       notes_all)])

    def convert(self, notes, reference):
        if 'notes_dict' not in dir(self):
            self.make_dict()
        notes_ = [reference + note for note in notes]
        notes__ = [self.notes_dict[note] for note in notes_]
        return notes__


converter = Notes()

if __name__ == '__main__':
    narray = sing()
    logging.info("finished")
