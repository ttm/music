* absorb InteratorSynth in amysic.py
* add sympy dependency (and numpy and scipy.io)
  - and colorama
  - and termcolor
linux/bash:
  - and abc2midi (abcmidi in ubuntu)
  - MIDI.pm (libmidi-perl in ubuntu)
  - FFT.pm (sudo cpan install Math::FFT)
  - sox

* make routine to keep the state o music:
  - Annotate if not linux so ecantorix might not be usable

* automate downloading of ecantorix from ttm

* merge main ecantorix with ttm OK

* use home repo/.mass to clone eCantorix
  - Simply write a function in utils.py to:
    * create a ~/.mass directory
    * clone ttm/eCantorix inside it
  - use the correct paths in singing/ module
  - ask user if it is ok to clone repo when she
  tries to use the singing module and no ~/.mass/ecantorix
  directory is found

* How to make exponential fades reach 0??
