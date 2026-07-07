"""modesearch — a searchable database of detected modes (design stage).

Nothing is implemented here yet; this package reserves the name and records
the intended design. See README.md in this directory and docs/ROADMAP.md.

The idea: an offline crawler runs the TokEye suite (big_tf_unet masks,
modespec mode-number fits, elmspec ELM events, alfvenspec AE detections)
over shot archives and indexes every detected mode into a database — one
record per mode event: shot, machine, time interval, frequency band, mode
numbers when known, amplitude, and detector provenance. Researchers then
query it ("find shots with an n=2 tearing mode between 2-4 kHz during an
ELM-free period") instead of re-scanning raw data, and the lab's
fusion-world-model shot designer can pull mode statistics from the same
index. Complements the sibling ``shotsearch`` project, which searches
discharges by actuator/setup similarity rather than by MHD activity.
"""

from __future__ import annotations
