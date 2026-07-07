# modesearch — mode database and query layer (design stage)

Status: **descriptive text only.** No schema, storage engine, or query API has
been chosen; this document records intent so the other suite tools can grow
toward it.

## The idea

1. **Offline crawler / cataloguer.** A batch job walks shot archives (local
   HDF5 stores, MDSplus when reachable) and runs the TokEye suite on each
   shot: `big_tf_unet` masks, `modespec` toroidal mode-number fits, `elmspec`
   ELM events, `alfvenspec` AE detections. Every detection becomes a mode
   record in the database.
2. **Mode record.** One row per mode event — the working sketch:
   shot, machine, diagnostic, time interval, frequency band, mode numbers
   (n, and m when available), amplitude, confidence, detector name + version,
   and a pointer back to the artifact (mask file, CSV row) that produced it.
   A shared record type that all suite tools can emit is the first concrete
   deliverable (see docs/ROADMAP.md, "Mode catalogue schema").
3. **Query layer.** "Find shots with an n=2 tearing mode between 2-4 kHz
   during an ELM-free period" — filters on the record fields, returning shot
   lists with the matching events. Interface undecided (CLI first, probably).
4. **Consumers.** Researchers hunting for reference shots; validation studies
   (mode statistics vs. campaign); and the lab's fusion-world-model shot
   designer, which can learn mode occurrence statistics conditioned on
   plasma parameters from the same index.

## Relationship to shotsearch

The sibling `shotsearch` project answers "which discharges look like this
setup?" (actuator/setup similarity). modesearch answers "which discharges
contained this MHD activity?". They meet at the shot list: a designer query
could intersect both ("shots near this setup that developed an n=1 locked
mode").

## Non-goals for v1

Real-time/inter-shot operation, cross-machine schema unification, and
automatic mode labeling beyond what the detectors already emit.
