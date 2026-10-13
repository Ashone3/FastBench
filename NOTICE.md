# Notices

This release contains code adapted from the PhoStream evaluation repository. Preserve the original copyright and license notices when redistributing derived code.

The public launchers can load checkpoints and components maintained by independent projects, including Qwen3-VL, AURA, JoyAI-VL-Interaction, MOSS-VL-Realtime, and VideoChat3. Their model weights, model code, and data are subject to the licenses and terms published by their respective authors.

The optional MOSS framewise queue synchronization in `moss_framewise.py` is adapted from the OpenMoss team's `ShacklesLay/FastBench` fork, commit `959a4779905257a6314bfad1ab056937183035a2`, especially `moss_native/transport.py`, `gate.py`, and `server.py`. We thank Chenkun and the OpenMoss team at Fudan University for the native-interface adaptation, reproduction and supporting traces. The source's MIT notice is retained in the adapted module. FastBench's integration keeps its own chunk boundaries and scoring timestamps; see `docs/native_interfaces.md`.
