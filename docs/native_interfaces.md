# Native MOSS framewise and AURAv2 interfaces

## MOSS-VL-Realtime framewise

The optional `--native-moss-framewise` path adapts the queue synchronization in
[OpenMoss's fork](https://github.com/ShacklesLay/FastBench/tree/959a4779905257a6314bfad1ab056937183035a2/moss_native).
It uses the fork's realtime system prompt, a persistent native session, and one
frame per input. Historical frames are ingested continuously, including chunks
outside the old sparse response window. Every frame and text input waits for a
**fresh model-generated `<|silence|>` after ingestion** before advancing. Startup
silence and synthetic queue silence cannot acknowledge a new input. The native
updater is restored when a session closes. Checkpoint weights are not changed.

This is an adaptation to FastBench's existing chunk protocol, with these differences:

| Aspect | Team fork at the referenced commit | This optional FastBench path |
| --- | --- | --- |
| Question placement | Text at its input-event time, before the following visual frame | Text before the **last permitted frame of the chunk ending at the question time** |
| Visual boundary for a question at Q | The next frame may be timestamped Q | Frames strictly inside `[Q-1, Q)` at the default one-second chunk size; exclude the Q boundary frame |
| Question synchronization | First native decoded text or generated silence | Fresh generated silence, including completion of the full answer |
| Scored output time | Generating input event, including fractional frame times | Existing integer chunk-end timestamp; generated output is collected within that chunk |
| Scoring | Fork supplies input-event reconstruction and fractional timestamp handling | Existing FastBench response selection, judge and early/late rules |

For a question annotated at **10 seconds**, default 2 FPS input is:

```text
9.0-second frame -> generated silence
question text   -> answer and/or generated silence
9.5-second frame -> answer and/or generated silence
collect this chunk's outputs with benchmark timestamp 00:10
```

This deliberately advances **text delivery within the chunk**, by one frame
interval at 2 FPS. It does not edit question/answer annotations or move a response
back to a fractional timestamp. No 10.0-second frame is supplied in this request,
and the next chunk waits until this chunk finishes. This retains the same
per-chunk visual cutoff as the other benchmark paths, while allowing MOSS's
text-before-vision interaction. It should not be described as an identical replay
of the fork or as chronological text delivery at its annotation time.

`question_time == answer_time` still makes that time the response deadline;
there is no additional four-second grace period for those questions. The four-second
window parameter retains its existing application to later answer events.
Benchmark timestamps represent input video time, not GPU or HTTP wall-clock latency.
The original `--native-moss` script remains available, and the new protocol is opt-in.

From the release root, with the MOSS-compatible environment activated:

```bash
export MODEL_PATH=/path/to/MOSS-VL-Realtime
export NUM_GPUS=8 BASE_PORT=8000
bash scripts/restart_hf_servers_clean.sh moss
VIDEO_ROOT="$PWD" NUM_WORKERS=8 NUM_SCORERS=20 \
bash scripts/eval_moss_vl_realtime_framewise.sh --run-id moss-framewise
```

Equivalent opt-in with the original public launcher:

```bash
EXTRA_ARGS='--native-moss-framewise' VIDEO_ROOT="$PWD" \
bash scripts/eval_moss_vl_realtime.sh
```

Use `EXTRA_ARGS='--dialog-dump-dir /path/to/dialog'` to retain raw HTTP request and
response dumps. EXTRA_ARGS accepts shell-style quoting but does not execute shell
expressions. `--dry-run` prints the evaluation command without contacting a server.
Failures and timeouts abort the affected sample rather than treating incomplete
output as ordinary silence. Failed persistent requests are not retried because
inputs may already have advanced the native cache. Native exception chains are
logged for diagnosis. Check the successful sample count and resume failed samples
before interpreting aggregate scores.

The launcher rejects occupied ports before loading weights and verifies the
framewise protocol and PID of each new process. The evaluation checks protocol
compatibility before distributing samples. It does not accept `/health` alone as
proof of a compatible MOSS deployment.

## AURAv2

AURAv2 checkpoints based on **Qwen3.5** are loaded with
`Qwen3_5ForConditionalGeneration`, not the Qwen3-VL class used by the original AURA
entry point. The checkpoint processor receives per-video metadata with frame
indices and FPS; the supplied frames are not sampled twice. Generation retains
the checkpoint's EOS tokens and also recognizes the tokenizer's `<|im_end|>` turn
terminator. Native AURA silence tokens remain visible during decoding.

Use a separate AURAv2 environment. The legacy `requirements.txt` pins
Transformers 4.57.3 and does not provide the Qwen3.5 class. Install that environment's
base dependencies and then the AURAv2 overlay:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-aurav2.txt
python -c 'from transformers import Qwen3_5ForConditionalGeneration'
```

Additional optimized kernels may be required by the checkpoint; follow its runtime
instructions. Keep MOSS in its own checkpoint-compatible environment rather than
upgrading its remote-code runtime for AURAv2.

```bash
export MODEL_PATH=/path/to/AURAv2/checkpoint
export NUM_GPUS=8 BASE_PORT=8000
bash scripts/restart_hf_servers_clean.sh aurav2
VIDEO_ROOT="$PWD" NUM_WORKERS=8 NUM_SCORERS=20 \
bash scripts/eval_aurav2.sh --run-id aurav2
```

This preset preserves the **original native AURA system prompt**. It selects
`configs/aurav2.yaml`, which inherits the release configuration and adds 13 exact
acknowledgement/meta responses plus two promise-prefix patterns. These treat
phrases such as "I'll keep an eye out" as placeholders so they do not consume the
answer slot. Every sentence/clause must match for a prefix-based placeholder;
a separately stated factual answer remains eligible. Examples and factual
exclusions are in `configs/aurav2_placeholder_examples.json`.

The patterns match wording rather than semantic meaning; facts within the same
promise clause can still match. Record the configuration in reported results.
These additions do not change timing deadlines, judge prompts or baseline
placeholder matching when the dedicated configuration is not selected.

`--native-aurav2-prompt` remains an optional prompt ablation requiring
`--native-aura`; the AURAv2 baseline script does **not** enable it.

## Clean deployment and directories

`NUM_GPUS=8 BASE_PORT=8000 bash scripts/stop_hf_servers_clean.sh` stops listeners
and HF processes still loading on ports 8000–8007, waits for exit and uses SIGKILL
if needed. The public helper defaults to **one** GPU/port. Select the intended
range explicitly when switching experiments. This operation terminates services
on that range, including services started from another checkout; run it between
evaluations.

`restart_hf_servers_clean.sh` creates a fresh log/PID directory and verifies
`/server-info` against each newly launched PID, checkpoint path and model type.
It supports `moss`, `aura`, `aurav2`, and `joyai`. Linux `ss` must be available;
run under the service owner account.

Run these public scripts from the release root containing `scripts/` and
`dataset/`. The research checkout uses `scripts_parallel/` and different annotation
paths; those experiment scripts are not part of this release. Public defaults use
`dataset/proactive_perception_annotations.json` and its unchanged `dataset/qa_video/`
paths. Set `VIDEO_ROOT` and `MODEL_PATH` for the selected checkout. Judge credentials
continue to come from `STREAMEVAL_JUDGER_*` environment variables.
