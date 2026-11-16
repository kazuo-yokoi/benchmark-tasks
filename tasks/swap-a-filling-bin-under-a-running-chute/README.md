# Swap a filling bin under a running chute between two drops

> The machine drops a finished part down its chute into the bin under it every few seconds, and it doesn't stop. A bin
> takes a lot of exactly 6 parts: once the bin under the chute holds 6, put it on the grey FULL mark and put the empty
> bin under the chute. Every part must land in a bin.

A machine at the far end of the table (an automatic cut-off saw, press or moulding machine stands in for any machine
that discharges finished parts one at a time) sends each finished part, a steel bushing, down a yellow chute that ends
in a short drop tube 19 cm above the table. The part falls straight down into the blue bin under the tube. A second,
empty bin stands to the left, and a FULL mark is taped to the right. Each bin has a handle tab on both of its ends.

Changing bins (totes) at a machine's discharge while the machine keeps running is routine work for machine operators
and hand packers: a bin changed late overflows or mixes two lots, one changed early leaves a short lot, and a part that
falls while neither bin is under the chute lands on the floor.

A bin takes a lot of exactly six parts, and the machine does not stop for the change. The full bin can leave the tube
only after the sixth part has landed, and the empty bin has to be under the tube before the seventh falls: a gap of 1.8
to 2.4 s depending on the lot. Knowing when to change is the easy half; anyone watching the video could count to six.
The hard half is physical: two 13 cm bins have to trade places under one point in under two seconds, without tipping
parts out of the full one and without a gripper in the path of the next part. Carrying the full bin to the mark and
then fetching the empty one is too slow (part 7 falls on the table, below); the change has to be one movement, for
example both arms at once, one sliding the full bin out while the other slides the empty one in, or the empty bin
pushed in against the full one. The scoring does not prescribe how: only where each part lands and where the bins end
up count.

**Lots.** Three lots differ only in when the first part falls and how often parts fall. All three start from the same
picture, so the robot cannot tell which lot it is in until it sees the parts fall; a swap timed on a clock fits only
one lot. A run is three trials, one per lot (`--epochs 3`: trial k uses lot k mod 3, lot A first), and its score is
their mean.

| Lot | First part leaves the tube | Then one every | Sixth part | Seventh part |
|---|---|---|---|---|
| A | 3.5 s | 2.0 s | 13.5 s | 15.5 s |
| B | 4.6 s | 2.4 s | 16.6 s | 19.0 s |
| C | 3.0 s | 1.8 s | 12.0 s | 13.8 s |

Ten parts fall in each lot; the last one falls by 26.2 s, well inside the 60 s run limit.

## What a run measures

To score 1.0 the robot has to count the parts as they fall, start the change only once the sixth has landed, and
finish it before the seventh, keeping a bin under the tube the whole time. Each failure mode is measured with
`demo.py --variant`:

| Behaviour | Lot A | Lot B | Lot C |
|---|---|---|---|
| Counts the parts, then swaps both bins at once (the demo, from the top camera) | 1.0 | 1.0 | 1.0 |
| Counts, but takes the full bin to the mark before fetching the empty one (`slow`) | 0 (part 7 falls on the table) | | |
| Swaps at the moment lot A's sixth part has landed, without looking (`timer`) | 1.0 (its own lot) | 0 (after 4 parts; part 5 falls between the moving bins) | 0 (after 7 parts; a part comes out of the moving full bin; at best 0.5 for a lot of 7) |
| Does nothing (`idle`) | 0 (all ten parts in the first bin) | | |
| Dream-1 (one run of three trials, 1,200 steps each) | 0 (moved its arms by the bins, never changed them) | 0 (carried the full bin to the mark after all ten had fallen) | 0 (lifted the empty bin next to the full one, never changed them) |

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the machine, chute and drop tube, the bins, the FULL mark, the parts and the feed per lot, `success()` and `score()` |
| `assets/bin.usda`, `make_bins.py` | The bin (a floor, four walls and two handle tabs as one rigid body) and the script that writes it |
| `perceive.py` | The demo's eyes: counts the parts falling out of the tube in the top camera's image |
| `demo.py` | The scripted full-score demo and the `slow`, `timer` and `idle` variants |
| `test_task.py` | Offline checks (no Isaac Sim) of the feed's timing and the scoring rules (`python -m pytest test_task.py`) |
| `dynkit.py` | Kinematic bodies, an episode clock, the robot's links and the lots (extends the `scenekit.py` of an accepted task) |
| `demokit.py` | A closed-loop scripted demo runner (Cartesian moves with the URDF's IK; writes the videos and `setup.png`) |

`dynkit.py` hooks `yam_arena.scene` and the embodiment's step, reset and trial start without modifying yam_arena, so
the task runs with the standard `yam_arena` embodiment and `task_score` scorer. Tested with yam-isaac-lab-arena at
`0059d02` (Isaac Lab-Arena 0.3.1, inspect-robots 1.0.0), the commit the Brev launchable installs.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env`), three trials,
one per lot:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The machine drops a finished part down its chute into the bin under it every few seconds, and it doesn't stop. A bin takes a lot of exactly 6 parts: once the bin under the chute holds 6, put it on the grey FULL mark and put the empty bin under the chute. Every part must land in a bin." \
  --scorer task_score --max-steps 1200 --epochs 3 --store-frames --log-dir ~/yam-eval/chute-dream
for log in ~/yam-eval/chute-dream/*.json; do inspect-robots video "$log"; done
```

`task_score` prints each trial's score: 1.0 on success, otherwise `score()` at the end of the trial. `TASK_LOT=B`
pins every trial to one lot.

## Scripted demo

```bash
TASK_LOT=A python demo.py task.py out_a                    # full score (also TASK_LOT=B, TASK_LOT=C)
TASK_LOT=A python demo.py task.py out --variant slow       # one bin after the other
TASK_LOT=B python demo.py task.py out --variant timer      # lot A's clock, without looking
TASK_LOT=A python demo.py task.py out --variant idle       # does nothing
python -m pytest test_task.py                              # offline checks of the timing and the scoring
```

The demo knows the starting layout (where the bins and the FULL mark are, as the setup gives them) and, while it
runs, only the top camera's image (the 320 x 176 JPEG frame Dream-1 is sent) and its own joint angles. It counts the
parts falling out of the tube from the image: `perceive.py` watches the column of air between the tube's mouth and the
bin's rim, found from the camera's calibration, and counts a part each time that small window suddenly changes and
settles back; it does not count while an arm that has just moved is in or next to that window. It never reads the
simulator's object poses or the task's counters, and the bins are wherever its own grippers put them.

1. The right arm takes the first bin (under the tube) by the tab on its right end, the left arm the empty bin by the
   tab on its left end; the left arm lifts the empty bin 1 cm and holds it beside the first, 4 cm away.
2. From the moment the right arm holds the first bin (2.3 s, before the first part in any lot) it counts the falling
   parts. 0.15 s after it has seen the sixth fall (it has landed), the right arm slides the full bin out to the right
   while the left arm, 0.2 s later, slides the empty bin in under the tube.
3. The left arm sets the empty bin down and lets go at once, lifting its fingers off the tab and backing out low to
   its own side, clear of the next part; the right arm sets the full bin on the FULL mark and lets go. Both go back to
   the ready pose. (Letting go is not scored; the trial succeeds as soon as both bins stand in place.)

## Scoring

All distances are measured on the table; in a real trial they are judged from an overhead video (which bin each part
falls into) and a tape measure at the end.

- **The feed:** part k leaves the tube's mouth, 19 cm above the table over the spot the first bin starts on, at the
  lot's time (table above), falling at 0.5 m/s.
- **Caught:** a part is caught by a bin once its centre has been inside that bin (within its 12 x 12 cm inner outline,
  between its floor and its rim) for 0.3 s in a row. It counts for that bin from then on.
- **Violation (score 0, latched):** a part not caught by a bin within 1.5 s of leaving the tube (on the table, on a
  rim, in a gripper, off the table), or a caught part out of the bin that caught it (more than 1 cm outside its inner
  outline or 3 cm above its rim) for 0.25 s or more, for example tipped out while the bin is carried; or a bin fallen
  off the table (its underside 5 cm or more below the table top).
- **The split:** the parts must go as the job needs: parts 1 to n into the first bin, and every later part that has
  left the tube into the second, at least one. Any other split (a part missed, the bins swapped back and forth)
  scores 0.
- **Placed:** the first bin is on the FULL mark when its centre is inside the mark's 17 x 17 cm inner square; the
  second bin is under the tube when its centre is within 3 cm of the spot the first bin started on. Either one must
  stand on the table (its underside within 1 cm of it, tilted under 10 degrees); whether a gripper still holds it does
  not matter.
- **Score:** with the split as above, 0.5 for the first bin placed on the FULL mark plus 0.5 for the second placed
  under the tube when n is 6 (an exact lot); 0.25 each when n is 5 or 7 (one off); 0 for any other n.
- **Success (1.0):** the first bin on the FULL mark with parts 1-6, the second under the tube with the seventh part
  (and every later one) in it, both standing on the table, no violation, held for 1 s; the trial then ends.
- **End of a trial:** a trial ends as soon as success has held for 1 s (that is 1.4 s after the seventh part leaves
  the tube at the earliest, before parts 8 to 10 fall), or else after 1,200 steps (60 s), long after the tenth part
  (26.2 s at the latest); it then scores `score()` as it stands at the end. A violation does not end the trial, but its
  0 stands.
- Doing nothing scores 0: every part goes into the first bin, which never reaches the mark. Waiting until the line has
  run out scores 0 too: the first bin then holds ten.

## Real-world setup and reset

A grey cabinet stands for the machine. Fix a 4.4 cm tube vertically over the first bin's spot with its mouth 19 cm
above the table, and run a closed chute (a pipe will do) from the back of the cabinet down to the tube's top. A helper
behind the cabinet, out of the arms' reach, drops one bushing (steel, 25 mm across, 20 mm long, about 55 g) into the
chute on the lot's beat (a metronome or a phone timer: lot A from 3.5 s every 2.0 s, lot B from 4.6 s every 2.4 s,
lot C from 3.0 s every 1.8 s), ten in all; the times are when a part leaves the tube's mouth, so drop each one earlier
by the chute's travel time, measured once from a video. The two bins are
plastic boxes 12 x 12 cm inside and 7 cm deep with a tab glued to each end; tape the FULL mark. Reset: empty both
bins, put the first under the tube and the second on its start mark, and collect the ten bushings. Safety: a 55 g
bushing falling 19 cm and two 150 g plastic bins are harmless, and the helper never reaches into the robot's
workspace.

## Runtime and seeds

A full 1,200-step trial with Dream-1 takes about 3 minutes on an RTX GPU, plus about 1 minute for Isaac Sim to start;
three trials (one per lot) take about 10 minutes when the Dream-1 service answers at once (our run took 27 minutes,
most of it waiting for the service to warm up). The layout and each lot's feed are fixed (`-E object_jitter=0`), so
there are no seeds to choose. The demo reaches success after 337 steps (16.9 s) in lot A, 407 (20.4 s) in lot B and
305 (15.3 s) in lot C.
