# Take an empty cup out of a person's trembling hand

> Take the cup the person is holding out to you and put it on the coaster on the tray.

## An everyday skill

People with a hand tremor (essential tremor, Parkinson's disease) hand their cup back after a drink many times a
day, at home and in care homes, and whoever takes it takes it from a shaking hand that keeps hold of it. The cup sways
about a centimetre each way four or five times a second, the person's fingers are wrapped round its lower half and
their little finger carries its weight; they let go only once they feel that weight taken. You take it by the part
above their fingers, lift it a little until they let go, then lift it clear. Close too
low and you catch a finger; push or pull their hand and you hurt them. The tremor is not an event: the hand shakes
from the first frame to the last.

## Scene (station frame: x forward from the arms, y to the robot's left, cm; heights above the table top)

| Object | Details |
| --- | --- |
| Armrest | firm foam block, vinyl covered, 30 x 16 x 10.8 cm, taped to the table, x 45.5 to 75.5, centred y 4.9 |
| Trainer forearm and hand | life-size right forearm and hand printed from `print/*.stl` (PLA, the fingers' inner faces silicone coated). The forearm (1.2 kg with a steel-shot bag inside) lies free on the armrest, pointing at the robot. At the wrist, a pan-tilt bracket with two MG90S micro servos: pan (vertical axis) and tilt (roll, along the forearm). Then the palm and thumb (one part), then the four fingers as one curled block on a third MG90S at the knuckles. The hand is thumb up with the palm against the cup's left (+y) side; the fingers wrap the cup's front (the robot's side) and right up to 7.1 cm, the thumb its back, the little finger curls under its base. A microswitch in the little finger's pad is pressed by the cup's weight. |
| Tremor | `tremor/tremor.ino` on an Arduino Uno/Nano: pan +-5.5 deg and roll +-3.8 deg, sine, a quarter cycle apart, at 4.5 Hz, for the whole trial. The person lets go (the finger servo opens the fingers 38 deg) once the cup's base has been off the microswitch for 0.25 s. |
| Cup | clear polypropylene tumbler, empty, 7 cm across, 14 cm tall, 70 g; base 10 cm up |
| Tray | hospital meal tray 35 x 26 x 1.5 cm, centred (26, -22); a 9 cm cork coaster, 4 mm, at its centre |

One fixed layout, as in the screenshot: the cup's centre at (37.0, 0.0), the wrist at (43.8, 4.9), 13.4 cm up; tremor
at 4.5 Hz.

## Scoring (task.py)

- **0 (latched for the trial):**
  - the cup is knocked or dropped: it tips past 45 deg, or drops to the table top anywhere but the tray;
  - the hand is pushed or pulled: the pan or roll pointer goes past +-25 deg on the wrist's scale (the tremor itself
    stays under 15 deg), or the forearm moves over 1.5 cm, lifts over 1 cm or turns over 8 deg from its marks;
  - a finger or the thumb is caught: it lies between a gripper's two finger plates as they close (the plates are
    68 mm wide; the check covers them out to 13 mm past the tool point).
- Otherwise, at the end of the trial:
  - **1.0** once the cup has stood upright (within 10 deg) on the tray, its base centre within 1 cm of the
    coaster's centre (the whole 7 cm base on the 9 cm coaster), with no gripper within 7 cm of its axis, for 1 s
    (`FULL_HOLD_S`; until then it reads 0.5);
  - **0.5** if it stands upright on the tray otherwise;
  - **0.25** if the cup is out of the hand (5 cm clear of every finger) and held by a gripper;
  - **0** otherwise. Doing nothing scores 0.
- Real trial: a side camera sees the wrist pointers, the fingers and the gripper's plates; the forearm's and the
  cup's positions are checked against tape marks with a ruler.

## Running it

    source <IsaacLab-Arena>/yam.env
    python make_assets.py                                   # writes assets/*.usda (pxr only)
    python export_stl.py                                    # writes print/*.stl (the trainer hand's parts, mm)
    python demo.py task.py out                   # scripted full-score demo
    python demo.py task.py out --variant low     # control: plates round the cup's middle
    python demo.py task.py out --variant still   # control: aimed at the cup's position at one instant
    python demo.py task.py out --variant idle    # control: does nothing

A policy run (Dream-1) uses the same task file through Inspect Robots:

    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 --policy dream \
        --max-steps 1200 \
        --instruction "$(python -c 'import task; print(task.INSTRUCTION)')" --scorer task_score

Run limit: 1,200 steps (60 s at 20 Hz) per trial; the demo takes about 320. Each trial takes about 3 minutes of
wall time on one GPU. No randomness.

## Results (simulation)

| Run | Score |
| --- | --- |
| demo | 1.0 (15.8 s; the person lets go at 5.8 s; cup 3 mm from the coaster's centre) |
| low: plates round the cup's middle | 0 (finger caught) |
| still: aimed at the cup's position at one instant | 1.0 |
| idle | 0 |
| Dream-1 (dream-1-beta, one trial, 1,200 steps) | 0 (reaches toward the cup twice with the left gripper, never takes it) |

The demo: the right gripper points forward and 10 deg down, its plates open (94 mm) either side of the cup (70 mm),
centred on where the cup sways about, with the tool point 12.8 cm above the cup's base so the plates' lower edges
(68 mm wide) are about 1.5 cm above the palm and fingers. It closes in one go to a firm grip, lifts 3 cm (the person
lets go), lifts clear, carries the cup to the coaster, looks where the cup (not the gripper) is from just above it
and corrects twice, sets it down and lets go. The `still` control aims at where the cup is seen at one instant
instead of where it sways about, and also scores 1.0: with this cup the plates' 12 mm of side clearance takes the
sway. What the trial tests is getting the plates into the narrow band above the person's fingers on a cup that sways
and bobs, closing without catching a finger, and transferring the cup's weight gently enough that the person lets go,
all without pushing or dragging their hand.

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. The servos are joint drives with an MG90S's torque (0.25 N m); their
sine targets are written every physics step (not only every control step), so the tremor is the rig's own, and a
gripper holding the cup steadies the hand. The trainer hand is rigid, its fingers and thumb are capsules along the
printed parts' centre lines; the cup is held by the fingers' light grip and the curled little finger, and the
release servo opens the fingers the way the microswitch and sketch do. The forearm is free on the foam. The
gripper's finger plates are YAM's own meshes. Measured in simulation, the rim sways 2 to 3.5 cm peak to peak side to
side (the cup also wobbles a little in the loose grip).

## Real rig: parts

| Part | Notes |
| --- | --- |
| `print/forearm.stl`, `print/palm_thumb.stl`, `print/fingers.stl` | PLA, 3 walls, 10 % infill; fingers' inner faces dipped in tool-handle silicone |
| pan-tilt bracket for 9 g servos | the common two-axis bracket sold for SG90/MG90S servos |
| 3 x MG90S micro servo | pan, roll, release |
| Arduino Uno or Nano, 5-6 V 2 A supply for the servos | `tremor/tremor.ino` |
| subminiature microswitch (lever, about 0.3 N) | in the little finger's pad, under the cup's base |
| momentary button | reset between trials |
| 0.6 kg steel-shot bag | inside the forearm shell, for 1.2 kg |
| foam block 30 x 16 x 10.8 cm, vinyl cover | armrest |
| polypropylene tumbler 7 x 14 cm, 70 g | e.g. a standard 400 ml clear PP tumbler |
| hospital meal tray, cork coaster | |

Cost: about 50 USD plus printing.

## Files

`task.py` (scene, servos, monitor, scoring), `demo.py` (scripted demo and controls), `make_assets.py` (USD
generator: hand articulation and cup), `assets/` (generated USD), `export_stl.py` and `print/` (printable trainer
hand), `tremor/tremor.ino` (servo sketch), `dynkit.py` and `demokit.py` (shared helpers: articulated USD bodies, the
episode clock, robot capsules, Cartesian IK demo runner). All assets are made by this task's code; the gripper's
plates are the YAM's own; no third-party assets.
