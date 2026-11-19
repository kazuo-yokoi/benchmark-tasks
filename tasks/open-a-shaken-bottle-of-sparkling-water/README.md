# Open a just-shaken bottle of sparkling water without it foaming over

> Open the bottle of sparkling water and put the cap in the dish.

## An everyday skill

Everyone has had a fizzy drink that was knocked about foam over their hands. People open one in stages: turn the cap
just until it hisses; if the foam climbs into the neck, turn it back until the foam drops; crack it again; and only
take the cap off once the hiss has stopped. Twist it straight off and the foam pours over the rim. One hand holds
the bottle upright, the other works the cap back and forth in small turns while watching the foam through the clear
bottle. The shaking is the bottle's state at the start; nothing happens during the trial that the robot doesn't
cause.

## Scene (station frame: x forward from the arms, y to the robot's left, cm; heights above the table top)

| Object | Details |
| --- | --- |
| Bottle | 500 mL PET bottle of Perrier sparkling natural mineral water (the 50 cl / 16.9 fl oz PET bottle; one case per session): 6.5 cm across, 20.5 cm tall, 28 mm screw cap. Chilled to 6 +-1 C. Before chilling, the cap is turned about 30 deg, just enough to break the tamper ring's bridges and short of the seal (which opens near 45 deg, so no gas escapes), then turned back to its stop. Shaken just before the trial (ten full inversions in 5 s); a layer of foam shows on the water at the start. Base centred at (28, -12). |
| Drip mat | grey absorbent mat, 30 x 20 cm, 3 mm, centred under the bottle |
| Dish | white ceramic dish, 10 cm across, 1.2 cm tall, at (17, -27) |
| Splash screen | clear acrylic sheet 30 x 30 cm on a stand, 15 cm behind the bottle (real trials; operators wear safety glasses) |

Both arms start at the ready pose. One fixed layout, as in the screenshot.

## Scoring (task.py)

- **0 (latched for the trial):** foam or water comes out over the rim (judged on the side video of the neck: liquid or
  foam crossing the rim, not condensation), or the bottle tips past 30 deg.
- Otherwise, at the end of the trial:
  - **1.0** once the cap is off, in the dish and let go (no gripper within 5 cm of it), and the bottle stands upright (within 10 deg) on the mat with no
    gripper within 7 cm of its axis, for 1 s (`FULL_HOLD_S`; until then it reads 0.5);
  - **0.5** if the cap is off but not in the dish, or the bottle is still held;
  - **0.25** if the cap is still on but the bottle has been vented (with the cap cracked past 45 deg, no foam rises
    above the water for 3 s);
  - **0** otherwise. Doing nothing scores 0.
- Real trial: a side camera on the neck against a dark card, at 60 fps. Grippers and wrists wear waterproof sleeves.

## Running it

    source <IsaacLab-Arena>/yam.env
    python make_assets.py                        # writes assets/bottle.usda (pxr only)
    python demo.py task.py out                   # scripted full-score demo
    python demo.py task.py out --variant twist   # control: unscrews it at an ordinary pace
    python demo.py task.py out --variant crack   # control: cracks it to the demo's 70 deg and holds, not watching
    python demo.py task.py out --variant idle    # control: does nothing

A policy run (Dream-1) uses the same task file through Inspect Robots:

    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 --policy dream \
        --max-steps 2400 --instruction "$(python -c 'import task; print(task.INSTRUCTION)')" --scorer task_score

Run limit: 2,400 steps (2 minutes at 20 Hz); the demo takes 1,521. A trial takes about 6 minutes of wall time on one
GPU. No randomness.

## Results (simulation)

| Run | Score |
| --- | --- |
| demo | 1.0 (vented at 47.8 s after three re-closings, cap off at 69 s, done at 76 s; foam peaked at 17.0 cm, the rim is at 20.5 cm) |
| twist: unscrewed at 90 deg/s, as for an unshaken bottle | 0 (foam over the rim at 9.0 s) |
| crack: cracked to 70 deg and held there | 0 (foam over the rim at 9.6 s) |
| idle | 0 |
| Dream-1 (dream-1-beta, one trial, 2,400 steps) | 0 (moves both grippers near the bottle and the dish, never takes hold of the cap; the bottle stays shut and upright) |

The controls (twisted off, cracked and held: 0; idle: 0) are in `demo.py`.

The demo: the left gripper holds the bottle round its body; the right comes straight down onto the cap and turns it
with its wrist in strokes of up to 100 deg, letting go and turning back between strokes. It cracks the cap to 70 deg;
whenever the foam reaches the shoulder (16.5 cm) it turns the cap back to 20 deg, where the seal closes and the foam
collapses, and opens it again once the foam is below 15.7 cm. When the hiss stops it unscrews the cap the rest of the
way, lifts it off and puts it in the dish. It reads the foam's height from the task state; a camera sees the same white
column through the bottle. Re-closing is not the only way: in the model, cracking the cap only just past the seal
(about 55 deg) and holding it there vents the bottle without foaming over, in about 70 s; a wider crack held without
watching (the `crack` control) foams over.

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. The cap turns on a damped joint (about 0.08 N m at 90 deg/s); its seal
opens past 45 deg and the thread lets go at 1.55 turns, after which the cap is free. The gas and foam are a model
(`fizz.py`): the headspace pressure (2.2 bar at the start) leaks through the opened seal at a rate set by the cap's
angle; gas leaves the shaken water the faster, the lower the pressure; the foam it makes collapses within seconds and
is drawn as a white column inside the bottle. The constants are set so that, as with a real shaken bottle, twisting
the cap off foams over within seconds, cracking it slightly and holding vents it in about a minute, and closing the
cap again stops the foam within a few seconds; the session check in the reset notes confirms the real case behaves
the same way. The clear PET is drawn only as its edges so the water and foam stay visible. Water is rigid mass (no
sloshing).

## Files

`task.py` (scene, joints, monitor, scoring), `fizz.py` (the gas and foam model), `demo.py` (scripted demo and
controls), `make_assets.py` (USD generator), `assets/` (generated USD), `dynkit.py` and `demokit.py` (shared helpers:
articulated USD bodies, kinematic visual bodies, the episode clock, Cartesian IK demo runner). All assets are made by
this task's code; no third-party assets.
