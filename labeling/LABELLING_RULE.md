# How to judge a window: on task or off task

One page. Read it before every sitting, not just the first.

---

## Why this exists

127 already-judged windows were re-judged blind. The second judgement matched
the first **64.6% of the time** — Cohen's κ = 0.261, which is "fair" agreement
at best. The flips were symmetric (7 on→off, 6 off→on in the first batch), so
this is not a standard that drifted. It is a question that was never pinned
down, answered twice.

That number is the ceiling on everything else. A model is scored against these
labels, so it cannot agree with them more often than they agree with
themselves. **Improving the rule raises the ceiling; nothing else does.**

## Where the disagreement actually was

Comparing the 82 windows judged the same way twice against the 45 that flipped:

| | agreed twice | flipped |
|---|---|---|
| Angle away from the teacher | 44.9° | **35.5°** |
| Fraction of frames on-task | 0.378 | **0.447** |
| Fraction of frames off-task | 0.586 | **0.515** |

The pattern is clear: **the flips are the mixed windows.** A student who is
clearly on a phone for fifteen seconds gets the same answer twice. A student
who glances away, comes back, writes a little, looks down — that is where the
answer changed.

So the rule below spends its length on exactly that case.

---

## The question you are answering

> Over these 15 seconds, was this student **engaged with the lesson**?

Not "were they well-behaved". Not "were they looking at the teacher". A
student reading the textbook with their head down is on task. A student
staring attentively at a friend is not.

## The decision, in order

**1. Is more than half the window clearly one or the other?**
Judge by the majority of the fifteen seconds. Three seconds of looking away
inside twelve seconds of writing is **on task**.

**2. Mixed, with no majority? Use the dominant activity, not the posture.**
Ask what the student was *doing* for most of the window, and judge that. A
student who writes for seven seconds and gazes around for eight is on task if
the writing was the activity and the gazing was between bouts of it.

**3. Still undecided? Apply these tie-breaks, in order.**

| Signal | Answer |
|---|---|
| A phone in their hand at any point | **off** |
| Eyes closed for more than ~5 of the 15 seconds | **off** |
| Head down over a book, paper or desk work | **on** |
| Talking to a neighbour while the teacher is talking | **off** |
| Talking to a neighbour during group work | **on** |
| Looking at the board, teacher, or their own work | **on** |
| Looking around the room with no apparent task | **off** |

**4. If two tie-breaks conflict, off task wins.** A phone in hand while
writing is off task. This is a deliberate asymmetry: the system exists to
surface students who may need attention, and a false "needs attention" costs a
glance from the teacher while a false "fine" costs nothing being noticed.

**5. Only now, use `unclear`.** Press `u` when you genuinely cannot see enough
to apply the above — the student is occluded, too small, turned fully away, or
in frame for only a moment. About 8% of windows should land here. If you are
using it more than 1 in 5 times, the crops may be too small; say so rather
than guessing.

`unclear` is excluded from training, not counted as a wrong answer. Using it
honestly makes the dataset better. Using it to avoid a hard call makes it
smaller for no gain.

---

## Things that are **not** evidence

These appeared in the flips and should not move your judgement:

- **A neutral face.** It is the most common expression in the data (4,035
  frames a session). It means nothing on its own.
- **Slouching.** Posture is comfort, not attention.
- **What the rest of the class is doing.** The model uses that; you should
  not. If you judge a student partly by their neighbours, you are labelling
  the feature the model is supposed to learn, and its measured accuracy
  becomes circular.
- **What you labelled the previous window.** Each window is judged on its own.
  A student who was off task ten seconds ago may be on task now; that change
  is the signal.

## Things you must not look at

`manifest.json` carries the rule pipeline's own verdict for every window. It
is there so your labels can be compared against the rules afterwards — not so
you can check them before deciding. Seeing it first anchors your judgement to
the thing being tested and makes the comparison meaningless.

---

## Practical notes

- **Label in sittings of 30–45 minutes.** Fatigue shows up as noise, and noise
  is indistinguishable from a wrong answer when the model is fitted.
- **Re-check yourself periodically.** `python recheck.py --n 40` replays
  windows you have judged, without showing your earlier answer, and reports
  agreement and κ. Run it after every few hundred labels.
- **Target:** κ above 0.6 is "substantial" agreement and roughly doubles the
  usable ceiling. κ above 0.4 would already be a material improvement on 0.261.
- **If a second person labels**, have them do 40 windows you have also done.
  Between-rater agreement is the figure a reviewer will ask for, and it cannot
  be computed without deliberate overlap.

## What good looks like

At κ = 0.261 the per-label error rate is about 23%, capping any model near
0.77. At κ = 0.6 that cap rises substantially, and every accuracy figure in
the project becomes worth more — without changing a line of model code.
