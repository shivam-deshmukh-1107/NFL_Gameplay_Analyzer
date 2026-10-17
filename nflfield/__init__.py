"""NFL Blitz field tracker - merged pipeline.

Every frame leaves this library in one of four tiers, and a frame that cannot
be solved is refused by name rather than filled in:

    CERTIFIED  camera solved from this frame's own paint, all 8 gates passed
    CARRIED    ECC turf alignment from a certified keyframe, re-checked against
               the visible paint, bounded at 30 frames
    TRACKED    pose moved through measured optical flow; NO camera claim
    REFUSED    the evidence is not there; the reason names the gate

Never sum them. Adding the tiers reports the weakest as if it were the
strongest, which is the whole point of having four.

STAGES, in the order they run
-----------------------------
    0   ingest                 active picture, invertible transform, (fx, fy)
    1   fieldcolor, gating     turf model, ridge response, play/menu gate
    2   hudmask                static HUD overlay map
    3   markings, geometry     paint -> segments -> flank test -> pencil
    4   lattice, ladder,       two independent identity proposers, cross-checked
        propose
    4R  drawnlines, hudtext,   absolute scale against the game's own
        scalecheck             down-and-distance - the promotion gate
    5   camera, walk           the 6-param solve and all eight refusal gates
    6   motion, observe, walk  LK flow, cached observations, bidirectional walk
    7   carry, flowcheck,      the CARRIED tier and two independent referees
        indep_paint
    10  render                 the four-tier overlay

Stage 8 (absolute yard naming) and 9 (the metric template) do not exist yet;
stage 8 is blocked on footage, not on code.

WHERE TO READ BEFORE CHANGING ANYTHING
--------------------------------------
    ../README.md          what runs, what is verified, what is open
    ../docs/LESSONS.md    every measurement made here, and every claim retracted
    ../../MERGE_PLAN.md   the architecture and the provenance of every stage

Two rules this library is built around, both inherited from both parents:
a false accept is far worse than a false refusal, and **fit quality proves
nothing** - only a measurement against something independent means anything.
Measured constants carry their measurement beside them in the code.
"""
__version__ = "0.1.0.dev0"
