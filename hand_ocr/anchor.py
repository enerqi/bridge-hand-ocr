"""Universal ROWS anchor: locate hands by their suit-glyph colour quadruple.

The compass anchor in `rows.py` only works for BridgeWebs (it needs that
source's green/red compass square). RealBridge results, RealBridge replay and
the club-print grids carry no such compass, so `_compass_bbox` raises before
recognition ever runs -- "Mode ROWS" was really "Mode BridgeWebs".

This module supplies the *source-independent* anchor the plan calls for. Every
ROWS hand, in every source, is four stacked suit rows in the fixed order
spade, heart, diamond, club -- and every source colours the two red suits
(heart, diamond) red and the two black suits (spade, club) black. So each hand
prints a vertical **black, red, red, black** quadruple of suit glyphs at a
consistent left edge with a regular row pitch, ranks running off to the right.
That colour quadruple is the anchor: no per-source template, no compass.

Two decks are handled. Most sources are 2-colour (the B,R,R,B quadruple above).
IntoBridge's analysis popup is a 4-colour deck -- ♠blue, ♦orange, ♣green -- so its
♠/♣ are not black and no B,R,R,B quadruple forms. There, all four suit glyphs are
instead *saturated colour* (the rank digits are dark), so a hand is four saturated
glyphs stacked S,H,D,C; that 4-colour path is tried when the 2-colour one is empty.

Detection:

1. Colour-mask the image into red ink (heart/diamond glyphs, plus stray red
   text) and black ink (spade/club glyphs, plus all the rank digits).
2. Find every vertical **red pair** -- a red glyph directly above another, at
   the same left edge, about one row-pitch apart. That is a hand's heart-over-
   diamond middle. Stray red text (a DD-trick-table header, a contract like
   "1H") does not form such an aligned vertical pair, so it drops out.
3. Confirm each pair is a hand by finding at least one aligned black glyph a
   row above (the spade suit symbol) or a row below (the club). This rejects
   any remaining coincidental red pair. The missing side, if any, is synthes-
   ised from the pitch so every hand yields four row centres.

Verified hand counts (clean-resolution renders): `realbridge-4-results` 4/4,
`print-3x4-format` 48/48, and `bridgewebs-4` still 4/4 (the three stray red
pairs from its DD grid are rejected by the black-neighbour test).

Scale robustness: sources render at very different glyph sizes, and some mix
scales within one image (RealBridge replay has big hand glyphs alongside small
bidding-table text). So the size window is generous, and a *dominant-scale
filter* keeps only components near the big, repeated hand glyphs before the
median-relative geometry runs; a *pitch-consistency* pass then drops any
surviving off-scale false stack. This carries `realbridge-replay` (4/4 hands)
with no change to the uniform-scale counts below.

Known limitation -- low resolution: the two densest print grids
(`print-4x5`, `print-5x6`) render the suit glyphs so small that the red mask
shatters each into sub-pixel fragments, so the pair detector under-counts.
A morphological close trades fragmentation for merging adjacent black digits
and is a net loss; the real fix is upscaling the tile before masking, which
belongs with the per-source low-res atlas work (see PLAN step 5), not here.

Vision imported lazily so importing this module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# suit-glyph component size window (pixels). Excludes specks and merged blobs.
# The upper bound is generous enough to admit big-font sources (RealBridge
# replay glyphs run to ~60px); replay's small red UI text is discarded not by
# this window but by the dominant-scale filter below -- see the module note.
_AREA_MIN = 25
_W_MIN, _W_MAX = 5, 70
_H_MIN, _H_MAX = 6, 70

# dominant-scale filter: hand suit glyphs are the big, repeated cluster, so keep
# only components at least this fraction of the 75th-percentile red height. This
# drops contaminating small UI text (replay's bidding table) that would skew the
# median scale; uniform-scale sources (every glyph one size) are unaffected.
_SCALE_KEEP = 0.55

# vertical red-pair geometry, all relative to the median glyph height:
_PITCH_LO, _PITCH_HI = 0.7, 3.4  # heart->diamond row gap, as a multiple of glyph height
_X_TOL = 0.7  # left-edge alignment tolerance, ditto
_Y_TOL = 0.6  # black-neighbour row-centre tolerance, as a multiple of the pitch
# a board's hands share one row pitch; keep only stacks within this band of the
# median stack pitch, dropping an off-scale false stack (e.g. replay's bidding
# table, pitch ~0.5x the hands'). Uniform multi-board pages are unaffected.
_PITCH_CLUSTER_LO, _PITCH_CLUSTER_HI = 0.7, 1.4

# 4-colour-deck anchor (IntoBridge analysis popup): its ♠ is blue and ♣ green, not
# black, so the B,R,R,B quadruple above never forms and the 2-colour anchor finds
# nothing. But all four suit symbols are *saturated colour* (unlike the dark rank
# digits), so a hand is instead four saturated glyphs stacked S,H,D,C at a common
# left edge. This mask is any hue at/above this saturation and value.
_SUIT_SAT_MIN, _SUIT_VAL_MIN = 80, 60


@dataclass
class HandStack:
    """One hand located by its suit-glyph quadruple.

    `left` is the suit-glyph left edge (x); `rows_y` are the four suit-row
    y-centres in S, H, D, C order; `pitch` is the inter-row spacing. These
    define the hand box and its four row bands for the existing row segmenter,
    without any compass."""

    left: int
    rows_y: tuple[float, float, float, float]  # S, H, D, C centres
    pitch: float

    @property
    def top(self) -> float:
        return self.rows_y[0] - self.pitch * 0.6

    @property
    def bottom(self) -> float:
        return self.rows_y[3] + self.pitch * 0.6

    @property
    def centre(self) -> tuple[float, float]:
        return float(self.left), (self.rows_y[0] + self.rows_y[3]) / 2.0


# a component as (x, y, w, h, cx, cy)
_Comp = tuple[int, int, int, int, float, float]


def _colour_masks(img_bgr: Any) -> tuple[Any, Any]:
    """(red_mask, black_mask) for suit-glyph ink. Red is the two hue wraps at
    good saturation; black is dark low-saturation ink."""
    import cv2
    import numpy as np

    def hsv_bounds(*v: int) -> Any:  # uint8 array; opencv stubs reject bare tuples
        return np.array(v, dtype=np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, hsv_bounds(0, 90, 60), hsv_bounds(12, 255, 255)),
        cv2.inRange(hsv, hsv_bounds(168, 90, 60), hsv_bounds(180, 255, 255)),
    )
    black = cv2.inRange(hsv, hsv_bounds(0, 0, 0), hsv_bounds(180, 90, 110))
    return red, black


def _glyph_components(mask: Any) -> list[_Comp]:
    """Connected components in a colour mask, filtered to suit-glyph size."""
    import cv2

    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[_Comp] = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, k]) for k in range(5))
        if area > _AREA_MIN and _W_MIN <= w <= _W_MAX and _H_MIN <= h <= _H_MAX:
            out.append((x, y, w, h, float(cent[i, 0]), float(cent[i, 1])))
    return out


def _suit_colour_mask(img_bgr: Any) -> Any:
    """Saturated coloured ink of any hue -- the four suit symbols of a 4-colour
    deck (♠blue ♥red ♦orange ♣green). Dark, unsaturated rank digits drop out, so
    (unlike the black mask) this isolates the suit glyphs alone."""
    import cv2
    import numpy as np

    def hsv_bounds(*v: int) -> Any:  # uint8 array; opencv stubs reject bare tuples
        return np.array(v, dtype=np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, hsv_bounds(0, _SUIT_SAT_MIN, _SUIT_VAL_MIN), hsv_bounds(180, 255, 255))


def _scan_stacks(pairs: list[_Comp], neighbours: list[_Comp], *, require_both: bool) -> list[HandStack]:
    """Core anchor scan: find every hand as a vertical suit quadruple.

    `pairs` supplies the candidate heart-over-diamond middle (the colour that only
    marks suit glyphs, never rank digits); `neighbours` the spade-above / club-
    below confirmation. `require_both` demands both flanks (the 4-colour deck,
    where all four glyphs share the `pairs` set, so both flanks are real and
    demanding both avoids counting a hand once per adjacent pairing); the 2-colour
    deck needs only one flank (the other may be a void's absent symbol)."""
    import numpy as np

    if not pairs:
        return []
    # dominant-scale filter: discard components far below the big repeated hand
    # glyphs (small UI text), so the median scale below reflects the hands only.
    min_h = _SCALE_KEEP * float(np.percentile([c[3] for c in pairs], 75))
    pairs = [c for c in pairs if c[3] >= min_h]
    neighbours = [c for c in neighbours if c[3] >= min_h]
    if not pairs:
        return []

    med_h = float(np.median([c[3] for c in pairs]))
    pitch_lo, pitch_hi = med_h * _PITCH_LO, med_h * _PITCH_HI
    x_tol = med_h * _X_TOL

    stacks: list[HandStack] = []
    seen: set[tuple[int, int]] = set()
    for heart in pairs:
        for diamond in pairs:
            if heart is diamond:
                continue
            pitch = diamond[5] - heart[5]  # heart above diamond
            if not (pitch_lo <= pitch <= pitch_hi and abs(heart[0] - diamond[0]) <= x_tol):
                continue
            left = heart[0]
            spade_y, club_y = heart[5] - pitch, diamond[5] + pitch
            y_tol = pitch * _Y_TOL
            has_spade = any(abs(c[5] - spade_y) <= y_tol and abs(c[0] - left) <= x_tol for c in neighbours)
            has_club = any(abs(c[5] - club_y) <= y_tol and abs(c[0] - left) <= x_tol for c in neighbours)
            ok = (has_spade and has_club) if require_both else (has_spade or has_club)
            if not ok:
                continue  # stray pair (contract text / DD header / off-middle), not a hand
            key = (round(left / 8), round(heart[5] / 8))
            if key in seen:
                continue
            seen.add(key)
            stacks.append(HandStack(left=left, rows_y=(spade_y, heart[5], diamond[5], club_y), pitch=pitch))
    if not stacks:
        return stacks
    # pitch-consistency: a board's hands share one row pitch; drop any off-scale
    # false stack (e.g. replay's smaller-pitch bidding table).
    med_pitch = float(np.median([s.pitch for s in stacks]))
    stacks = [s for s in stacks if _PITCH_CLUSTER_LO * med_pitch <= s.pitch <= _PITCH_CLUSTER_HI * med_pitch]
    return _dedupe_stacks(stacks, med_pitch)


def _dedupe_stacks(stacks: list[HandStack], med_pitch: float) -> list[HandStack]:
    """Collapse near-duplicate stacks of one hand into one.

    The 4-colour scan finds several valid heart/diamond pairings per hand (all
    four suit glyphs are candidates, and their differing shapes jitter the left
    edge past the coarse key dedupe), yielding 2-3 overlapping stacks per hand.
    Two stacks are the same hand when their left edge and heart row nearly
    coincide; distinct hands sit many pitches apart, so this never merges them."""
    kept: list[HandStack] = []
    for s in sorted(stacks, key=lambda s: (s.centre[1], s.centre[0])):
        dup = any(
            abs(s.left - k.left) <= 1.5 * med_pitch and abs(s.rows_y[1] - k.rows_y[1]) <= 2.0 * med_pitch for k in kept
        )
        if not dup:
            kept.append(s)
    return kept


def find_hand_stacks(img_bgr: Any) -> list[HandStack]:
    """Locate every ROWS hand in the image by its suit-glyph quadruple.

    Source-independent (no compass). Returns one `HandStack` per hand found, in
    no particular order -- callers group them into boards / assign seats by
    geometry. Empty list if no quadruple is present (e.g. a CARDS view).

    Two decks: the common 2-colour case (♥♦ red, ♠♣ black) anchors on the red
    pair confirmed by a black flank. When that finds nothing, a 4-colour deck
    (IntoBridge: ♠blue ♦orange ♣green) is tried, where all four suit glyphs are
    saturated colour -- pairs and flanks come from one saturated-colour mask and a
    full S,H,D,C stack is required."""
    return find_hand_stacks_deck(img_bgr)[0]


# which deck the anchor matched -- lets the recogniser pick the matching atlas
# ("2colour" -> RealBridge/print render; "4colour" -> IntoBridge popup).
Deck = str


def find_hand_stacks_deck(img_bgr: Any) -> tuple[list[HandStack], Deck]:
    """`find_hand_stacks` plus which deck matched ("2colour" | "4colour").

    Same detection, but reports the deck so the caller can select the render's
    atlas. "4colour" only when the 2-colour anchor was empty and the saturated-
    colour path found the hands (IntoBridge's analysis popup)."""
    red_mask, black_mask = _colour_masks(img_bgr)
    stacks = _scan_stacks(_glyph_components(red_mask), _glyph_components(black_mask), require_both=False)
    if stacks:
        return stacks, "2colour"
    coloured = _glyph_components(_suit_colour_mask(img_bgr))
    return _scan_stacks(coloured, coloured, require_both=True), "4colour"
