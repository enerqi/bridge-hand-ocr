# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "docopt-ng",
#     # vision stack: reading a diagram IS the tool. Mirrors pyproject's core deps
#     # (opencv 4.12 pins a numpy with no cp314 wheel; 5.0.x's gapi shim crashes on
#     # import on 3.14) -- keep these in sync with pyproject.toml.
#     "opencv-python-headless>=4.10,<4.12",
#     "numpy>=2.0",
#     "pillow>=10.0",
#     # PaddleOCR fallback is optional (no cp314 wheel yet); add "paddleocr" here to
#     # run the OCR path standalone.
# ]
# ///
"""hand-ocr

Bridge hand-diagram image -> PBN (canonical) or LIN (view) deal text.

Usage:
    hand-ocr <image> [--first=<seat>] [--format=<fmt>]
    hand-ocr --clipboard [--first=<seat>] [--format=<fmt>]
    hand-ocr --demo [--format=<fmt>]

Options:
    <image>            path to a raster hand diagram (screenshot or scan)
    --clipboard        read the image from the OS clipboard instead of a path
    --first=<seat>     PBN 'first' seat: N/E/S/W [default: N]
    --format=<fmt>     output format: pbn | lin [default: pbn]
    --demo             skip vision; emit a hardcoded sample deal (spine check)

Notes:
    PBN can mark unknown hands with '-' (declarer+dummy case). LIN cannot, so
    the lin format requires all four hands known and errors otherwise.
"""

from __future__ import annotations

import sys

from docopt import docopt

from hand_ocr.model import Deal, DealError, Hand


def _demo_deal(first: str) -> Deal:
    # N/S known, E/W unknown -- the declarer+dummy scenario
    north = Hand.from_rows("AKQ4", "KJ3", "AQ5", "T87")
    south = Hand.from_rows("J932", "AQ4", "K87", "AQ2")
    return Deal(hands={"N": north, "E": None, "S": south, "W": None}, first=first)


def _emit(deal: Deal, fmt: str) -> str:
    if fmt == "pbn":
        # emit Board/Dealer/Vulnerable tags too when a Mode-B diagram carried them
        return deal.to_pbn_tags()
    if fmt == "lin":
        return deal.to_lin()
    raise SystemExit(f"unknown --format {fmt!r}")


def main() -> None:
    args = docopt(__doc__)
    first = args["--first"].upper()
    fmt = args["--format"].lower()

    if args["--demo"]:
        deals = [_demo_deal(first)]
    else:
        from hand_ocr.pipeline import image_to_deals  # lazy: needs vision extras

        if args["--clipboard"]:
            from hand_ocr.preprocess import grab_clipboard

            source = grab_clipboard()  # BGR array; raises with a clear message if empty
        else:
            source = args["<image>"]
        deals = image_to_deals(source, first=first)  # one per board found

    exit_code = 0
    blocks = []
    for i, deal in enumerate(deals):
        label = f"deal {i + 1}" if len(deals) > 1 else "deal"
        if deal.note is not None:
            # a contained reader failure: the tile produced no cards (see pipeline)
            print(f"unread {label} (flag for manual fix): {deal.note}", file=sys.stderr)
            exit_code = 2
            continue
        try:
            deal.validate()
            blocks.append(_emit(deal, fmt))
        except DealError as e:
            print(f"invalid {label} (flag for manual fix): {e}", file=sys.stderr)
            exit_code = 2
    if blocks:
        # blank line between PBN tag blocks; LIN lines just stack
        sep = "\n\n" if fmt == "pbn" else "\n"
        print(sep.join(blocks))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
