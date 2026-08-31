"""``python -m bench <command>`` -- one door onto the five entry points.

The commands stay separate modules and separate processes, because their costs differ by
three orders of magnitude: ``run`` is a half-hour grid, ``report`` is two seconds against
a CSV that already exists. Coupling them would mean re-running the grid to fix a table.

What was missing was a way to *find* them. Five ``python -m bench.<module>`` invocations
are only discoverable if you already know the module names, and nothing printed them.
This dispatcher forwards argv unchanged to the module's own ``main``, so every flag and
every default is the module's -- ``python -m bench run --help`` and
``python -m bench.runner --help`` print the same thing, and the longer form keeps working
for any saved command.

The order below is the order they are meant to be run: produce a grid, read it, then draw
from it.
"""

from __future__ import annotations

import argparse
import importlib
import sys

#: command -> (module, one-line description). The module is imported lazily: ``report``
#: needs no matplotlib, and ``run`` should not pay to import the figure stack.
COMMANDS: dict[str, tuple[str, str]] = {
    "run":         ("bench.runner", "run the methods x datasets x modes grid -> grid.csv"),
    "report":      ("bench.report", "render grid.csv as readable tables"),
    "figures":     ("bench.figures", "metric-vs-cardinality figures, one group per dataset"),
    "reconstruct": ("bench.reconstruction", "best/worst reconstructed snapshot per method"),
    "decrement":   ("bench.decrement", "what the next generator actually buys"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    width = max(len(c) for c in COMMANDS)
    epilog = "\n".join(f"  {c.ljust(width)}  {d}" for c, (_m, d) in COMMANDS.items())
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="rb_vi_bench: compare non-negative dual-cone reduction methods",
        epilog="commands:\n" + epilog + "\n\nEvery command takes --help of its own.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", nargs="?", choices=sorted(COMMANDS), metavar="COMMAND")
    # Everything after the command belongs to the command, not to this parser, so it is
    # never interpreted here -- otherwise `-h` after a command would print this help
    # instead of the command's.
    args, rest = p.parse_args(argv[:1]), argv[1:]
    if args.command is None:
        p.print_help()
        return 2
    module = importlib.import_module(COMMANDS[args.command][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
