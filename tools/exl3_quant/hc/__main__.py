"""Prepare HC Q8_0 sidecars from BF16 weights and recorded input statistics.

Run from the repository root. These are offline tools; no command starts a
model, captures GPU inputs, modifies a source pack or deploys a runtime loader.
"""
import argparse
import importlib
import sys


COMMANDS = {'inventory': 'inventory', 'quantize': 'quant', 'refine': 'refine'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS,
                        help='inventory: headers; quantize: GPTQ; refine: code/scale comparison')
    # Handle the command first so its --help belongs to its own parser.
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    module = importlib.import_module('.' + COMMANDS[args.command], __package__)
    module.main(argv[1:])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
