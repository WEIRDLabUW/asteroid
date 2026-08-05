"""Minimal YAML config loading for the training scripts.

A config file simply provides default values for argparse arguments. Any flag
passed on the command line still overrides the file, and keys the script does
not define are ignored, so one per-environment config can be shared by every
method.
"""

import yaml


def parse_with_config(parser):
    """Add a ``--config`` option and parse args, applying the YAML as defaults."""
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file whose keys override argument defaults")
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    return parser.parse_args()
