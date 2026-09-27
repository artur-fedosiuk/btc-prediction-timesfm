"""Compatibility entry point for the immutable H24 forward experiment."""

import sys

from .forward.cli import main as forward_main


def main():
  # This legacy flag now only confirms the sole supported live model.
  args = [arg for arg in sys.argv[1:] if arg != "--use-timesfm"]
  return forward_main(["generate", *args])


if __name__ == "__main__":
  raise SystemExit(main())
