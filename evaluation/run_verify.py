"""Verify matured immutable forecasts independently from generation."""

import sys

from .forward.cli import main as forward_main


def main():
  return forward_main(["verify", *sys.argv[1:]])


if __name__ == "__main__":
  raise SystemExit(main())
