"""Thin training entry point (T1): clone this repo, import the package.

No training logic lives here -- the notebook imports this same module so
reviewed code is trained code. Learned training lands in T5/T6.
"""

import subprocess
import sys

REPO = "https://github.com/abdullah12-bit/capstone-image-restoration.git"


def main() -> None:
    subprocess.run(["git", "clone", REPO], check=True)
    sys.path.insert(0, "capstone-image-restoration/src")
    from restore import restore  # noqa: F401

    print("restore seam importable")


if __name__ == "__main__":
    main()
