"""Run unittest discovery with output on stdout for clean PowerShell logs."""

from __future__ import annotations

import sys
import unittest


suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
