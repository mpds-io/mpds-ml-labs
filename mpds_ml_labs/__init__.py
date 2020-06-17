
import sys

MIN_PY_VER = (3, 5)

assert sys.version_info >= MIN_PY_VER, "Python version must be >= {}".format(MIN_PY_VER)