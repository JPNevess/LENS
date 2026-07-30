"""Locate a JVM before CapyMOA is imported.

CapyMOA starts a JVM at import time and needs ``JAVA_HOME`` to point at a JDK.
If it is already set we trust it; otherwise we look in the conventional install
locations for the platform and validate that the candidate really contains a JVM
library, because on macOS ``/usr/bin/java`` is a stub that exists even when no
JDK is installed.
"""
import glob
import os
import subprocess

# Conventional install roots, most specific first. These are platform
# conventions, not machine-specific paths.
_CANDIDATE_GLOBS = (
    "/Library/Java/JavaVirtualMachines/*/Contents/Home",
    "/opt/homebrew/opt/openjdk*/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk*",
    "/usr/local/opt/openjdk*/libexec/openjdk.jdk/Contents/Home",
    "/usr/local/opt/openjdk*",
    "/usr/lib/jvm/*",
)

_JVM_LIBS = ("lib/libjli.dylib", "lib/server/libjvm.dylib",
             "lib/server/libjvm.so", "bin/server/jvm.dll")


def _is_jdk(home):
    if not home or not os.path.isdir(home):
        return False
    return any(os.path.exists(os.path.join(home, lib)) for lib in _JVM_LIBS)


def _from_java_home_tool():
    try:
        out = subprocess.run(["/usr/libexec/java_home"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_java_home():
    """Set ``JAVA_HOME`` if it is missing or invalid. Raises if no JDK is found."""
    current = os.environ.get("JAVA_HOME")
    if _is_jdk(current):
        return current

    candidates = []
    tool = _from_java_home_tool()
    if tool:
        candidates.append(tool)
    for pattern in _CANDIDATE_GLOBS:
        candidates.extend(sorted(glob.glob(pattern), reverse=True))

    for home in candidates:
        if _is_jdk(home):
            os.environ["JAVA_HOME"] = home
            return home

    raise RuntimeError(
        "No JDK found. Install a JDK (>= 11) and set JAVA_HOME to it, for "
        "example:\n"
        "  macOS  : brew install openjdk && "
        "export JAVA_HOME=$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home\n"
        "  Linux  : apt install default-jdk && "
        "export JAVA_HOME=/usr/lib/jvm/default-java")
