"""
CFFI build script for minc2_simple.

Locates libminc in one of three ways (checked in order):
  1. LIBMINC_DIR  env var  -- path to a libminc install or build tree
  2. MINC_TOOLKIT env var  -- path to a full MINC Toolkit install
  3. Auto-build            -- downloads libminc source from GitHub and
                             builds a minimal static libminc2.a

Auto-build requires: cmake, a C compiler, HDF5-dev, and zlib-dev.
"""

import os
import sys
import shutil
import subprocess
import multiprocessing
import cffi
from sys import platform

ffibuilder = cffi.FFI()

source_path = "src"

# ── libminc commit to download when auto-building ──────────────────────
_LIBMINC_REPO = "https://github.com/BIC-MNI/libminc"
_LIBMINC_BRANCH = "modernize_cmake"
_LIBMINC_COMMIT = "d2a0f474"

# ── helpers ─────────────────────────────────────────────────────────────

def _find_hdf5_includes():
  """Return a list of include directories for HDF5 headers."""
  dirs = []
  try:
    cflags = subprocess.check_output(
        ["pkg-config", "--cflags", "hdf5"],
        stderr=subprocess.DEVNULL).decode().strip()
    for flag in cflags.split():
      if flag.startswith("-I"):
        dirs.append(flag[2:])
  except (subprocess.CalledProcessError, FileNotFoundError):
    pass
  if not dirs:
    for p in ["/usr/include/hdf5/serial", "/usr/include"]:
      if os.path.exists(os.path.join(p, "hdf5.h")):
        dirs.append(p)
        break
  return dirs


def _find_hdf5_link():
  """Return (library_dirs, libraries, extra_link_args) for HDF5."""
  lib_dirs = []
  libs = []
  try:
    raw = subprocess.check_output(
        ["pkg-config", "--libs", "hdf5"],
        stderr=subprocess.DEVNULL).decode().strip()
    for flag in raw.split():
      if flag.startswith("-L"):
        lib_dirs.append(flag[2:])
      elif flag.startswith("-l"):
        libs.append(flag[2:])
  except (subprocess.CalledProcessError, FileNotFoundError):
    pass
  if not libs:
    libs = ["hdf5"]
    for p in ["/usr/lib/x86_64-linux-gnu/hdf5/serial",
              "/usr/lib/aarch64-linux-gnu/hdf5/serial",
              "/usr/local/lib"]:
      if os.path.exists(p) and any(
          f.startswith("libhdf5") for f in os.listdir(p)):
        lib_dirs.append(p)
        break
  return lib_dirs, libs


def _has_libminc(prefix):
  """Check whether *prefix* looks like a usable libminc install."""
  inc = os.path.join(prefix, "include", "minc2.h")
  lib_so = os.path.join(prefix, "lib", "libminc2.so")
  lib_a = os.path.join(prefix, "lib", "libminc2.a")
  lib_dylib = os.path.join(prefix, "lib", "libminc2.dylib")
  return os.path.isfile(inc) and (
      os.path.isfile(lib_so) or os.path.isfile(lib_a) or os.path.isfile(lib_dylib))


def _download_libminc(dest_dir):
  """Download and extract the libminc source tarball into *dest_dir*.

  Returns the path to the extracted source root.
  """
  import tarfile
  import urllib.request

  url = "{}/archive/{}.tar.gz".format(_LIBMINC_REPO, _LIBMINC_COMMIT)
  tarball = os.path.join(dest_dir, "libminc-{}.tar.gz".format(_LIBMINC_COMMIT))

  # The extracted directory GitHub creates
  src_dir = os.path.join(dest_dir, "libminc-{}".format(_LIBMINC_COMMIT))
  if os.path.isdir(src_dir):
    print("libminc source already present at {}".format(src_dir))
    return src_dir

  os.makedirs(dest_dir, exist_ok=True)
  print("Downloading libminc from {} ...".format(url))
  urllib.request.urlretrieve(url, tarball)

  print("Extracting ...")
  with tarfile.open(tarball, "r:gz") as tf:
    tf.extractall(dest_dir)
  os.remove(tarball)

  # GitHub archives use the full commit hash; find the directory
  if not os.path.isdir(src_dir):
    # Try to find the extracted directory by pattern
    for entry in os.listdir(dest_dir):
      full = os.path.join(dest_dir, entry)
      if os.path.isdir(full) and entry.startswith("libminc-"):
        os.rename(full, src_dir)
        break

  if not os.path.isdir(src_dir):
    raise RuntimeError("Failed to locate extracted libminc source in " + dest_dir)

  print("libminc source extracted to {}".format(src_dir))
  return src_dir


def _build_libminc(src_dir, build_dir):
  """Build a minimal static libminc2.a and return the path to it.

  Uses cmake + the system C compiler.
  """
  libminc_a = os.path.join(build_dir, "libminc2.a")
  if os.path.isfile(libminc_a):
    print("libminc2.a already built at {}".format(libminc_a))
    return libminc_a

  cmake = shutil.which("cmake")
  if cmake is None:
    raise RuntimeError(
        "cmake is required to build libminc from source but was not found.\n"
        "Install cmake (e.g. 'apt install cmake' or 'brew install cmake') "
        "and try again.")

  os.makedirs(build_dir, exist_ok=True)
  njobs = str(multiprocessing.cpu_count())

  configure_cmd = [
      cmake,
      "-B", build_dir,
      "-S", src_dir,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DLIBMINC_BUILD_SHARED_LIBS=OFF",
      "-DLIBMINC_MINC1_SUPPORT=OFF",
      "-DLIBMINC_BUILD_EZMINC=OFF",
      "-DLIBMINC_USE_NIFTI=OFF",
      "-DBUILD_TESTING=OFF",
      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
  ]

  print("Configuring libminc ...")
  print("  " + " ".join(configure_cmd))
  subprocess.check_call(configure_cmd)

  build_cmd = [cmake, "--build", build_dir, "-j", njobs]
  print("Building libminc ...")
  print("  " + " ".join(build_cmd))
  subprocess.check_call(build_cmd)

  if not os.path.isfile(libminc_a):
    raise RuntimeError(
        "Build succeeded but libminc2.a not found at {}".format(libminc_a))

  print("Built {}".format(libminc_a))
  return libminc_a


def _setup_preinstalled(prefix):
  """Configure for a pre-installed libminc (shared linking)."""
  include_dirs = [os.path.join(prefix, "include")]
  library_dirs = [os.path.join(prefix, "lib")]
  libraries = ["minc2"]
  extra_objects = []

  # HDF5 includes may be needed even with a pre-installed libminc,
  # because minc2.h includes <hdf5.h>
  hdf5_inc = os.path.join(prefix, "include", "hdf5.h")
  if not os.path.isfile(hdf5_inc):
    include_dirs += _find_hdf5_includes()

  # rpath so the shared library is found at runtime
  extra_link_args = []
  libdir = os.path.join(prefix, "lib")
  if platform == "linux" or platform == "linux2":
    extra_link_args = ["-Wl,-rpath={}".format(libdir)]
  elif platform == "darwin":
    extra_link_args = ["-Xlinker", "-rpath", "-Xlinker", libdir]

  return dict(
      include_dirs=include_dirs,
      library_dirs=library_dirs,
      libraries=libraries,
      extra_objects=extra_objects,
      extra_link_args=extra_link_args,
  )


def _setup_autobuild():
  """Download, build, and configure for a static libminc2.a."""
  base_dir = os.path.join("build", "libminc")

  src_dir = _download_libminc(base_dir)
  build_dir = os.path.join(base_dir, "build")
  libminc_a = _build_libminc(src_dir, build_dir)

  # Include dirs from the source tree
  include_dirs = [
      os.path.join(src_dir, "libsrc2"),
      os.path.join(src_dir, "libcommon"),
      os.path.join(src_dir, "volume_io", "Include"),
  ]
  # HDF5 includes
  include_dirs += _find_hdf5_includes()

  # Link flags: static libminc2 + dynamic HDF5/ZLIB/system
  hdf5_lib_dirs, hdf5_libs = _find_hdf5_link()
  libraries = hdf5_libs + ["z", "m"]
  if platform == "linux" or platform == "linux2":
    libraries.append("dl")
  library_dirs = hdf5_lib_dirs

  extra_link_args = []
  # rpath for HDF5 if it's in a non-standard location
  for d in hdf5_lib_dirs:
    if platform == "linux" or platform == "linux2":
      extra_link_args.append("-Wl,-rpath={}".format(d))
    elif platform == "darwin":
      extra_link_args += ["-Xlinker", "-rpath", "-Xlinker", d]

  return dict(
      include_dirs=include_dirs,
      library_dirs=library_dirs,
      libraries=libraries,
      extra_objects=[libminc_a],
      extra_link_args=extra_link_args,
  )


def _resolve_libminc():
  """Determine how to obtain libminc and return build configuration dict.

  Returns a dict with keys: include_dirs, library_dirs, libraries,
  extra_objects, extra_link_args.
  """
  # 1. LIBMINC_DIR — explicit path to a libminc install/build
  libminc_dir = os.environ.get("LIBMINC_DIR")
  if libminc_dir and _has_libminc(libminc_dir):
    print("Using libminc from LIBMINC_DIR={}".format(libminc_dir))
    return _setup_preinstalled(libminc_dir)

  # 2. MINC_TOOLKIT — full toolkit install
  minc_toolkit = os.environ.get("MINC_TOOLKIT")
  if minc_toolkit and _has_libminc(minc_toolkit):
    print("Using libminc from MINC_TOOLKIT={}".format(minc_toolkit))
    return _setup_preinstalled(minc_toolkit)

  # 3. Check default toolkit path
  default_prefix = "/opt/minc/1.9.17"
  if _has_libminc(default_prefix):
    print("Using libminc from default path {}".format(default_prefix))
    return _setup_preinstalled(default_prefix)

  # 4. Auto-build
  if libminc_dir:
    print("WARNING: LIBMINC_DIR={} set but libminc not found there".format(
        libminc_dir))
  if minc_toolkit:
    print("WARNING: MINC_TOOLKIT={} set but libminc not found there".format(
        minc_toolkit))
  print("No pre-installed libminc found. Building static libminc from source ...")
  return _setup_autobuild()


# ── main build configuration ────────────────────────────────────────────

print("*******************")
print("os.getcwd()='{}'".format(os.getcwd()))
print("source_path='{}'".format(source_path))
print("*******************")

cfg = _resolve_libminc()

_extra_compile_args = []
_extra_link_args = cfg["extra_link_args"]

# Read minc2-simple C sources and header
minc2_simple_src = ""
minc2_simple_defs = ""

with open(os.path.join(source_path, "minc2-simple.c"), "r") as f:
  minc2_simple_src = f.read()
with open(os.path.join(source_path, "minc2-matrix-ops.c"), "r") as f:
  minc2_simple_src += f.read()
with open(os.path.join(source_path, "minc2-simple-int.h"), "r") as f:
  minc2_simple_defs = f.read()

# add free system call
minc2_simple_defs += """
void free(void *ptr);
"""

ffibuilder.set_source("minc2_simple._simple",
    minc2_simple_src,
    libraries=cfg["libraries"] + ["c"],
    include_dirs=cfg["include_dirs"] + [source_path],
    library_dirs=cfg["library_dirs"],
    extra_objects=cfg.get("extra_objects", []),
    extra_compile_args=_extra_compile_args,
    extra_link_args=_extra_link_args,
)

ffibuilder.cdef(minc2_simple_defs)


if __name__ == "__main__":
  ffibuilder.compile(verbose=True)
