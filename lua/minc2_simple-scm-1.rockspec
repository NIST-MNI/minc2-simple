package = "minc2_simple"
version = "scm-1"
source = {
   url = "..."
}
description = {
   homepage = "https://github.com/vfonov/minc2-simple",
   license = "BSD"
}
dependencies = {}
build = {
   type = "cmake",
   modules = {},
   variables = {
    CMAKE_BUILD_TYPE="Release",
    CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
    CMAKE_INSTALL_PREFIX="$(PREFIX)"
  }
}
