#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "coreneuron" for configuration "Release"
set_property(TARGET coreneuron APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(coreneuron PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcoreneuron.so"
  IMPORTED_SONAME_RELEASE "libcoreneuron.so"
  )

list(APPEND _cmake_import_check_targets coreneuron )
list(APPEND _cmake_import_check_files_for_coreneuron "${_IMPORT_PREFIX}/lib/libcoreneuron.so" )

# Import target "scopmath" for configuration "Release"
set_property(TARGET scopmath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scopmath PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libscopmath.a"
  )

list(APPEND _cmake_import_check_targets scopmath )
list(APPEND _cmake_import_check_files_for_scopmath "${_IMPORT_PREFIX}/lib/libscopmath.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
