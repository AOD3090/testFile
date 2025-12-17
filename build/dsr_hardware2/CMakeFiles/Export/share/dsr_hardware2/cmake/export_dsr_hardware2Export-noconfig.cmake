#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dsr_hardware2::dsr_hardware2" for configuration ""
set_property(TARGET dsr_hardware2::dsr_hardware2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(dsr_hardware2::dsr_hardware2 PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libdsr_hardware2.so"
  IMPORTED_SONAME_NOCONFIG "libdsr_hardware2.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS dsr_hardware2::dsr_hardware2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_dsr_hardware2::dsr_hardware2 "${_IMPORT_PREFIX}/lib/libdsr_hardware2.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
