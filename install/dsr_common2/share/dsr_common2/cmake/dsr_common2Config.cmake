# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_dsr_common2_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED dsr_common2_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(dsr_common2_FOUND FALSE)
  elseif(NOT dsr_common2_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(dsr_common2_FOUND FALSE)
  endif()
  return()
endif()
set(_dsr_common2_CONFIG_INCLUDED TRUE)

# output package information
if(NOT dsr_common2_FIND_QUIETLY)
  message(STATUS "Found dsr_common2: 1.1.0 (${dsr_common2_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'dsr_common2' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${dsr_common2_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(dsr_common2_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${dsr_common2_DIR}/${_extra}")
endforeach()
