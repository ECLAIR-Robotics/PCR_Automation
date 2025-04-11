# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_sipeed_tof_ms_a010_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED sipeed_tof_ms_a010_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(sipeed_tof_ms_a010_FOUND FALSE)
  elseif(NOT sipeed_tof_ms_a010_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(sipeed_tof_ms_a010_FOUND FALSE)
  endif()
  return()
endif()
set(_sipeed_tof_ms_a010_CONFIG_INCLUDED TRUE)

# output package information
if(NOT sipeed_tof_ms_a010_FIND_QUIETLY)
  message(STATUS "Found sipeed_tof_ms_a010: 0.0.0 (${sipeed_tof_ms_a010_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'sipeed_tof_ms_a010' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${sipeed_tof_ms_a010_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(sipeed_tof_ms_a010_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${sipeed_tof_ms_a010_DIR}/${_extra}")
endforeach()
