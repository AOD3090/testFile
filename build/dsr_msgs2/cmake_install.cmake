# Install script for directory: /home/user/Jina/src/doosan-robot2/dsr_msgs2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/user/Jina/install/dsr_msgs2")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rosidl_interfaces" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_index/share/ament_index/resource_index/rosidl_interfaces/dsr_msgs2")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_c/dsr_msgs2/" REGEX "/[^/]*\\.h$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/opt/ros/humble/lib/python3.10/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/library_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_generator_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_typesupport_fastrtps_c/dsr_msgs2/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_cpp/dsr_msgs2/" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_typesupport_fastrtps_cpp/dsr_msgs2/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_fastrtps_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_typesupport_introspection_c/dsr_msgs2/" REGEX "/[^/]*\\.h$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_introspection_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/dsr_msgs2/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_typesupport_introspection_cpp/dsr_msgs2/" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_introspection_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/libdsr_msgs2__rosidl_typesupport_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_typesupport_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/pythonpath.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/pythonpath.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2-1.1.0-py3.10.egg-info" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_python/dsr_msgs2/dsr_msgs2.egg-info/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2" TYPE DIRECTORY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2/" REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(
        COMMAND
        "/usr/bin/python3" "-m" "compileall"
        "/home/user/Jina/install/dsr_msgs2/local/lib/python3.10/dist-packages/dsr_msgs2"
      )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2:/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2:/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2:/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/dsr_msgs2/dsr_msgs2_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/user/Jina/build/dsr_msgs2/rosidl_generator_py/dsr_msgs2/libdsr_msgs2__rosidl_generator_py.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so"
         OLD_RPATH "/home/user/Jina/build/dsr_msgs2:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdsr_msgs2__rosidl_generator_py.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/LogAlarm.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/ModbusState.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/RobotError.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/RobotState.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/RobotStop.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/JogMultiAxis.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/AlterMotionStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/ServojStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/ServolStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/SpeedjStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/SpeedlStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/RobotDisconnection.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/RobotStateRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/ServojRtStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/ServolRtStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/SpeedjRtStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/SpeedlRtStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/msg/TorqueRtStream.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRobotMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRobotMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRobotSystem.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRobotSystem.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRobotSpeedMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRobotSpeedMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentPose.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetSafeStopResetType.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetLastAlarm.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRobotState.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ServoOff.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRobotControl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ChangeCollisionSensitivity.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetSafetyMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveJoint.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveLine.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveJointx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveCircle.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveSplineJoint.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveSplineTask.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveBlending.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveSpiral.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MovePeriodic.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveWait.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Jog.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/JogMulti.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveStop.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MovePause.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveResume.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Trans.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Fkin.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Ikin.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRefCoord.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/MoveHome.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CheckMotion.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ChangeOperationSpeed.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/EnableAlterMotion.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/AlterMotion.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DisableAlterMotion.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetSingularityHandling.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetControlMode.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetControlSpace.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentPosj.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetDesiredPosj.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentVelj.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetDesiredVelj.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentPosx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentToolFlangePosx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentVelx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetDesiredPosx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetDesiredVelx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentSolutionSpace.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentRotm.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetJointTorque.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetExternalTorque.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetToolForce.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetSolutionSpace.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetOrientationError.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ParallelAxis1.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ParallelAxis2.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/AlignAxis1.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/AlignAxis2.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/IsDoneBoltTightening.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ReleaseComplianceCtrl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/TaskComplianceCtrl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetStiffnessx.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CalcCoord.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetUserCartCoord1.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetUserCartCoord2.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetUserCartCoord3.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/OverwriteUserCartCoord.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetUserCartCoord.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetDesiredForce.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ReleaseForce.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CheckPositionCondition.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CheckForceCondition.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CheckOrientationCondition1.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CheckOrientationCondition2.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/CoordTransform.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetWorkpieceWeight.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ResetWorkpieceWeight.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigCreateTool.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigDeleteTool.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCurrentTool.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentTool.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetToolShape.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigCreateTcp.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigDeleteTcp.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCurrentTcp.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCurrentTcp.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetToolDigitalOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetToolDigitalOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetToolDigitalInput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCtrlBoxDigitalOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCtrlBoxDigitalOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCtrlBoxDigitalInput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCtrlBoxAnalogInputType.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCtrlBoxAnalogOutputType.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetCtrlBoxAnalogOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetCtrlBoxAnalogInput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigCreateModbus.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConfigDeleteModbus.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetModbusOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetModbusInput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DrlStart.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DrlStop.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DrlPause.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DrlResume.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetDrlState.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Robotiq2FClose.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Robotiq2FOpen.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/Robotiq2FMove.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SerialSendData.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ConnectRtControl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/DisconnectRtControl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRtControlInputDataList.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRtControlInputVersionList.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRtControlOutputDataList.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/GetRtControlOutputVersionList.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/ReadDataRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetAccjRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetAccxRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRtControlInput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetRtControlOutput.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetVeljRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/SetVelxRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/StartRtControl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/StopRtControl.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/srv" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_adapter/dsr_msgs2/srv/WriteDataRt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/LogAlarm.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/ModbusState.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/RobotError.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/RobotState.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/RobotStop.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/JogMultiAxis.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/AlterMotionStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/ServojStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/ServolStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/SpeedjStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/SpeedlStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/RobotDisconnection.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/RobotStateRt.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/ServojRtStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/ServolRtStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/SpeedjRtStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/SpeedlRtStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/msg" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/msg/TorqueRtStream.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetRobotMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetRobotMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetRobotSystem.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotSystem_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotSystem_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetRobotSystem.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotSystem_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotSystem_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetRobotSpeedMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotSpeedMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotSpeedMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetRobotSpeedMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotSpeedMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotSpeedMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetCurrentPose.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetCurrentPose_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetCurrentPose_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetSafeStopResetType.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetSafeStopResetType_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetSafeStopResetType_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetLastAlarm.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetLastAlarm_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetLastAlarm_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/GetRobotState.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotState_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/GetRobotState_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/ServoOff.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/ServoOff_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/ServoOff_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetRobotControl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotControl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetRobotControl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/ChangeCollisionSensitivity.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/ChangeCollisionSensitivity_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/ChangeCollisionSensitivity_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/system/SetSafetyMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetSafetyMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/system" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/system/SetSafetyMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveJoint.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveJoint_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveJoint_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveLine.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveLine_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveLine_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveJointx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveJointx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveJointx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveCircle.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveCircle_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveCircle_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveSplineJoint.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSplineJoint_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSplineJoint_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveSplineTask.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSplineTask_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSplineTask_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveBlending.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveBlending_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveBlending_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveSpiral.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSpiral_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveSpiral_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MovePeriodic.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MovePeriodic_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MovePeriodic_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveWait.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveWait_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveWait_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/Jog.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Jog_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Jog_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/JogMulti.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/JogMulti_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/JogMulti_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveStop.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveStop_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveStop_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MovePause.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MovePause_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MovePause_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveResume.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveResume_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveResume_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/Trans.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Trans_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Trans_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/Fkin.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Fkin_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Fkin_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/Ikin.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Ikin_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/Ikin_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/SetRefCoord.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/SetRefCoord_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/SetRefCoord_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/MoveHome.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveHome_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/MoveHome_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/CheckMotion.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/CheckMotion_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/CheckMotion_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/ChangeOperationSpeed.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/ChangeOperationSpeed_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/ChangeOperationSpeed_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/EnableAlterMotion.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/EnableAlterMotion_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/EnableAlterMotion_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/AlterMotion.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/AlterMotion_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/AlterMotion_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/DisableAlterMotion.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/DisableAlterMotion_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/DisableAlterMotion_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/motion/SetSingularityHandling.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/SetSingularityHandling_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/motion" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/motion/SetSingularityHandling_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetControlMode.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetControlMode_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetControlMode_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetControlSpace.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetControlSpace_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetControlSpace_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentPosj.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentPosj_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentPosj_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetDesiredPosj.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredPosj_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredPosj_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentVelj.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentVelj_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentVelj_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetDesiredVelj.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredVelj_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredVelj_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentPosx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentPosx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentPosx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentToolFlangePosx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentToolFlangePosx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentToolFlangePosx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentVelx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentVelx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentVelx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetDesiredPosx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredPosx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredPosx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetDesiredVelx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredVelx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetDesiredVelx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentSolutionSpace.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentSolutionSpace_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentSolutionSpace_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetCurrentRotm.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentRotm_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetCurrentRotm_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetJointTorque.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetJointTorque_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetJointTorque_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetExternalTorque.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetExternalTorque_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetExternalTorque_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetToolForce.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetToolForce_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetToolForce_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetSolutionSpace.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetSolutionSpace_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetSolutionSpace_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/aux_control/GetOrientationError.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetOrientationError_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/aux_control" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/aux_control/GetOrientationError_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/ParallelAxis1.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ParallelAxis1_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ParallelAxis1_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/ParallelAxis2.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ParallelAxis2_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ParallelAxis2_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/AlignAxis1.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/AlignAxis1_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/AlignAxis1_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/AlignAxis2.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/AlignAxis2_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/AlignAxis2_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/IsDoneBoltTightening.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/IsDoneBoltTightening_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/IsDoneBoltTightening_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/ReleaseComplianceCtrl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ReleaseComplianceCtrl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ReleaseComplianceCtrl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/TaskComplianceCtrl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/TaskComplianceCtrl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/TaskComplianceCtrl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/SetStiffnessx.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetStiffnessx_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetStiffnessx_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CalcCoord.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CalcCoord_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CalcCoord_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/SetUserCartCoord1.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord1_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord1_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/SetUserCartCoord2.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord2_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord2_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/SetUserCartCoord3.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord3_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetUserCartCoord3_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/OverwriteUserCartCoord.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/OverwriteUserCartCoord_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/OverwriteUserCartCoord_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/GetUserCartCoord.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/GetUserCartCoord_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/GetUserCartCoord_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/SetDesiredForce.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetDesiredForce_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/SetDesiredForce_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/ReleaseForce.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ReleaseForce_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ReleaseForce_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CheckPositionCondition.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckPositionCondition_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckPositionCondition_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CheckForceCondition.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckForceCondition_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckForceCondition_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CheckOrientationCondition1.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckOrientationCondition1_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckOrientationCondition1_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CheckOrientationCondition2.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckOrientationCondition2_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CheckOrientationCondition2_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/CoordTransform.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CoordTransform_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/CoordTransform_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/GetWorkpieceWeight.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/GetWorkpieceWeight_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/GetWorkpieceWeight_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/force/ResetWorkpieceWeight.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ResetWorkpieceWeight_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/force" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/force/ResetWorkpieceWeight_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tool/ConfigCreateTool.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/ConfigCreateTool_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/ConfigCreateTool_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tool/ConfigDeleteTool.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/ConfigDeleteTool_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/ConfigDeleteTool_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tool/SetCurrentTool.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/SetCurrentTool_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/SetCurrentTool_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tool/GetCurrentTool.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/GetCurrentTool_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/GetCurrentTool_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tool/SetToolShape.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/SetToolShape_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tool" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tool/SetToolShape_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tcp/ConfigCreateTcp.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/ConfigCreateTcp_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/ConfigCreateTcp_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tcp/ConfigDeleteTcp.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/ConfigDeleteTcp_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/ConfigDeleteTcp_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tcp/SetCurrentTcp.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/SetCurrentTcp_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/SetCurrentTcp_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/tcp/GetCurrentTcp.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/GetCurrentTcp_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/tcp" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/tcp/GetCurrentTcp_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/SetToolDigitalOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetToolDigitalOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetToolDigitalOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/GetToolDigitalOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetToolDigitalOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetToolDigitalOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/GetToolDigitalInput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetToolDigitalInput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetToolDigitalInput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/SetCtrlBoxDigitalOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxDigitalOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxDigitalOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/GetCtrlBoxDigitalOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxDigitalOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxDigitalOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/GetCtrlBoxDigitalInput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxDigitalInput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxDigitalInput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/SetCtrlBoxAnalogInputType.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogInputType_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogInputType_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/SetCtrlBoxAnalogOutputType.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogOutputType_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogOutputType_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/SetCtrlBoxAnalogOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/SetCtrlBoxAnalogOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/io/GetCtrlBoxAnalogInput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxAnalogInput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/io" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/io/GetCtrlBoxAnalogInput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/modbus/ConfigCreateModbus.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/ConfigCreateModbus_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/ConfigCreateModbus_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/modbus/ConfigDeleteModbus.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/ConfigDeleteModbus_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/ConfigDeleteModbus_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/modbus/SetModbusOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/SetModbusOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/SetModbusOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/modbus/GetModbusInput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/GetModbusInput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/modbus" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/modbus/GetModbusInput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/drl/DrlStart.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlStart_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlStart_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/drl/DrlStop.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlStop_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlStop_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/drl/DrlPause.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlPause_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlPause_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/drl/DrlResume.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlResume_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/DrlResume_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/drl/GetDrlState.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/GetDrlState_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/drl" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/drl/GetDrlState_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/gripper/Robotiq2FClose.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FClose_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FClose_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/gripper/Robotiq2FOpen.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FOpen_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FOpen_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/gripper/Robotiq2FMove.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FMove_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/Robotiq2FMove_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/gripper/SerialSendData.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/SerialSendData_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/gripper" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/gripper/SerialSendData_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/ConnectRtControl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/ConnectRtControl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/ConnectRtControl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/DisconnectRtControl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/DisconnectRtControl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/DisconnectRtControl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/GetRtControlInputDataList.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlInputDataList_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlInputDataList_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/GetRtControlInputVersionList.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlInputVersionList_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlInputVersionList_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/GetRtControlOutputDataList.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlOutputDataList_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlOutputDataList_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/GetRtControlOutputVersionList.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlOutputVersionList_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/GetRtControlOutputVersionList_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/ReadDataRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/ReadDataRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/ReadDataRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetAccjRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetAccjRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetAccjRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetAccxRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetAccxRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetAccxRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetRtControlInput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetRtControlInput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetRtControlInput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetRtControlOutput.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetRtControlOutput_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetRtControlOutput_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetVeljRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetVeljRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetVeljRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/SetVelxRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetVelxRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/SetVelxRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/StartRtControl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/StartRtControl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/StartRtControl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/StopRtControl.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/StopRtControl_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/StopRtControl_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/srv/realtime/WriteDataRt.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/WriteDataRt_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/realtime" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/srv/realtime/WriteDataRt_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/dsr_msgs2")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/dsr_msgs2")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/environment" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_index/share/ament_index/resource_index/packages/dsr_msgs2")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_cppExport.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_typesupport_fastrtps_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_introspection_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/dsr_msgs2__rosidl_typesupport_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport.cmake"
         "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/CMakeFiles/Export/share/dsr_msgs2/cmake/export_dsr_msgs2__rosidl_generator_pyExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/rosidl_cmake-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/ament_cmake_export_targets/ament_cmake_export_targets-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/rosidl_cmake_export_typesupport_targets-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES "/home/user/Jina/build/dsr_msgs2/rosidl_cmake/rosidl_cmake_export_typesupport_libraries-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2/cmake" TYPE FILE FILES
    "/home/user/Jina/build/dsr_msgs2/ament_cmake_core/dsr_msgs2Config.cmake"
    "/home/user/Jina/build/dsr_msgs2/ament_cmake_core/dsr_msgs2Config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/dsr_msgs2" TYPE FILE FILES "/home/user/Jina/src/doosan-robot2/dsr_msgs2/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/user/Jina/build/dsr_msgs2/dsr_msgs2__py/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/user/Jina/build/dsr_msgs2/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
