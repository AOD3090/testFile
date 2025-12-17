// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/Ikin.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__IKIN__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__IKIN__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/Ikin in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__Ikin_Request
{
  /// task pos(posx)
  double pos[6];
  /// solution space : 0 ~ 7
  int8_t sol_space;
  /// = 0      # DR_BASE(0), DR_WORLD(2)
  ///  <ref is only available in M2.40 or later>
  int8_t ref;
} dsr_msgs2__srv__Ikin_Request;

// Struct for a sequence of dsr_msgs2__srv__Ikin_Request.
typedef struct dsr_msgs2__srv__Ikin_Request__Sequence
{
  dsr_msgs2__srv__Ikin_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__Ikin_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Ikin in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__Ikin_Response
{
  /// joint pos(posj)
  double conv_posj[6];
  bool success;
} dsr_msgs2__srv__Ikin_Response;

// Struct for a sequence of dsr_msgs2__srv__Ikin_Response.
typedef struct dsr_msgs2__srv__Ikin_Response__Sequence
{
  dsr_msgs2__srv__Ikin_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__Ikin_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__IKIN__STRUCT_H_
