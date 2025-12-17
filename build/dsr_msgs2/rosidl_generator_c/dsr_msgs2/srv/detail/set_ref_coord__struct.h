// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetRefCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetRefCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetRefCoord_Request
{
  /// DR_BASE(0), DR_TOOL(1), DR_WORLD(2), user coord(101~200)
  /// <DR_WORLD is only available in M2.40 or later>
  int8_t coord;
} dsr_msgs2__srv__SetRefCoord_Request;

// Struct for a sequence of dsr_msgs2__srv__SetRefCoord_Request.
typedef struct dsr_msgs2__srv__SetRefCoord_Request__Sequence
{
  dsr_msgs2__srv__SetRefCoord_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetRefCoord_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetRefCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetRefCoord_Response
{
  bool success;
} dsr_msgs2__srv__SetRefCoord_Response;

// Struct for a sequence of dsr_msgs2__srv__SetRefCoord_Response.
typedef struct dsr_msgs2__srv__SetRefCoord_Response__Sequence
{
  dsr_msgs2__srv__SetRefCoord_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetRefCoord_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__STRUCT_H_
