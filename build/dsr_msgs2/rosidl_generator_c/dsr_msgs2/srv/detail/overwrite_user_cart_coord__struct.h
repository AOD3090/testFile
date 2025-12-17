// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/OverwriteUserCartCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__OVERWRITE_USER_CART_COORD__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__OVERWRITE_USER_CART_COORD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/OverwriteUserCartCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__OverwriteUserCartCoord_Request
{
  /// ID of user coord
  int8_t id;
  /// task pos(posx)
  double pos[6];
  /// = 0   # DR_BASE(0), DR_WORLD(2)
  int8_t ref;
} dsr_msgs2__srv__OverwriteUserCartCoord_Request;

// Struct for a sequence of dsr_msgs2__srv__OverwriteUserCartCoord_Request.
typedef struct dsr_msgs2__srv__OverwriteUserCartCoord_Request__Sequence
{
  dsr_msgs2__srv__OverwriteUserCartCoord_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__OverwriteUserCartCoord_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/OverwriteUserCartCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__OverwriteUserCartCoord_Response
{
  /// Successful coordinate setting, Set user coordinate ID (101 - 200)
  /// (-1) Failed coordinate setting
  int8_t id;
  bool success;
} dsr_msgs2__srv__OverwriteUserCartCoord_Response;

// Struct for a sequence of dsr_msgs2__srv__OverwriteUserCartCoord_Response.
typedef struct dsr_msgs2__srv__OverwriteUserCartCoord_Response__Sequence
{
  dsr_msgs2__srv__OverwriteUserCartCoord_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__OverwriteUserCartCoord_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__OVERWRITE_USER_CART_COORD__STRUCT_H_
