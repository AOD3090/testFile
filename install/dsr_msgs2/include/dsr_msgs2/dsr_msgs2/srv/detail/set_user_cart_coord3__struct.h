// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetUserCartCoord3.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetUserCartCoord3 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetUserCartCoord3_Request
{
  /// X-axis unit vector
  double u1[3];
  /// Y-axis unit vector
  double v1[3];
  /// task pos(posx)
  double pos[6];
  /// DR_BASE(0), DR_WORLD(2)
  /// <ref is only available in M2.40 or later>
  int8_t ref;
} dsr_msgs2__srv__SetUserCartCoord3_Request;

// Struct for a sequence of dsr_msgs2__srv__SetUserCartCoord3_Request.
typedef struct dsr_msgs2__srv__SetUserCartCoord3_Request__Sequence
{
  dsr_msgs2__srv__SetUserCartCoord3_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetUserCartCoord3_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetUserCartCoord3 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetUserCartCoord3_Response
{
  /// set user coord (101~120) or fail(-1)
  int8_t id;
  bool success;
} dsr_msgs2__srv__SetUserCartCoord3_Response;

// Struct for a sequence of dsr_msgs2__srv__SetUserCartCoord3_Response.
typedef struct dsr_msgs2__srv__SetUserCartCoord3_Response__Sequence
{
  dsr_msgs2__srv__SetUserCartCoord3_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetUserCartCoord3_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__STRUCT_H_
