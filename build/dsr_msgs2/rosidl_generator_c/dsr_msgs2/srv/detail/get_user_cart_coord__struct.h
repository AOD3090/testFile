// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/GetUserCartCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetUserCartCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetUserCartCoord_Request
{
  /// ID of user coord
  int8_t id;
} dsr_msgs2__srv__GetUserCartCoord_Request;

// Struct for a sequence of dsr_msgs2__srv__GetUserCartCoord_Request.
typedef struct dsr_msgs2__srv__GetUserCartCoord_Request__Sequence
{
  dsr_msgs2__srv__GetUserCartCoord_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetUserCartCoord_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GetUserCartCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetUserCartCoord_Response
{
  /// task pos(posx)
  double conv_posx[6];
  /// Reference coordinate of the coordinate to get
  int8_t ref;
  bool success;
} dsr_msgs2__srv__GetUserCartCoord_Response;

// Struct for a sequence of dsr_msgs2__srv__GetUserCartCoord_Response.
typedef struct dsr_msgs2__srv__GetUserCartCoord_Response__Sequence
{
  dsr_msgs2__srv__GetUserCartCoord_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetUserCartCoord_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__STRUCT_H_
