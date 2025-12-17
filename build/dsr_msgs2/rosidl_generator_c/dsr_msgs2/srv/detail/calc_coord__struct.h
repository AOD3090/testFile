// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/CalcCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/CalcCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CalcCoord_Request
{
  /// input_pos_cnt
  int8_t input_pos_cnt;
  /// task pos(posx)
  double x1[6];
  /// task pos(posx)
  double x2[6];
  /// task pos(posx)
  double x3[6];
  /// task pos(posx)
  double x4[6];
  /// DR_BASE(0), DR_WORLD(2)
  int8_t ref;
  /// input mode(only valid when the number of input poses is 2)
  /// 0: defining z-axis based on the current Tool-z direction
  /// 1: defining z-axis based on the z direction of x1
  int8_t mod;
} dsr_msgs2__srv__CalcCoord_Request;

// Struct for a sequence of dsr_msgs2__srv__CalcCoord_Request.
typedef struct dsr_msgs2__srv__CalcCoord_Request__Sequence
{
  dsr_msgs2__srv__CalcCoord_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CalcCoord_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/CalcCoord in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CalcCoord_Response
{
  /// task pos(posx)
  double conv_posx[6];
  bool success;
} dsr_msgs2__srv__CalcCoord_Response;

// Struct for a sequence of dsr_msgs2__srv__CalcCoord_Response.
typedef struct dsr_msgs2__srv__CalcCoord_Response__Sequence
{
  dsr_msgs2__srv__CalcCoord_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CalcCoord_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_H_
