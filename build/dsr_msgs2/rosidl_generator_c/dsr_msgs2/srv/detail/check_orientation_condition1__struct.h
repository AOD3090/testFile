// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/CheckOrientationCondition1.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/CheckOrientationCondition1 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CheckOrientationCondition1_Request
{
  /// DR_AXIS_A(10), DR_AXIS_B(11), DR_AXIS_C(12)
  int8_t axis;
  /// task pos(posx)
  double min[6];
  /// task pos(posx)
  double max[6];
  /// = 0         # DR_BASE(0), DR_TOOL(1), DR_WORLD(2), user_coordinate(101~200)
  ///  <DR_WORLD is only available in M2.40 or later>
  int8_t ref;
  /// = 0         # DR_MV_MOD_ABS(0)
  int8_t mode;
} dsr_msgs2__srv__CheckOrientationCondition1_Request;

// Struct for a sequence of dsr_msgs2__srv__CheckOrientationCondition1_Request.
typedef struct dsr_msgs2__srv__CheckOrientationCondition1_Request__Sequence
{
  dsr_msgs2__srv__CheckOrientationCondition1_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CheckOrientationCondition1_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/CheckOrientationCondition1 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CheckOrientationCondition1_Response
{
  /// True or False
  bool success;
} dsr_msgs2__srv__CheckOrientationCondition1_Response;

// Struct for a sequence of dsr_msgs2__srv__CheckOrientationCondition1_Response.
typedef struct dsr_msgs2__srv__CheckOrientationCondition1_Response__Sequence
{
  dsr_msgs2__srv__CheckOrientationCondition1_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CheckOrientationCondition1_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__STRUCT_H_
