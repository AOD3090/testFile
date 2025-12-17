// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/MoveCircle.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'pos'
#include "std_msgs/msg/detail/float64_multi_array__struct.h"

/// Struct defined in srv/MoveCircle in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveCircle_Request
{
  /// target[2][6]
  std_msgs__msg__Float64MultiArray__Sequence pos;
  /// set velocity: [mm/sec], [deg/sec]
  double vel[2];
  /// set acceleration: [mm/sec2], [deg/sec2]
  double acc[2];
  /// = 0.0       # Time
  double time;
  /// =0.0      # Radius under blending mode
  double radius;
  /// DR_BASE(0), DR_TOOL(1), DR_WORLD(2)
  /// <DR_WORLD is only available in M2.40 or later>
  int8_t ref;
  /// = 0         # MOVE_MODE_ABSOLUTE=0, MOVE_MODE_RELATIVE=1
  int8_t mode;
  /// = 0.0     # angle1
  double angle1;
  /// = 0.0     # angle2
  double angle2;
  /// = 0    # BLENDING_SPEED_TYPE_DUPLICATE=0, BLENDING_SPEED_TYPE_OVERRIDE=1
  int8_t blend_type;
  /// =0      # SYNC = 0, ASYNC = 1
  int8_t sync_type;
} dsr_msgs2__srv__MoveCircle_Request;

// Struct for a sequence of dsr_msgs2__srv__MoveCircle_Request.
typedef struct dsr_msgs2__srv__MoveCircle_Request__Sequence
{
  dsr_msgs2__srv__MoveCircle_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveCircle_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/MoveCircle in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveCircle_Response
{
  bool success;
} dsr_msgs2__srv__MoveCircle_Response;

// Struct for a sequence of dsr_msgs2__srv__MoveCircle_Response.
typedef struct dsr_msgs2__srv__MoveCircle_Response__Sequence
{
  dsr_msgs2__srv__MoveCircle_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveCircle_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__STRUCT_H_
