// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/MoveSpiral.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/MoveSpiral in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveSpiral_Request
{
  /// Total number of revolutions
  double revolution;
  /// Final spiral radius
  double max_radius;
  /// Distance moved in the axis direction
  double max_length;
  /// Target position. If used, max_radius and max_length are ignored
  double target_pos[3];
  /// set velocity: [mm/sec], [deg/sec]
  double vel[2];
  /// set acceleration: [mm/sec2], [deg/sec2]
  double acc[2];
  /// = 0.0      # Total execution time <sec>
  double time;
  /// TASK_AXIS_X = 0, TASK_AXIS_Y = 1, TASK_AXIS_Z = 2
  int8_t task_axis;
  /// = 1        # DR_BASE(0), DR_TOOL(1), DR_WORLD(2)
  ///  <DR_WORLD is only available in M2.40 or later
  int8_t ref;
  /// = 0        # MOVE_MODE_ABSOLUTE=0, MOVE_MODE_RELATIVE=1
  int8_t mode;
  /// = 0  # MOVE_SPIRAL_OUTWARD=0, MOVE_SPIRAL_INWARD=1
  int8_t spiral_dir;
  /// =0      # MOVE_SPIRAL_FORWARD=0, MOVE_SPIRAL_REVERSE=1
  int8_t rot_dir;
  /// =0    # SYNC = 0, ASYNC = 1
  int8_t sync_type;
} dsr_msgs2__srv__MoveSpiral_Request;

// Struct for a sequence of dsr_msgs2__srv__MoveSpiral_Request.
typedef struct dsr_msgs2__srv__MoveSpiral_Request__Sequence
{
  dsr_msgs2__srv__MoveSpiral_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveSpiral_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/MoveSpiral in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveSpiral_Response
{
  bool success;
} dsr_msgs2__srv__MoveSpiral_Response;

// Struct for a sequence of dsr_msgs2__srv__MoveSpiral_Response.
typedef struct dsr_msgs2__srv__MoveSpiral_Response__Sequence
{
  dsr_msgs2__srv__MoveSpiral_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveSpiral_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_H_
