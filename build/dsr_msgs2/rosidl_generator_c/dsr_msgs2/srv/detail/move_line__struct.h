// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/MoveLine.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/MoveLine in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveLine_Request
{
  /// target
  double pos[6];
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
  /// = 0         # DR_MV_MOD_ABS(0), DR_MV_MOD_REL(1)
  int8_t mode;
  /// = 0    # BLENDING_SPEED_TYPE_DUPLICATE=0, BLENDING_SPEED_TYPE_OVERRIDE=1
  int8_t blend_type;
  /// =0      # SYNC = 0, ASYNC = 1
  int8_t sync_type;
} dsr_msgs2__srv__MoveLine_Request;

// Struct for a sequence of dsr_msgs2__srv__MoveLine_Request.
typedef struct dsr_msgs2__srv__MoveLine_Request__Sequence
{
  dsr_msgs2__srv__MoveLine_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveLine_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/MoveLine in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveLine_Response
{
  bool success;
} dsr_msgs2__srv__MoveLine_Response;

// Struct for a sequence of dsr_msgs2__srv__MoveLine_Response.
typedef struct dsr_msgs2__srv__MoveLine_Response__Sequence
{
  dsr_msgs2__srv__MoveLine_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveLine_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_H_
