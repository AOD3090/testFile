// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetDesiredForce.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetDesiredForce in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetDesiredForce_Request
{
  /// Three translational target forces + Three rotational target moments
  double fd[6];
  /// Force control in the corresponding direction if 1, Compliance control in the corresponding direction if 0
  int8_t dir[6];
  /// Reference coordinate of the coordinate to get
  int8_t ref;
  /// 0          # Transition time of target force to take effect (0 ~ 1.0 sec)
  double time;
  /// DR_FC_MOD_ABS(0): force control with absolute value,
  /// DR_FC_MOD_REL(1): force control with relative value to initial state (the instance when this function is called)
  int8_t mod;
} dsr_msgs2__srv__SetDesiredForce_Request;

// Struct for a sequence of dsr_msgs2__srv__SetDesiredForce_Request.
typedef struct dsr_msgs2__srv__SetDesiredForce_Request__Sequence
{
  dsr_msgs2__srv__SetDesiredForce_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetDesiredForce_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetDesiredForce in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetDesiredForce_Response
{
  bool success;
} dsr_msgs2__srv__SetDesiredForce_Response;

// Struct for a sequence of dsr_msgs2__srv__SetDesiredForce_Response.
typedef struct dsr_msgs2__srv__SetDesiredForce_Response__Sequence
{
  dsr_msgs2__srv__SetDesiredForce_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetDesiredForce_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__STRUCT_H_
