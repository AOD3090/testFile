// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/ServoOff.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SERVO_OFF__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SERVO_OFF__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'STOP_TYPE_QUICK_STO'.
enum
{
  dsr_msgs2__srv__ServoOff_Request__STOP_TYPE_QUICK_STO = 0
};

/// Constant 'STOP_TYPE_QUICK'.
enum
{
  dsr_msgs2__srv__ServoOff_Request__STOP_TYPE_QUICK = 1
};

/// Constant 'STOP_TYPE_SLOW'.
enum
{
  dsr_msgs2__srv__ServoOff_Request__STOP_TYPE_SLOW = 2
};

/// Constant 'STOP_TYPE_HOLD'.
enum
{
  dsr_msgs2__srv__ServoOff_Request__STOP_TYPE_HOLD = 3
};

/// Constant 'STOP_TYPE_EMERGENCY'.
enum
{
  dsr_msgs2__srv__ServoOff_Request__STOP_TYPE_EMERGENCY = 3
};

/// Struct defined in srv/ServoOff in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__ServoOff_Request
{
  int8_t stop_type;
} dsr_msgs2__srv__ServoOff_Request;

// Struct for a sequence of dsr_msgs2__srv__ServoOff_Request.
typedef struct dsr_msgs2__srv__ServoOff_Request__Sequence
{
  dsr_msgs2__srv__ServoOff_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__ServoOff_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/ServoOff in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__ServoOff_Response
{
  bool success;
} dsr_msgs2__srv__ServoOff_Response;

// Struct for a sequence of dsr_msgs2__srv__ServoOff_Response.
typedef struct dsr_msgs2__srv__ServoOff_Response__Sequence
{
  dsr_msgs2__srv__ServoOff_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__ServoOff_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SERVO_OFF__STRUCT_H_
