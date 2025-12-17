// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/TaskComplianceCtrl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/TaskComplianceCtrl in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__TaskComplianceCtrl_Request
{
  /// Three translational stiffnesses + Three rotational stiffnesses
  /// default  [3000, 3000, 3000, 200, 200, 200]
  double stx[6];
  /// the preset reference coordinate system.
  int8_t ref;
  /// Stiffness varying time, Linear transition during the specified time
  double time;
} dsr_msgs2__srv__TaskComplianceCtrl_Request;

// Struct for a sequence of dsr_msgs2__srv__TaskComplianceCtrl_Request.
typedef struct dsr_msgs2__srv__TaskComplianceCtrl_Request__Sequence
{
  dsr_msgs2__srv__TaskComplianceCtrl_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__TaskComplianceCtrl_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/TaskComplianceCtrl in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__TaskComplianceCtrl_Response
{
  bool success;
} dsr_msgs2__srv__TaskComplianceCtrl_Response;

// Struct for a sequence of dsr_msgs2__srv__TaskComplianceCtrl_Response.
typedef struct dsr_msgs2__srv__TaskComplianceCtrl_Response__Sequence
{
  dsr_msgs2__srv__TaskComplianceCtrl_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__TaskComplianceCtrl_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__STRUCT_H_
