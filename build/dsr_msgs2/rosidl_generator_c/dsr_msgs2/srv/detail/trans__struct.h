// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/Trans.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/Trans in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__Trans_Request
{
  /// task pos(posx)
  double pos[6];
  /// delta (posx)
  double delta[6];
  /// = 0      # DR_BASE(0), DR_TOOL(1), DR_WORLD(2)
  ///  <DR_WORLD is only available in M2.40 or later>
  int8_t ref;
  /// = 0      # DR_BASE(0), DR_WORLD(2)
  ///  <ref_out is only available in M2.40 or later>
  int8_t ref_out;
} dsr_msgs2__srv__Trans_Request;

// Struct for a sequence of dsr_msgs2__srv__Trans_Request.
typedef struct dsr_msgs2__srv__Trans_Request__Sequence
{
  dsr_msgs2__srv__Trans_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__Trans_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Trans in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__Trans_Response
{
  /// trans pos(posx)
  double trans_pos[6];
  bool success;
} dsr_msgs2__srv__Trans_Response;

// Struct for a sequence of dsr_msgs2__srv__Trans_Response.
typedef struct dsr_msgs2__srv__Trans_Response__Sequence
{
  dsr_msgs2__srv__Trans_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__Trans_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_H_
