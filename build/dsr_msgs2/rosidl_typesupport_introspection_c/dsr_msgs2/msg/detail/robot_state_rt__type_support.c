// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "dsr_msgs2/msg/detail/robot_state_rt__rosidl_typesupport_introspection_c.h"
#include "dsr_msgs2/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "dsr_msgs2/msg/detail/robot_state_rt__functions.h"
#include "dsr_msgs2/msg/detail/robot_state_rt__struct.h"


// Include directives for member types
// Member `jacobian_matrix`
// Member `coriolis_matrix`
// Member `mass_matrix`
#include "std_msgs/msg/float64_multi_array.h"
// Member `jacobian_matrix`
// Member `coriolis_matrix`
// Member `mass_matrix`
#include "std_msgs/msg/detail/float64_multi_array__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dsr_msgs2__msg__RobotStateRt__init(message_memory);
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_fini_function(void * message_memory)
{
  dsr_msgs2__msg__RobotStateRt__fini(message_memory);
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_position_abs(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position_abs(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position_abs(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_position_abs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position_abs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_position_abs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position_abs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_velocity_abs(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity_abs(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity_abs(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_velocity_abs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity_abs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_velocity_abs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity_abs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_tcp_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_tcp_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_tcp_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_tcp_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_flange_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_flange_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_flange_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_flange_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_flange_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_flange_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_motor_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_motor_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_motor_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_motor_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_motor_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_motor_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_motor_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__raw_joint_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_joint_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_joint_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__raw_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_joint_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__raw_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_joint_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__raw_force_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_force_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_force_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__raw_force_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_force_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__raw_force_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_force_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_joint_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_joint_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_joint_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_joint_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_joint_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_tcp_force(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_tcp_force(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_tcp_force(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_tcp_force(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_tcp_force(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_tcp_force(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_tcp_force(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_acceleration(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_acceleration(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_acceleration(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_acceleration(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_acceleration(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_acceleration(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_acceleration(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_motor_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_motor_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_motor_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_motor_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_motor_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_motor_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_motor_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_tcp_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_tcp_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_tcp_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_tcp_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__jacobian_matrix(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__jacobian_matrix(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__jacobian_matrix(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__jacobian_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__jacobian_matrix(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__jacobian_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__jacobian_matrix(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__jacobian_matrix(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__gravity_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__gravity_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__gravity_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__gravity_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__gravity_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__gravity_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__gravity_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__coriolis_matrix(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__coriolis_matrix(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__coriolis_matrix(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__coriolis_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__coriolis_matrix(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__coriolis_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__coriolis_matrix(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__coriolis_matrix(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__mass_matrix(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__mass_matrix(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__mass_matrix(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__mass_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__mass_matrix(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__mass_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__mass_matrix(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__mass_matrix(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__joint_temperature(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__joint_temperature(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__joint_temperature(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__joint_temperature(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__joint_temperature(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__joint_temperature(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__joint_temperature(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_input_type(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input_type(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input_type(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_input_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input_type(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_input_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input_type(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_output_type(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output_type(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output_type(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_output_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output_type(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_output_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output_type(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_output(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_output(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_output(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__flange_analog_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__flange_analog_input(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__flange_analog_input(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__flange_analog_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__flange_analog_input(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__flange_analog_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__flange_analog_input(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_encoder_strobe_count(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_strobe_count(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_strobe_count(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_encoder_strobe_count(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_strobe_count(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_encoder_strobe_count(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_strobe_count(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_encoder_count(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_count(
  const void * untyped_member, size_t index)
{
  const uint16_t * member =
    (const uint16_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_count(
  void * untyped_member, size_t index)
{
  uint16_t * member =
    (uint16_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_encoder_count(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint16_t * item =
    ((const uint16_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_count(untyped_member, index));
  uint16_t * value =
    (uint16_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_encoder_count(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint16_t * item =
    ((uint16_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_count(untyped_member, index));
  const uint16_t * value =
    (const uint16_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__goal_joint_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_joint_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_joint_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__goal_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_joint_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__goal_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_joint_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__goal_tcp_position(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_tcp_position(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_tcp_position(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__goal_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_tcp_position(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__goal_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_tcp_position(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__reserved(
  const void * untyped_member)
{
  (void)untyped_member;
  return 256;
}

const void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__reserved(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__reserved(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__reserved(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__reserved(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__reserved(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__reserved(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_member_array[46] = {
  {
    "time_stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, time_stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_joint_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_joint_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_joint_position_abs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_joint_position_abs),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_position_abs,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_position_abs,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_position_abs,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_position_abs,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_position_abs,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_joint_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_joint_velocity),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_velocity,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_velocity,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_joint_velocity_abs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_joint_velocity_abs),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_velocity_abs,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_velocity_abs,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_velocity_abs,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_velocity_abs,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_velocity_abs,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_tcp_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_tcp_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_tcp_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_tcp_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_tcp_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_tcp_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_tcp_velocity),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_tcp_velocity,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_tcp_velocity,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_tcp_velocity,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_tcp_velocity,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_tcp_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_flange_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_flange_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_flange_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_flange_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_flange_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_flange_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_flange_velocity),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_flange_velocity,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_flange_velocity,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_flange_velocity,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_flange_velocity,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_flange_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_motor_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_motor_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_motor_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_motor_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_motor_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_motor_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_motor_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_joint_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, actual_joint_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__actual_joint_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__actual_joint_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__actual_joint_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__actual_joint_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__actual_joint_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_joint_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, raw_joint_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__raw_joint_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_joint_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_joint_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__raw_joint_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__raw_joint_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "raw_force_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, raw_force_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__raw_force_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__raw_force_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__raw_force_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__raw_force_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__raw_force_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "external_joint_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, external_joint_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_joint_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_joint_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_joint_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_joint_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_joint_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "external_tcp_force",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, external_tcp_force),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_tcp_force,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_tcp_force,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_tcp_force,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_tcp_force,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_tcp_force,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_joint_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_joint_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_joint_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_joint_velocity),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_velocity,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_velocity,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_velocity,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_velocity,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_joint_acceleration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_joint_acceleration),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_joint_acceleration,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_joint_acceleration,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_joint_acceleration,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_joint_acceleration,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_joint_acceleration,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_motor_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_motor_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_motor_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_motor_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_motor_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_motor_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_motor_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_tcp_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_tcp_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_tcp_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_tcp_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_tcp_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_tcp_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, target_tcp_velocity),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__target_tcp_velocity,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__target_tcp_velocity,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__target_tcp_velocity,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__target_tcp_velocity,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__target_tcp_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "jacobian_matrix",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, jacobian_matrix),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__jacobian_matrix,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__jacobian_matrix,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__jacobian_matrix,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__jacobian_matrix,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__jacobian_matrix,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__jacobian_matrix  // resize(index) function pointer
  },
  {
    "gravity_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, gravity_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__gravity_torque,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__gravity_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__gravity_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__gravity_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__gravity_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "coriolis_matrix",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, coriolis_matrix),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__coriolis_matrix,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__coriolis_matrix,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__coriolis_matrix,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__coriolis_matrix,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__coriolis_matrix,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__coriolis_matrix  // resize(index) function pointer
  },
  {
    "mass_matrix",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, mass_matrix),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__mass_matrix,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__mass_matrix,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__mass_matrix,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__mass_matrix,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__mass_matrix,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__resize_function__RobotStateRt__mass_matrix  // resize(index) function pointer
  },
  {
    "solution_space",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, solution_space),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "singularity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, singularity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "operation_speed_rate",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, operation_speed_rate),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_temperature",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, joint_temperature),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__joint_temperature,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__joint_temperature,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__joint_temperature,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__joint_temperature,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__joint_temperature,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_digital_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_digital_input),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_digital_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_digital_output),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_analog_input_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_analog_input_type),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_input_type,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input_type,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input_type,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_input_type,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_input_type,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_analog_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_analog_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_input,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_analog_output_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_analog_output_type),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_output_type,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output_type,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output_type,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_output_type,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_output_type,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "controller_analog_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, controller_analog_output),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__controller_analog_output,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__controller_analog_output,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__controller_analog_output,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__controller_analog_output,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__controller_analog_output,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "flange_digital_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, flange_digital_input),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "flange_digital_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, flange_digital_output),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "flange_analog_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, flange_analog_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__flange_analog_input,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__flange_analog_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__flange_analog_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__flange_analog_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__flange_analog_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "external_encoder_strobe_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, external_encoder_strobe_count),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_encoder_strobe_count,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_strobe_count,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_strobe_count,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_encoder_strobe_count,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_encoder_strobe_count,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "external_encoder_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, external_encoder_count),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__external_encoder_count,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__external_encoder_count,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__external_encoder_count,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__external_encoder_count,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__external_encoder_count,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "goal_joint_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, goal_joint_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__goal_joint_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_joint_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_joint_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__goal_joint_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__goal_joint_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "goal_tcp_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, goal_tcp_position),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__goal_tcp_position,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__goal_tcp_position,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__goal_tcp_position,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__goal_tcp_position,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__goal_tcp_position,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "robot_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, robot_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "robot_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, robot_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "control_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, control_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "reserved",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    256,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotStateRt, reserved),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__size_function__RobotStateRt__reserved,  // size() function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_const_function__RobotStateRt__reserved,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__get_function__RobotStateRt__reserved,  // get(index) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__fetch_function__RobotStateRt__reserved,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__assign_function__RobotStateRt__reserved,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_members = {
  "dsr_msgs2__msg",  // message namespace
  "RobotStateRt",  // message name
  46,  // number of fields
  sizeof(dsr_msgs2__msg__RobotStateRt),
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_member_array,  // message members
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_init_function,  // function to initialize message memory (memory has to be allocated)
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_type_support_handle = {
  0,
  &dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dsr_msgs2
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, msg, RobotStateRt)() {
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_member_array[21].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_member_array[23].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_member_array[24].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  if (!dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_type_support_handle.typesupport_identifier) {
    dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dsr_msgs2__msg__RobotStateRt__rosidl_typesupport_introspection_c__RobotStateRt_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
