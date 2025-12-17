// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "dsr_msgs2/msg/detail/robot_state__rosidl_typesupport_introspection_c.h"
#include "dsr_msgs2/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "dsr_msgs2/msg/detail/robot_state__functions.h"
#include "dsr_msgs2/msg/detail/robot_state__struct.h"


// Include directives for member types
// Member `robot_state_str`
#include "rosidl_runtime_c/string_functions.h"
// Member `rotation_matrix`
// Member `f_current_pos_world`
// Member `f_rotation_matrix_world`
// Member `f_current_pos_user`
// Member `f_rotation_matrix_user`
#include "std_msgs/msg/float64_multi_array.h"
// Member `rotation_matrix`
// Member `f_current_pos_world`
// Member `f_rotation_matrix_world`
// Member `f_current_pos_user`
// Member `f_rotation_matrix_user`
#include "std_msgs/msg/detail/float64_multi_array__rosidl_typesupport_introspection_c.h"
// Member `modbus_state`
#include "dsr_msgs2/msg/modbus_state.h"
// Member `modbus_state`
#include "dsr_msgs2/msg/detail/modbus_state__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dsr_msgs2__msg__RobotState__init(message_memory);
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_fini_function(void * message_memory)
{
  dsr_msgs2__msg__RobotState__fini(message_memory);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_posj(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posj(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posj(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_posj(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posj(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_posj(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posj(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_velj(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velj(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velj(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_velj(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velj(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_velj(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velj(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__joint_abs(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_abs(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_abs(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__joint_abs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_abs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__joint_abs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_abs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__joint_err(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_err(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_err(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__joint_err(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_err(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__joint_err(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_err(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_posj(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posj(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posj(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_posj(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posj(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_posj(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posj(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_velj(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velj(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velj(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_velj(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velj(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_velj(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velj(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_posx(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posx(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posx(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_posx(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posx(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_posx(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posx(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_tool_posx(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_tool_posx(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_tool_posx(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_tool_posx(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_tool_posx(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_tool_posx(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_tool_posx(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_velx(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velx(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velx(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_velx(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velx(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_velx(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velx(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__task_err(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__task_err(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__task_err(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__task_err(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__task_err(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__task_err(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__task_err(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_velx(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velx(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velx(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_velx(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velx(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_velx(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velx(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_posx(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posx(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posx(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_posx(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posx(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_posx(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posx(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__dynamic_tor(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__dynamic_tor(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__dynamic_tor(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__dynamic_tor(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__dynamic_tor(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__dynamic_tor(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__dynamic_tor(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_jts(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_jts(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_jts(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_jts(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_jts(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_jts(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_jts(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_ejt(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ejt(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ejt(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_ejt(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ejt(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_ejt(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ejt(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_ett(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ett(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ett(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_ett(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ett(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_ett(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ett(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_bk(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bk(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bk(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_bk(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bk(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_bk(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bk(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_mc(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mc(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mc(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_mc(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mc(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_mc(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mc(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_mt(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mt(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mt(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_mt(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mt(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_mt(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mt(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_bt(
  const void * untyped_member)
{
  (void)untyped_member;
  return 5;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bt(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bt(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_bt(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bt(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_bt(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bt(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__rotation_matrix(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__rotation_matrix(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__rotation_matrix(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__rotation_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__rotation_matrix(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__rotation_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__rotation_matrix(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__rotation_matrix(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__ctrlbox_digital_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 16;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_input(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_input(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__ctrlbox_digital_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_input(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__ctrlbox_digital_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_input(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__ctrlbox_digital_output(
  const void * untyped_member)
{
  (void)untyped_member;
  return 16;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_output(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_output(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__ctrlbox_digital_output(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_output(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__ctrlbox_digital_output(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_output(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__flange_digital_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_input(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_input(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__flange_digital_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_input(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__flange_digital_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_input(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__flange_digital_output(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_output(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_output(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__flange_digital_output(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_output(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__flange_digital_output(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_output(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__modbus_state(
  const void * untyped_member)
{
  const dsr_msgs2__msg__ModbusState__Sequence * member =
    (const dsr_msgs2__msg__ModbusState__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__modbus_state(
  const void * untyped_member, size_t index)
{
  const dsr_msgs2__msg__ModbusState__Sequence * member =
    (const dsr_msgs2__msg__ModbusState__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__modbus_state(
  void * untyped_member, size_t index)
{
  dsr_msgs2__msg__ModbusState__Sequence * member =
    (dsr_msgs2__msg__ModbusState__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__modbus_state(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const dsr_msgs2__msg__ModbusState * item =
    ((const dsr_msgs2__msg__ModbusState *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__modbus_state(untyped_member, index));
  dsr_msgs2__msg__ModbusState * value =
    (dsr_msgs2__msg__ModbusState *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__modbus_state(
  void * untyped_member, size_t index, const void * untyped_value)
{
  dsr_msgs2__msg__ModbusState * item =
    ((dsr_msgs2__msg__ModbusState *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__modbus_state(untyped_member, index));
  const dsr_msgs2__msg__ModbusState * value =
    (const dsr_msgs2__msg__ModbusState *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__modbus_state(
  void * untyped_member, size_t size)
{
  dsr_msgs2__msg__ModbusState__Sequence * member =
    (dsr_msgs2__msg__ModbusState__Sequence *)(untyped_member);
  dsr_msgs2__msg__ModbusState__Sequence__fini(member);
  return dsr_msgs2__msg__ModbusState__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_actual_w2b(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_w2b(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_w2b(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_actual_w2b(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_w2b(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_actual_w2b(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_w2b(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_pos_world(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_world(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_world(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_pos_world(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_world(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_pos_world(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_world(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_current_pos_world(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_vel_world(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_world(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_world(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_vel_world(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_world(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_vel_world(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_world(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_world_ext_target_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_world_ext_target_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_world_ext_target_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_world_ext_target_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_world_ext_target_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_world_ext_target_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_world_ext_target_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_pos_world(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_world(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_world(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_pos_world(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_world(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_pos_world(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_world(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_vel_world(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_world(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_world(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_vel_world(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_world(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_vel_world(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_world(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_rotation_matrix_world(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_world(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_world(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_rotation_matrix_world(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_world(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_rotation_matrix_world(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_world(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_rotation_matrix_world(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_pos_user(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_user(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_user(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_pos_user(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_user(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_pos_user(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_user(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_current_pos_user(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_vel_user(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_user(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_user(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_vel_user(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_user(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_vel_user(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_user(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_user_ext_task_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_user_ext_task_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_user_ext_task_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_user_ext_task_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_user_ext_task_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_user_ext_task_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_user_ext_task_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_pos_user(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_user(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_user(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_pos_user(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_user(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_pos_user(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_user(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_vel_user(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_user(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_user(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_vel_user(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_user(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_vel_user(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_user(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_rotation_matrix_user(
  const void * untyped_member)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return member->size;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_user(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float64MultiArray__Sequence * member =
    (const std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_user(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  return &member->data[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_rotation_matrix_user(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const std_msgs__msg__Float64MultiArray * item =
    ((const std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_user(untyped_member, index));
  std_msgs__msg__Float64MultiArray * value =
    (std_msgs__msg__Float64MultiArray *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_rotation_matrix_user(
  void * untyped_member, size_t index, const void * untyped_value)
{
  std_msgs__msg__Float64MultiArray * item =
    ((std_msgs__msg__Float64MultiArray *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_user(untyped_member, index));
  const std_msgs__msg__Float64MultiArray * value =
    (const std_msgs__msg__Float64MultiArray *)(untyped_value);
  *item = *value;
}

bool dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_rotation_matrix_user(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float64MultiArray__Sequence * member =
    (std_msgs__msg__Float64MultiArray__Sequence *)(untyped_member);
  std_msgs__msg__Float64MultiArray__Sequence__fini(member);
  return std_msgs__msg__Float64MultiArray__Sequence__init(member, size);
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_actual_analog_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_analog_input(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_analog_input(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_actual_analog_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_analog_input(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_actual_analog_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_analog_input(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_switch_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 3;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_switch_input(
  const void * untyped_member, size_t index)
{
  const bool * member =
    (const bool *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_switch_input(
  void * untyped_member, size_t index)
{
  bool * member =
    (bool *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_switch_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_switch_input(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_switch_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_switch_input(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_safety_input(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_safety_input(
  const void * untyped_member, size_t index)
{
  const bool * member =
    (const bool *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_safety_input(
  void * untyped_member, size_t index)
{
  bool * member =
    (bool *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_safety_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_safety_input(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_safety_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_safety_input(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_actual_analog_input_type(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_analog_input_type(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_analog_input_type(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_actual_analog_input_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_analog_input_type(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_actual_analog_input_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_analog_input_type(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_analog_output(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_analog_output(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_analog_output(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_analog_output(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_analog_output(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_analog_output(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_analog_output(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_target_analog_output_type(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_target_analog_output_type(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_target_analog_output_type(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_target_analog_output_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_target_analog_output_type(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_target_analog_output_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_target_analog_output_type(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_encorder_strove_signal(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_strove_signal(
  const void * untyped_member, size_t index)
{
  const bool * member =
    (const bool *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_strove_signal(
  void * untyped_member, size_t index)
{
  bool * member =
    (bool *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_encorder_strove_signal(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_strove_signal(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_encorder_strove_signal(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_strove_signal(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_actual_encorder_raw_data(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_encorder_raw_data(
  const void * untyped_member, size_t index)
{
  const int8_t * member =
    (const int8_t *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_encorder_raw_data(
  void * untyped_member, size_t index)
{
  int8_t * member =
    (int8_t *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_actual_encorder_raw_data(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int8_t * item =
    ((const int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_encorder_raw_data(untyped_member, index));
  int8_t * value =
    (int8_t *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_actual_encorder_raw_data(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int8_t * item =
    ((int8_t *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_encorder_raw_data(untyped_member, index));
  const int8_t * value =
    (const int8_t *)(untyped_value);
  *item = *value;
}

size_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_encorder_reset_signal(
  const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_reset_signal(
  const void * untyped_member, size_t index)
{
  const bool * member =
    (const bool *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_reset_signal(
  void * untyped_member, size_t index)
{
  bool * member =
    (bool *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_encorder_reset_signal(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_reset_signal(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_encorder_reset_signal(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_reset_signal(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[62] = {
  {
    "robot_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, robot_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "robot_state_str",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, robot_state_str),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_space",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_space),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_posj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, current_posj),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_posj,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posj,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posj,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_posj,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_posj,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_velj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, current_velj),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_velj,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velj,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velj,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_velj,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_velj,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_abs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, joint_abs),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__joint_abs,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_abs,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_abs,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__joint_abs,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__joint_abs,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, joint_err),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__joint_err,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__joint_err,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__joint_err,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__joint_err,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__joint_err,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_posj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, target_posj),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_posj,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posj,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posj,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_posj,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_posj,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_velj",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, target_velj),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_velj,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velj,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velj,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_velj,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_velj,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_posx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, current_posx),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_posx,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_posx,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_posx,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_posx,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_posx,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_tool_posx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, current_tool_posx),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_tool_posx,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_tool_posx,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_tool_posx,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_tool_posx,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_tool_posx,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_velx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, current_velx),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__current_velx,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__current_velx,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__current_velx,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__current_velx,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__current_velx,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "task_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, task_err),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__task_err,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__task_err,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__task_err,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__task_err,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__task_err,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_velx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, target_velx),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_velx,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_velx,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_velx,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_velx,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_velx,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_posx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, target_posx),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__target_posx,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__target_posx,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__target_posx,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__target_posx,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__target_posx,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dynamic_tor",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, dynamic_tor),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__dynamic_tor,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__dynamic_tor,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__dynamic_tor,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__dynamic_tor,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__dynamic_tor,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_jts",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_jts),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_jts,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_jts,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_jts,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_jts,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_jts,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_ejt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_ejt),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_ejt,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ejt,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ejt,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_ejt,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_ejt,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_ett",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_ett),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_ett,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_ett,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_ett,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_ett,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_ett,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_bk",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_bk),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_bk,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bk,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bk,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_bk,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_bk,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_mc",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_mc),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_mc,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mc,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mc,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_mc,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_mc,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_mt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_mt),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_mt,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_mt,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_mt,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_mt,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_mt,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "solution_space",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, solution_space),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sync_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, sync_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "actual_bt",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    5,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, actual_bt),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__actual_bt,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__actual_bt,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__actual_bt,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__actual_bt,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__actual_bt,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rotation_matrix",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, rotation_matrix),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__rotation_matrix,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__rotation_matrix,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__rotation_matrix,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__rotation_matrix,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__rotation_matrix,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__rotation_matrix  // resize(index) function pointer
  },
  {
    "ctrlbox_digital_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    16,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, ctrlbox_digital_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__ctrlbox_digital_input,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__ctrlbox_digital_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__ctrlbox_digital_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ctrlbox_digital_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    16,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, ctrlbox_digital_output),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__ctrlbox_digital_output,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__ctrlbox_digital_output,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__ctrlbox_digital_output,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__ctrlbox_digital_output,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__ctrlbox_digital_output,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "flange_digital_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, flange_digital_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__flange_digital_input,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__flange_digital_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__flange_digital_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "flange_digital_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, flange_digital_output),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__flange_digital_output,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__flange_digital_output,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__flange_digital_output,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__flange_digital_output,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__flange_digital_output,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "modbus_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, modbus_state),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__modbus_state,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__modbus_state,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__modbus_state,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__modbus_state,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__modbus_state,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__modbus_state  // resize(index) function pointer
  },
  {
    "access_control",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, access_control),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "homming_completed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, homming_completed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tp_initialized",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, tp_initialized),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mastering_need",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, mastering_need),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "drl_stopped",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, drl_stopped),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "disconnected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, disconnected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_actual_w2b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_actual_w2b),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_actual_w2b,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_w2b,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_w2b,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_actual_w2b,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_actual_w2b,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_current_pos_world",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_current_pos_world),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_pos_world,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_world,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_world,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_pos_world,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_pos_world,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_current_pos_world  // resize(index) function pointer
  },
  {
    "f_current_vel_world",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_current_vel_world),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_vel_world,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_world,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_world,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_vel_world,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_vel_world,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_world_ext_target_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_world_ext_target_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_world_ext_target_torque,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_world_ext_target_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_world_ext_target_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_world_ext_target_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_world_ext_target_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_target_pos_world",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_target_pos_world),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_pos_world,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_world,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_world,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_pos_world,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_pos_world,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_target_vel_world",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_target_vel_world),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_vel_world,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_world,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_world,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_vel_world,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_vel_world,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_rotation_matrix_world",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_rotation_matrix_world),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_rotation_matrix_world,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_world,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_world,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_rotation_matrix_world,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_rotation_matrix_world,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_rotation_matrix_world  // resize(index) function pointer
  },
  {
    "i_actual_user_coord_num",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, i_actual_user_coord_num),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "i_coord_ref",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, i_coord_ref),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_current_pos_user",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_current_pos_user),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_pos_user,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_pos_user,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_pos_user,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_pos_user,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_pos_user,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_current_pos_user  // resize(index) function pointer
  },
  {
    "f_current_vel_user",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_current_vel_user),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_current_vel_user,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_current_vel_user,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_current_vel_user,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_current_vel_user,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_current_vel_user,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_user_ext_task_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_user_ext_task_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_user_ext_task_torque,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_user_ext_task_torque,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_user_ext_task_torque,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_user_ext_task_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_user_ext_task_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_target_pos_user",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_target_pos_user),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_pos_user,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_pos_user,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_pos_user,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_pos_user,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_pos_user,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_target_vel_user",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_target_vel_user),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_vel_user,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_vel_user,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_vel_user,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_vel_user,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_vel_user,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_rotation_matrix_user",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_rotation_matrix_user),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_rotation_matrix_user,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_rotation_matrix_user,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_rotation_matrix_user,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_rotation_matrix_user,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_rotation_matrix_user,  // assign(index, value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__resize_function__RobotState__f_rotation_matrix_user  // resize(index) function pointer
  },
  {
    "f_actual_analog_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_actual_analog_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_actual_analog_input,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_actual_analog_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_actual_analog_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_actual_analog_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_actual_analog_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "b_actual_switch_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    3,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, b_actual_switch_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_switch_input,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_switch_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_switch_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_switch_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_switch_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "b_actual_safety_input",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, b_actual_safety_input),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_safety_input,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_safety_input,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_safety_input,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_safety_input,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_safety_input,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "i_actual_analog_input_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, i_actual_analog_input_type),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_actual_analog_input_type,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_analog_input_type,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_analog_input_type,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_actual_analog_input_type,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_actual_analog_input_type,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "f_target_analog_output",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, f_target_analog_output),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__f_target_analog_output,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__f_target_analog_output,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__f_target_analog_output,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__f_target_analog_output,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__f_target_analog_output,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "i_target_analog_output_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, i_target_analog_output_type),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_target_analog_output_type,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_target_analog_output_type,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_target_analog_output_type,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_target_analog_output_type,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_target_analog_output_type,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "b_actual_encorder_strove_signal",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, b_actual_encorder_strove_signal),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_encorder_strove_signal,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_strove_signal,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_strove_signal,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_encorder_strove_signal,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_encorder_strove_signal,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "i_actual_encorder_raw_data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, i_actual_encorder_raw_data),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__i_actual_encorder_raw_data,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__i_actual_encorder_raw_data,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__i_actual_encorder_raw_data,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__i_actual_encorder_raw_data,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__i_actual_encorder_raw_data,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "b_actual_encorder_reset_signal",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__msg__RobotState, b_actual_encorder_reset_signal),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__size_function__RobotState__b_actual_encorder_reset_signal,  // size() function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_const_function__RobotState__b_actual_encorder_reset_signal,  // get_const(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__get_function__RobotState__b_actual_encorder_reset_signal,  // get(index) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__fetch_function__RobotState__b_actual_encorder_reset_signal,  // fetch(index, &value) function pointer
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__assign_function__RobotState__b_actual_encorder_reset_signal,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_members = {
  "dsr_msgs2__msg",  // message namespace
  "RobotState",  // message name
  62,  // number of fields
  sizeof(dsr_msgs2__msg__RobotState),
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array,  // message members
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_init_function,  // function to initialize message memory (memory has to be allocated)
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_type_support_handle = {
  0,
  &dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dsr_msgs2
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, msg, RobotState)() {
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[26].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[31].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, msg, ModbusState)();
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[39].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[44].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[47].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_member_array[52].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float64MultiArray)();
  if (!dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_type_support_handle.typesupport_identifier) {
    dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dsr_msgs2__msg__RobotState__rosidl_typesupport_introspection_c__RobotState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
