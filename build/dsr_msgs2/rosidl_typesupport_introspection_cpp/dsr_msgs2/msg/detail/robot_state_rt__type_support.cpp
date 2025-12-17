// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "dsr_msgs2/msg/detail/robot_state_rt__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace dsr_msgs2
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void RobotStateRt_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) dsr_msgs2::msg::RobotStateRt(_init);
}

void RobotStateRt_fini_function(void * message_memory)
{
  auto typed_message = static_cast<dsr_msgs2::msg::RobotStateRt *>(message_memory);
  typed_message->~RobotStateRt();
}

size_t size_function__RobotStateRt__actual_joint_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_joint_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_joint_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_joint_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_joint_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_joint_position_abs(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_joint_position_abs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_joint_position_abs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_joint_position_abs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_joint_position_abs(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_joint_position_abs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_joint_position_abs(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_joint_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_joint_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_joint_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_joint_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_joint_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_joint_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_joint_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_joint_velocity_abs(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_joint_velocity_abs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_joint_velocity_abs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_joint_velocity_abs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_joint_velocity_abs(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_joint_velocity_abs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_joint_velocity_abs(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_tcp_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_tcp_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_tcp_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_tcp_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_tcp_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_tcp_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_tcp_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_tcp_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_tcp_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_tcp_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_tcp_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_tcp_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_flange_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_flange_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_flange_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_flange_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_flange_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_flange_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_flange_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_flange_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_flange_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_flange_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_flange_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_flange_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_flange_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_flange_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_motor_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_motor_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_motor_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_motor_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_motor_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_motor_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_motor_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__actual_joint_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__actual_joint_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__actual_joint_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__actual_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__actual_joint_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__actual_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__actual_joint_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__raw_joint_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__raw_joint_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__raw_joint_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__raw_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__raw_joint_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__raw_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__raw_joint_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__raw_force_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__raw_force_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__raw_force_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__raw_force_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__raw_force_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__raw_force_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__raw_force_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__external_joint_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__external_joint_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__external_joint_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__external_joint_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__external_joint_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__external_joint_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__external_joint_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__external_tcp_force(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__external_tcp_force(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__external_tcp_force(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__external_tcp_force(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__external_tcp_force(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__external_tcp_force(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__external_tcp_force(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_joint_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_joint_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_joint_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_joint_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_joint_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_joint_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_joint_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_joint_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_joint_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_joint_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_joint_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_joint_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_joint_acceleration(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_joint_acceleration(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_joint_acceleration(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_joint_acceleration(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_joint_acceleration(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_joint_acceleration(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_joint_acceleration(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_motor_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_motor_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_motor_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_motor_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_motor_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_motor_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_motor_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_tcp_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_tcp_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_tcp_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_tcp_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_tcp_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__target_tcp_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__target_tcp_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__target_tcp_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__target_tcp_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__target_tcp_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__target_tcp_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__target_tcp_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__jacobian_matrix(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRt__jacobian_matrix(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__jacobian_matrix(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__jacobian_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(
    get_const_function__RobotStateRt__jacobian_matrix(untyped_member, index));
  auto & value = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__jacobian_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(
    get_function__RobotStateRt__jacobian_matrix(untyped_member, index));
  const auto & value = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRt__jacobian_matrix(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRt__gravity_torque(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__gravity_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__gravity_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__gravity_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__gravity_torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__gravity_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__gravity_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__coriolis_matrix(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRt__coriolis_matrix(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__coriolis_matrix(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__coriolis_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(
    get_const_function__RobotStateRt__coriolis_matrix(untyped_member, index));
  auto & value = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__coriolis_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(
    get_function__RobotStateRt__coriolis_matrix(untyped_member, index));
  const auto & value = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRt__coriolis_matrix(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRt__mass_matrix(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRt__mass_matrix(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__mass_matrix(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__mass_matrix(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(
    get_const_function__RobotStateRt__mass_matrix(untyped_member, index));
  auto & value = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__mass_matrix(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<std_msgs::msg::Float64MultiArray *>(
    get_function__RobotStateRt__mass_matrix(untyped_member, index));
  const auto & value = *reinterpret_cast<const std_msgs::msg::Float64MultiArray *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRt__mass_matrix(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std_msgs::msg::Float64MultiArray> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRt__joint_temperature(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__joint_temperature(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__joint_temperature(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__joint_temperature(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__joint_temperature(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__joint_temperature(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__joint_temperature(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__controller_analog_input_type(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__controller_analog_input_type(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__controller_analog_input_type(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__controller_analog_input_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__RobotStateRt__controller_analog_input_type(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__controller_analog_input_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__RobotStateRt__controller_analog_input_type(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__controller_analog_input(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__controller_analog_input(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__controller_analog_input(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__controller_analog_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__controller_analog_input(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__controller_analog_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__controller_analog_input(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__controller_analog_output_type(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__controller_analog_output_type(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__controller_analog_output_type(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__controller_analog_output_type(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__RobotStateRt__controller_analog_output_type(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__controller_analog_output_type(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__RobotStateRt__controller_analog_output_type(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__controller_analog_output(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__controller_analog_output(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__controller_analog_output(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__controller_analog_output(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__controller_analog_output(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__controller_analog_output(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__controller_analog_output(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__flange_analog_input(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__RobotStateRt__flange_analog_input(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__flange_analog_input(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__flange_analog_input(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__flange_analog_input(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__flange_analog_input(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__flange_analog_input(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__external_encoder_strobe_count(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__external_encoder_strobe_count(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__external_encoder_strobe_count(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__external_encoder_strobe_count(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__RobotStateRt__external_encoder_strobe_count(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__external_encoder_strobe_count(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__RobotStateRt__external_encoder_strobe_count(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__external_encoder_count(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__RobotStateRt__external_encoder_count(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint16_t, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__external_encoder_count(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint16_t, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__external_encoder_count(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint16_t *>(
    get_const_function__RobotStateRt__external_encoder_count(untyped_member, index));
  auto & value = *reinterpret_cast<uint16_t *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__external_encoder_count(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint16_t *>(
    get_function__RobotStateRt__external_encoder_count(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint16_t *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__goal_joint_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__goal_joint_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__goal_joint_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__goal_joint_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__goal_joint_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__goal_joint_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__goal_joint_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__goal_tcp_position(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotStateRt__goal_tcp_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__goal_tcp_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__goal_tcp_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRt__goal_tcp_position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__goal_tcp_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRt__goal_tcp_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotStateRt__reserved(const void * untyped_member)
{
  (void)untyped_member;
  return 256;
}

const void * get_const_function__RobotStateRt__reserved(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 256> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRt__reserved(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 256> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRt__reserved(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__RobotStateRt__reserved(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRt__reserved(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__RobotStateRt__reserved(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RobotStateRt_message_member_array[46] = {
  {
    "time_stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, time_stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_joint_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_joint_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_joint_position,  // size() function pointer
    get_const_function__RobotStateRt__actual_joint_position,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_joint_position,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_joint_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_joint_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_joint_position_abs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_joint_position_abs),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_joint_position_abs,  // size() function pointer
    get_const_function__RobotStateRt__actual_joint_position_abs,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_joint_position_abs,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_joint_position_abs,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_joint_position_abs,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_joint_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_joint_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_joint_velocity,  // size() function pointer
    get_const_function__RobotStateRt__actual_joint_velocity,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_joint_velocity,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_joint_velocity,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_joint_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_joint_velocity_abs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_joint_velocity_abs),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_joint_velocity_abs,  // size() function pointer
    get_const_function__RobotStateRt__actual_joint_velocity_abs,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_joint_velocity_abs,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_joint_velocity_abs,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_joint_velocity_abs,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_tcp_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_tcp_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_tcp_position,  // size() function pointer
    get_const_function__RobotStateRt__actual_tcp_position,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_tcp_position,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_tcp_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_tcp_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_tcp_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_tcp_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_tcp_velocity,  // size() function pointer
    get_const_function__RobotStateRt__actual_tcp_velocity,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_tcp_velocity,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_tcp_velocity,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_tcp_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_flange_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_flange_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_flange_position,  // size() function pointer
    get_const_function__RobotStateRt__actual_flange_position,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_flange_position,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_flange_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_flange_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_flange_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_flange_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_flange_velocity,  // size() function pointer
    get_const_function__RobotStateRt__actual_flange_velocity,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_flange_velocity,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_flange_velocity,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_flange_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_motor_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_motor_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_motor_torque,  // size() function pointer
    get_const_function__RobotStateRt__actual_motor_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_motor_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_motor_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_motor_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "actual_joint_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, actual_joint_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__actual_joint_torque,  // size() function pointer
    get_const_function__RobotStateRt__actual_joint_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__actual_joint_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__actual_joint_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__actual_joint_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_joint_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, raw_joint_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__raw_joint_torque,  // size() function pointer
    get_const_function__RobotStateRt__raw_joint_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__raw_joint_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__raw_joint_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__raw_joint_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "raw_force_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, raw_force_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__raw_force_torque,  // size() function pointer
    get_const_function__RobotStateRt__raw_force_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__raw_force_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__raw_force_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__raw_force_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "external_joint_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, external_joint_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__external_joint_torque,  // size() function pointer
    get_const_function__RobotStateRt__external_joint_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__external_joint_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__external_joint_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__external_joint_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "external_tcp_force",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, external_tcp_force),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__external_tcp_force,  // size() function pointer
    get_const_function__RobotStateRt__external_tcp_force,  // get_const(index) function pointer
    get_function__RobotStateRt__external_tcp_force,  // get(index) function pointer
    fetch_function__RobotStateRt__external_tcp_force,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__external_tcp_force,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_joint_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_joint_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_joint_position,  // size() function pointer
    get_const_function__RobotStateRt__target_joint_position,  // get_const(index) function pointer
    get_function__RobotStateRt__target_joint_position,  // get(index) function pointer
    fetch_function__RobotStateRt__target_joint_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_joint_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_joint_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_joint_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_joint_velocity,  // size() function pointer
    get_const_function__RobotStateRt__target_joint_velocity,  // get_const(index) function pointer
    get_function__RobotStateRt__target_joint_velocity,  // get(index) function pointer
    fetch_function__RobotStateRt__target_joint_velocity,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_joint_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_joint_acceleration",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_joint_acceleration),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_joint_acceleration,  // size() function pointer
    get_const_function__RobotStateRt__target_joint_acceleration,  // get_const(index) function pointer
    get_function__RobotStateRt__target_joint_acceleration,  // get(index) function pointer
    fetch_function__RobotStateRt__target_joint_acceleration,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_joint_acceleration,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_motor_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_motor_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_motor_torque,  // size() function pointer
    get_const_function__RobotStateRt__target_motor_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__target_motor_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__target_motor_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_motor_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_tcp_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_tcp_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_tcp_position,  // size() function pointer
    get_const_function__RobotStateRt__target_tcp_position,  // get_const(index) function pointer
    get_function__RobotStateRt__target_tcp_position,  // get(index) function pointer
    fetch_function__RobotStateRt__target_tcp_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_tcp_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_tcp_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, target_tcp_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__target_tcp_velocity,  // size() function pointer
    get_const_function__RobotStateRt__target_tcp_velocity,  // get_const(index) function pointer
    get_function__RobotStateRt__target_tcp_velocity,  // get(index) function pointer
    fetch_function__RobotStateRt__target_tcp_velocity,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__target_tcp_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "jacobian_matrix",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Float64MultiArray>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, jacobian_matrix),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__jacobian_matrix,  // size() function pointer
    get_const_function__RobotStateRt__jacobian_matrix,  // get_const(index) function pointer
    get_function__RobotStateRt__jacobian_matrix,  // get(index) function pointer
    fetch_function__RobotStateRt__jacobian_matrix,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__jacobian_matrix,  // assign(index, value) function pointer
    resize_function__RobotStateRt__jacobian_matrix  // resize(index) function pointer
  },
  {
    "gravity_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, gravity_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__gravity_torque,  // size() function pointer
    get_const_function__RobotStateRt__gravity_torque,  // get_const(index) function pointer
    get_function__RobotStateRt__gravity_torque,  // get(index) function pointer
    fetch_function__RobotStateRt__gravity_torque,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__gravity_torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "coriolis_matrix",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Float64MultiArray>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, coriolis_matrix),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__coriolis_matrix,  // size() function pointer
    get_const_function__RobotStateRt__coriolis_matrix,  // get_const(index) function pointer
    get_function__RobotStateRt__coriolis_matrix,  // get(index) function pointer
    fetch_function__RobotStateRt__coriolis_matrix,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__coriolis_matrix,  // assign(index, value) function pointer
    resize_function__RobotStateRt__coriolis_matrix  // resize(index) function pointer
  },
  {
    "mass_matrix",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Float64MultiArray>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, mass_matrix),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__mass_matrix,  // size() function pointer
    get_const_function__RobotStateRt__mass_matrix,  // get_const(index) function pointer
    get_function__RobotStateRt__mass_matrix,  // get(index) function pointer
    fetch_function__RobotStateRt__mass_matrix,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__mass_matrix,  // assign(index, value) function pointer
    resize_function__RobotStateRt__mass_matrix  // resize(index) function pointer
  },
  {
    "solution_space",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, solution_space),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "singularity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, singularity),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "operation_speed_rate",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, operation_speed_rate),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joint_temperature",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, joint_temperature),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__joint_temperature,  // size() function pointer
    get_const_function__RobotStateRt__joint_temperature,  // get_const(index) function pointer
    get_function__RobotStateRt__joint_temperature,  // get(index) function pointer
    fetch_function__RobotStateRt__joint_temperature,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__joint_temperature,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_digital_input",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_digital_input),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_digital_output",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_digital_output),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_analog_input_type",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_analog_input_type),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__controller_analog_input_type,  // size() function pointer
    get_const_function__RobotStateRt__controller_analog_input_type,  // get_const(index) function pointer
    get_function__RobotStateRt__controller_analog_input_type,  // get(index) function pointer
    fetch_function__RobotStateRt__controller_analog_input_type,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__controller_analog_input_type,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_analog_input",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_analog_input),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__controller_analog_input,  // size() function pointer
    get_const_function__RobotStateRt__controller_analog_input,  // get_const(index) function pointer
    get_function__RobotStateRt__controller_analog_input,  // get(index) function pointer
    fetch_function__RobotStateRt__controller_analog_input,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__controller_analog_input,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_analog_output_type",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_analog_output_type),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__controller_analog_output_type,  // size() function pointer
    get_const_function__RobotStateRt__controller_analog_output_type,  // get_const(index) function pointer
    get_function__RobotStateRt__controller_analog_output_type,  // get(index) function pointer
    fetch_function__RobotStateRt__controller_analog_output_type,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__controller_analog_output_type,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "controller_analog_output",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, controller_analog_output),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__controller_analog_output,  // size() function pointer
    get_const_function__RobotStateRt__controller_analog_output,  // get_const(index) function pointer
    get_function__RobotStateRt__controller_analog_output,  // get(index) function pointer
    fetch_function__RobotStateRt__controller_analog_output,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__controller_analog_output,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "flange_digital_input",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, flange_digital_input),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "flange_digital_output",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, flange_digital_output),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "flange_analog_input",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, flange_analog_input),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__flange_analog_input,  // size() function pointer
    get_const_function__RobotStateRt__flange_analog_input,  // get_const(index) function pointer
    get_function__RobotStateRt__flange_analog_input,  // get(index) function pointer
    fetch_function__RobotStateRt__flange_analog_input,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__flange_analog_input,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "external_encoder_strobe_count",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, external_encoder_strobe_count),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__external_encoder_strobe_count,  // size() function pointer
    get_const_function__RobotStateRt__external_encoder_strobe_count,  // get_const(index) function pointer
    get_function__RobotStateRt__external_encoder_strobe_count,  // get(index) function pointer
    fetch_function__RobotStateRt__external_encoder_strobe_count,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__external_encoder_strobe_count,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "external_encoder_count",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, external_encoder_count),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__external_encoder_count,  // size() function pointer
    get_const_function__RobotStateRt__external_encoder_count,  // get_const(index) function pointer
    get_function__RobotStateRt__external_encoder_count,  // get(index) function pointer
    fetch_function__RobotStateRt__external_encoder_count,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__external_encoder_count,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "goal_joint_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, goal_joint_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__goal_joint_position,  // size() function pointer
    get_const_function__RobotStateRt__goal_joint_position,  // get_const(index) function pointer
    get_function__RobotStateRt__goal_joint_position,  // get(index) function pointer
    fetch_function__RobotStateRt__goal_joint_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__goal_joint_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "goal_tcp_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, goal_tcp_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__goal_tcp_position,  // size() function pointer
    get_const_function__RobotStateRt__goal_tcp_position,  // get_const(index) function pointer
    get_function__RobotStateRt__goal_tcp_position,  // get(index) function pointer
    fetch_function__RobotStateRt__goal_tcp_position,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__goal_tcp_position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "robot_mode",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, robot_mode),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "robot_state",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, robot_state),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "control_mode",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, control_mode),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "reserved",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    256,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::RobotStateRt, reserved),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRt__reserved,  // size() function pointer
    get_const_function__RobotStateRt__reserved,  // get_const(index) function pointer
    get_function__RobotStateRt__reserved,  // get(index) function pointer
    fetch_function__RobotStateRt__reserved,  // fetch(index, &value) function pointer
    assign_function__RobotStateRt__reserved,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RobotStateRt_message_members = {
  "dsr_msgs2::msg",  // message namespace
  "RobotStateRt",  // message name
  46,  // number of fields
  sizeof(dsr_msgs2::msg::RobotStateRt),
  RobotStateRt_message_member_array,  // message members
  RobotStateRt_init_function,  // function to initialize message memory (memory has to be allocated)
  RobotStateRt_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RobotStateRt_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RobotStateRt_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace dsr_msgs2


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<dsr_msgs2::msg::RobotStateRt>()
{
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::RobotStateRt_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, msg, RobotStateRt)() {
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::RobotStateRt_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
