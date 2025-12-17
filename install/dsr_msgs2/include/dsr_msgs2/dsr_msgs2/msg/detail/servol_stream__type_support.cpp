// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/ServolStream.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "dsr_msgs2/msg/detail/servol_stream__struct.hpp"
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

void ServolStream_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) dsr_msgs2::msg::ServolStream(_init);
}

void ServolStream_fini_function(void * message_memory)
{
  auto typed_message = static_cast<dsr_msgs2::msg::ServolStream *>(message_memory);
  typed_message->~ServolStream();
}

size_t size_function__ServolStream__pos(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__ServolStream__pos(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__ServolStream__pos(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__ServolStream__pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ServolStream__pos(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ServolStream__pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ServolStream__pos(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__ServolStream__vel(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__ServolStream__vel(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__ServolStream__vel(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__ServolStream__vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ServolStream__vel(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ServolStream__vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ServolStream__vel(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__ServolStream__acc(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__ServolStream__acc(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__ServolStream__acc(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__ServolStream__acc(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ServolStream__acc(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ServolStream__acc(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ServolStream__acc(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ServolStream_message_member_array[4] = {
  {
    "pos",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::ServolStream, pos),  // bytes offset in struct
    nullptr,  // default value
    size_function__ServolStream__pos,  // size() function pointer
    get_const_function__ServolStream__pos,  // get_const(index) function pointer
    get_function__ServolStream__pos,  // get(index) function pointer
    fetch_function__ServolStream__pos,  // fetch(index, &value) function pointer
    assign_function__ServolStream__pos,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vel",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::ServolStream, vel),  // bytes offset in struct
    nullptr,  // default value
    size_function__ServolStream__vel,  // size() function pointer
    get_const_function__ServolStream__vel,  // get_const(index) function pointer
    get_function__ServolStream__vel,  // get(index) function pointer
    fetch_function__ServolStream__vel,  // fetch(index, &value) function pointer
    assign_function__ServolStream__vel,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "acc",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::ServolStream, acc),  // bytes offset in struct
    nullptr,  // default value
    size_function__ServolStream__acc,  // size() function pointer
    get_const_function__ServolStream__acc,  // get_const(index) function pointer
    get_function__ServolStream__acc,  // get(index) function pointer
    fetch_function__ServolStream__acc,  // fetch(index, &value) function pointer
    assign_function__ServolStream__acc,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::ServolStream, time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ServolStream_message_members = {
  "dsr_msgs2::msg",  // message namespace
  "ServolStream",  // message name
  4,  // number of fields
  sizeof(dsr_msgs2::msg::ServolStream),
  ServolStream_message_member_array,  // message members
  ServolStream_init_function,  // function to initialize message memory (memory has to be allocated)
  ServolStream_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ServolStream_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ServolStream_message_members,
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
get_message_type_support_handle<dsr_msgs2::msg::ServolStream>()
{
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::ServolStream_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, msg, ServolStream)() {
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::ServolStream_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
