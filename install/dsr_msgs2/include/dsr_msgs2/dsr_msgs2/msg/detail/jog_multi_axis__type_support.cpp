// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/JogMultiAxis.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "dsr_msgs2/msg/detail/jog_multi_axis__struct.hpp"
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

void JogMultiAxis_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) dsr_msgs2::msg::JogMultiAxis(_init);
}

void JogMultiAxis_fini_function(void * message_memory)
{
  auto typed_message = static_cast<dsr_msgs2::msg::JogMultiAxis *>(message_memory);
  typed_message->~JogMultiAxis();
}

size_t size_function__JogMultiAxis__jog_axis(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__JogMultiAxis__jog_axis(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__JogMultiAxis__jog_axis(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__JogMultiAxis__jog_axis(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__JogMultiAxis__jog_axis(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__JogMultiAxis__jog_axis(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__JogMultiAxis__jog_axis(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JogMultiAxis_message_member_array[3] = {
  {
    "jog_axis",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::JogMultiAxis, jog_axis),  // bytes offset in struct
    nullptr,  // default value
    size_function__JogMultiAxis__jog_axis,  // size() function pointer
    get_const_function__JogMultiAxis__jog_axis,  // get_const(index) function pointer
    get_function__JogMultiAxis__jog_axis,  // get(index) function pointer
    fetch_function__JogMultiAxis__jog_axis,  // fetch(index, &value) function pointer
    assign_function__JogMultiAxis__jog_axis,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "move_reference",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::JogMultiAxis, move_reference),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "speed",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::msg::JogMultiAxis, speed),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JogMultiAxis_message_members = {
  "dsr_msgs2::msg",  // message namespace
  "JogMultiAxis",  // message name
  3,  // number of fields
  sizeof(dsr_msgs2::msg::JogMultiAxis),
  JogMultiAxis_message_member_array,  // message members
  JogMultiAxis_init_function,  // function to initialize message memory (memory has to be allocated)
  JogMultiAxis_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JogMultiAxis_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JogMultiAxis_message_members,
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
get_message_type_support_handle<dsr_msgs2::msg::JogMultiAxis>()
{
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::JogMultiAxis_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, msg, JogMultiAxis)() {
  return &::dsr_msgs2::msg::rosidl_typesupport_introspection_cpp::JogMultiAxis_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
