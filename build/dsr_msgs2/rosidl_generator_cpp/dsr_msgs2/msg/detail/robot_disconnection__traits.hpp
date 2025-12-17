// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:msg/RobotDisconnection.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__TRAITS_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/msg/detail/robot_disconnection__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotDisconnection & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotDisconnection & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotDisconnection & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace dsr_msgs2

namespace rosidl_generator_traits
{

[[deprecated("use dsr_msgs2::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dsr_msgs2::msg::RobotDisconnection & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::msg::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::msg::RobotDisconnection & msg)
{
  return dsr_msgs2::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::msg::RobotDisconnection>()
{
  return "dsr_msgs2::msg::RobotDisconnection";
}

template<>
inline const char * name<dsr_msgs2::msg::RobotDisconnection>()
{
  return "dsr_msgs2/msg/RobotDisconnection";
}

template<>
struct has_fixed_size<dsr_msgs2::msg::RobotDisconnection>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::msg::RobotDisconnection>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::msg::RobotDisconnection>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__TRAITS_HPP_
