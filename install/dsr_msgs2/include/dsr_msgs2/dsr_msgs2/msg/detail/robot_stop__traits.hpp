// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:msg/RobotStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__TRAITS_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/msg/detail/robot_stop__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotStop & msg,
  std::ostream & out)
{
  out << "{";
  // member: stop_mode
  {
    out << "stop_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.stop_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotStop & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stop_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stop_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.stop_mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotStop & msg, bool use_flow_style = false)
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
  const dsr_msgs2::msg::RobotStop & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::msg::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::msg::RobotStop & msg)
{
  return dsr_msgs2::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::msg::RobotStop>()
{
  return "dsr_msgs2::msg::RobotStop";
}

template<>
inline const char * name<dsr_msgs2::msg::RobotStop>()
{
  return "dsr_msgs2/msg/RobotStop";
}

template<>
struct has_fixed_size<dsr_msgs2::msg::RobotStop>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::msg::RobotStop>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::msg::RobotStop>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__TRAITS_HPP_
