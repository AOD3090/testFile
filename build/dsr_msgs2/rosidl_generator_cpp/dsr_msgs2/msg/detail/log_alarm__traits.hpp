// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:msg/LogAlarm.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__LOG_ALARM__TRAITS_HPP_
#define DSR_MSGS2__MSG__DETAIL__LOG_ALARM__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/msg/detail/log_alarm__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace msg
{

inline void to_flow_style_yaml(
  const LogAlarm & msg,
  std::ostream & out)
{
  out << "{";
  // member: level
  {
    out << "level: ";
    rosidl_generator_traits::value_to_yaml(msg.level, out);
    out << ", ";
  }

  // member: group
  {
    out << "group: ";
    rosidl_generator_traits::value_to_yaml(msg.group, out);
    out << ", ";
  }

  // member: index
  {
    out << "index: ";
    rosidl_generator_traits::value_to_yaml(msg.index, out);
    out << ", ";
  }

  // member: param
  {
    if (msg.param.size() == 0) {
      out << "param: []";
    } else {
      out << "param: [";
      size_t pending_items = msg.param.size();
      for (auto item : msg.param) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LogAlarm & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: level
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "level: ";
    rosidl_generator_traits::value_to_yaml(msg.level, out);
    out << "\n";
  }

  // member: group
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "group: ";
    rosidl_generator_traits::value_to_yaml(msg.group, out);
    out << "\n";
  }

  // member: index
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "index: ";
    rosidl_generator_traits::value_to_yaml(msg.index, out);
    out << "\n";
  }

  // member: param
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.param.size() == 0) {
      out << "param: []\n";
    } else {
      out << "param:\n";
      for (auto item : msg.param) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LogAlarm & msg, bool use_flow_style = false)
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
  const dsr_msgs2::msg::LogAlarm & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::msg::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::msg::LogAlarm & msg)
{
  return dsr_msgs2::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::msg::LogAlarm>()
{
  return "dsr_msgs2::msg::LogAlarm";
}

template<>
inline const char * name<dsr_msgs2::msg::LogAlarm>()
{
  return "dsr_msgs2/msg/LogAlarm";
}

template<>
struct has_fixed_size<dsr_msgs2::msg::LogAlarm>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::msg::LogAlarm>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::msg::LogAlarm>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__MSG__DETAIL__LOG_ALARM__TRAITS_HPP_
