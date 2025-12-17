// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/MoveStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_STOP__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_STOP__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/move_stop__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveStop_Request & msg,
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
  const MoveStop_Request & msg,
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

inline std::string to_yaml(const MoveStop_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace dsr_msgs2

namespace rosidl_generator_traits
{

[[deprecated("use dsr_msgs2::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dsr_msgs2::srv::MoveStop_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveStop_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveStop_Request>()
{
  return "dsr_msgs2::srv::MoveStop_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveStop_Request>()
{
  return "dsr_msgs2/srv/MoveStop_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveStop_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveStop_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::MoveStop_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveStop_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MoveStop_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MoveStop_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace dsr_msgs2

namespace rosidl_generator_traits
{

[[deprecated("use dsr_msgs2::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dsr_msgs2::srv::MoveStop_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveStop_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveStop_Response>()
{
  return "dsr_msgs2::srv::MoveStop_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveStop_Response>()
{
  return "dsr_msgs2/srv/MoveStop_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveStop_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveStop_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::MoveStop_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::MoveStop>()
{
  return "dsr_msgs2::srv::MoveStop";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveStop>()
{
  return "dsr_msgs2/srv/MoveStop";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveStop>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::MoveStop_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::MoveStop_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveStop>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::MoveStop_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::MoveStop_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::MoveStop>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::MoveStop_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::MoveStop_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_STOP__TRAITS_HPP_
