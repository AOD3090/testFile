// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/ReadDataRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/read_data_rt__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ReadDataRt_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ReadDataRt_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ReadDataRt_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ReadDataRt_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ReadDataRt_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ReadDataRt_Request>()
{
  return "dsr_msgs2::srv::ReadDataRt_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::ReadDataRt_Request>()
{
  return "dsr_msgs2/srv/ReadDataRt_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ReadDataRt_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ReadDataRt_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::ReadDataRt_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'data'
#include "dsr_msgs2/msg/detail/robot_state_rt__traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ReadDataRt_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: data
  {
    out << "data: ";
    to_flow_style_yaml(msg.data, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ReadDataRt_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "data:\n";
    to_block_style_yaml(msg.data, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ReadDataRt_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ReadDataRt_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ReadDataRt_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ReadDataRt_Response>()
{
  return "dsr_msgs2::srv::ReadDataRt_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::ReadDataRt_Response>()
{
  return "dsr_msgs2/srv/ReadDataRt_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ReadDataRt_Response>
  : std::integral_constant<bool, has_fixed_size<dsr_msgs2::msg::RobotStateRt>::value> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ReadDataRt_Response>
  : std::integral_constant<bool, has_bounded_size<dsr_msgs2::msg::RobotStateRt>::value> {};

template<>
struct is_message<dsr_msgs2::srv::ReadDataRt_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::ReadDataRt>()
{
  return "dsr_msgs2::srv::ReadDataRt";
}

template<>
inline const char * name<dsr_msgs2::srv::ReadDataRt>()
{
  return "dsr_msgs2/srv/ReadDataRt";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ReadDataRt>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::ReadDataRt_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::ReadDataRt_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::ReadDataRt>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::ReadDataRt_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::ReadDataRt_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::ReadDataRt>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::ReadDataRt_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::ReadDataRt_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__TRAITS_HPP_
