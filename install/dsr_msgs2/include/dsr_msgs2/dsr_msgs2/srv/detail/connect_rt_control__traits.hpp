// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/ConnectRtControl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/connect_rt_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConnectRtControl_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: ip_address
  {
    out << "ip_address: ";
    rosidl_generator_traits::value_to_yaml(msg.ip_address, out);
    out << ", ";
  }

  // member: port
  {
    out << "port: ";
    rosidl_generator_traits::value_to_yaml(msg.port, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ConnectRtControl_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: ip_address
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ip_address: ";
    rosidl_generator_traits::value_to_yaml(msg.ip_address, out);
    out << "\n";
  }

  // member: port
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "port: ";
    rosidl_generator_traits::value_to_yaml(msg.port, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConnectRtControl_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ConnectRtControl_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ConnectRtControl_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ConnectRtControl_Request>()
{
  return "dsr_msgs2::srv::ConnectRtControl_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::ConnectRtControl_Request>()
{
  return "dsr_msgs2/srv/ConnectRtControl_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConnectRtControl_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConnectRtControl_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::srv::ConnectRtControl_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConnectRtControl_Response & msg,
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
  const ConnectRtControl_Response & msg,
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

inline std::string to_yaml(const ConnectRtControl_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ConnectRtControl_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ConnectRtControl_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ConnectRtControl_Response>()
{
  return "dsr_msgs2::srv::ConnectRtControl_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::ConnectRtControl_Response>()
{
  return "dsr_msgs2/srv/ConnectRtControl_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConnectRtControl_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConnectRtControl_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::ConnectRtControl_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::ConnectRtControl>()
{
  return "dsr_msgs2::srv::ConnectRtControl";
}

template<>
inline const char * name<dsr_msgs2::srv::ConnectRtControl>()
{
  return "dsr_msgs2/srv/ConnectRtControl";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConnectRtControl>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::ConnectRtControl_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::ConnectRtControl_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConnectRtControl>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::ConnectRtControl_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::ConnectRtControl_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::ConnectRtControl>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::ConnectRtControl_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::ConnectRtControl_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__TRAITS_HPP_
