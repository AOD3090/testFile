// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/GetCurrentPosx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/get_current_posx__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetCurrentPosx_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: ref
  {
    out << "ref: ";
    rosidl_generator_traits::value_to_yaml(msg.ref, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetCurrentPosx_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: ref
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref: ";
    rosidl_generator_traits::value_to_yaml(msg.ref, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetCurrentPosx_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::GetCurrentPosx_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::GetCurrentPosx_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::GetCurrentPosx_Request>()
{
  return "dsr_msgs2::srv::GetCurrentPosx_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::GetCurrentPosx_Request>()
{
  return "dsr_msgs2/srv/GetCurrentPosx_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::GetCurrentPosx_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::GetCurrentPosx_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::GetCurrentPosx_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'task_pos_info'
#include "std_msgs/msg/detail/float64_multi_array__traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetCurrentPosx_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: task_pos_info
  {
    if (msg.task_pos_info.size() == 0) {
      out << "task_pos_info: []";
    } else {
      out << "task_pos_info: [";
      size_t pending_items = msg.task_pos_info.size();
      for (auto item : msg.task_pos_info) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetCurrentPosx_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: task_pos_info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.task_pos_info.size() == 0) {
      out << "task_pos_info: []\n";
    } else {
      out << "task_pos_info:\n";
      for (auto item : msg.task_pos_info) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

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

inline std::string to_yaml(const GetCurrentPosx_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::GetCurrentPosx_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::GetCurrentPosx_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::GetCurrentPosx_Response>()
{
  return "dsr_msgs2::srv::GetCurrentPosx_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::GetCurrentPosx_Response>()
{
  return "dsr_msgs2/srv/GetCurrentPosx_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::GetCurrentPosx_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::GetCurrentPosx_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::srv::GetCurrentPosx_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::GetCurrentPosx>()
{
  return "dsr_msgs2::srv::GetCurrentPosx";
}

template<>
inline const char * name<dsr_msgs2::srv::GetCurrentPosx>()
{
  return "dsr_msgs2/srv/GetCurrentPosx";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::GetCurrentPosx>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::GetCurrentPosx_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::GetCurrentPosx_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::GetCurrentPosx>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::GetCurrentPosx_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::GetCurrentPosx_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::GetCurrentPosx>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::GetCurrentPosx_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::GetCurrentPosx_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__TRAITS_HPP_
