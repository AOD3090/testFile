// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/ConfigCreateTool.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/config_create_tool__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConfigCreateTool_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: weight
  {
    out << "weight: ";
    rosidl_generator_traits::value_to_yaml(msg.weight, out);
    out << ", ";
  }

  // member: cog
  {
    if (msg.cog.size() == 0) {
      out << "cog: []";
    } else {
      out << "cog: [";
      size_t pending_items = msg.cog.size();
      for (auto item : msg.cog) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: inertia
  {
    if (msg.inertia.size() == 0) {
      out << "inertia: []";
    } else {
      out << "inertia: [";
      size_t pending_items = msg.inertia.size();
      for (auto item : msg.inertia) {
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
  const ConfigCreateTool_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: weight
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "weight: ";
    rosidl_generator_traits::value_to_yaml(msg.weight, out);
    out << "\n";
  }

  // member: cog
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.cog.size() == 0) {
      out << "cog: []\n";
    } else {
      out << "cog:\n";
      for (auto item : msg.cog) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: inertia
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.inertia.size() == 0) {
      out << "inertia: []\n";
    } else {
      out << "inertia:\n";
      for (auto item : msg.inertia) {
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

inline std::string to_yaml(const ConfigCreateTool_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ConfigCreateTool_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ConfigCreateTool_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ConfigCreateTool_Request>()
{
  return "dsr_msgs2::srv::ConfigCreateTool_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::ConfigCreateTool_Request>()
{
  return "dsr_msgs2/srv/ConfigCreateTool_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConfigCreateTool_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConfigCreateTool_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::srv::ConfigCreateTool_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const ConfigCreateTool_Response & msg,
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
  const ConfigCreateTool_Response & msg,
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

inline std::string to_yaml(const ConfigCreateTool_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::ConfigCreateTool_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::ConfigCreateTool_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::ConfigCreateTool_Response>()
{
  return "dsr_msgs2::srv::ConfigCreateTool_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::ConfigCreateTool_Response>()
{
  return "dsr_msgs2/srv/ConfigCreateTool_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConfigCreateTool_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConfigCreateTool_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::ConfigCreateTool_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::ConfigCreateTool>()
{
  return "dsr_msgs2::srv::ConfigCreateTool";
}

template<>
inline const char * name<dsr_msgs2::srv::ConfigCreateTool>()
{
  return "dsr_msgs2/srv/ConfigCreateTool";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::ConfigCreateTool>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::ConfigCreateTool_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::ConfigCreateTool_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::ConfigCreateTool>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::ConfigCreateTool_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::ConfigCreateTool_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::ConfigCreateTool>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::ConfigCreateTool_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::ConfigCreateTool_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__TRAITS_HPP_
