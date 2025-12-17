// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/SetDesiredForce.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/set_desired_force__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetDesiredForce_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: fd
  {
    if (msg.fd.size() == 0) {
      out << "fd: []";
    } else {
      out << "fd: [";
      size_t pending_items = msg.fd.size();
      for (auto item : msg.fd) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: dir
  {
    if (msg.dir.size() == 0) {
      out << "dir: []";
    } else {
      out << "dir: [";
      size_t pending_items = msg.dir.size();
      for (auto item : msg.dir) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ref
  {
    out << "ref: ";
    rosidl_generator_traits::value_to_yaml(msg.ref, out);
    out << ", ";
  }

  // member: time
  {
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << ", ";
  }

  // member: mod
  {
    out << "mod: ";
    rosidl_generator_traits::value_to_yaml(msg.mod, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetDesiredForce_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: fd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.fd.size() == 0) {
      out << "fd: []\n";
    } else {
      out << "fd:\n";
      for (auto item : msg.fd) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: dir
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dir.size() == 0) {
      out << "dir: []\n";
    } else {
      out << "dir:\n";
      for (auto item : msg.dir) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ref
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref: ";
    rosidl_generator_traits::value_to_yaml(msg.ref, out);
    out << "\n";
  }

  // member: time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << "\n";
  }

  // member: mod
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mod: ";
    rosidl_generator_traits::value_to_yaml(msg.mod, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetDesiredForce_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::SetDesiredForce_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::SetDesiredForce_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::SetDesiredForce_Request>()
{
  return "dsr_msgs2::srv::SetDesiredForce_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::SetDesiredForce_Request>()
{
  return "dsr_msgs2/srv/SetDesiredForce_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::SetDesiredForce_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::SetDesiredForce_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::SetDesiredForce_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetDesiredForce_Response & msg,
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
  const SetDesiredForce_Response & msg,
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

inline std::string to_yaml(const SetDesiredForce_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::SetDesiredForce_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::SetDesiredForce_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::SetDesiredForce_Response>()
{
  return "dsr_msgs2::srv::SetDesiredForce_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::SetDesiredForce_Response>()
{
  return "dsr_msgs2/srv/SetDesiredForce_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::SetDesiredForce_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::SetDesiredForce_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::SetDesiredForce_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::SetDesiredForce>()
{
  return "dsr_msgs2::srv::SetDesiredForce";
}

template<>
inline const char * name<dsr_msgs2::srv::SetDesiredForce>()
{
  return "dsr_msgs2/srv/SetDesiredForce";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::SetDesiredForce>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::SetDesiredForce_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::SetDesiredForce_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::SetDesiredForce>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::SetDesiredForce_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::SetDesiredForce_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::SetDesiredForce>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::SetDesiredForce_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::SetDesiredForce_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__TRAITS_HPP_
