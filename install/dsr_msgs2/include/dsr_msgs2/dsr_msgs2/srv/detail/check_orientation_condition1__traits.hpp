// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/CheckOrientationCondition1.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/check_orientation_condition1__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const CheckOrientationCondition1_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: axis
  {
    out << "axis: ";
    rosidl_generator_traits::value_to_yaml(msg.axis, out);
    out << ", ";
  }

  // member: min
  {
    if (msg.min.size() == 0) {
      out << "min: []";
    } else {
      out << "min: [";
      size_t pending_items = msg.min.size();
      for (auto item : msg.min) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: max
  {
    if (msg.max.size() == 0) {
      out << "max: []";
    } else {
      out << "max: [";
      size_t pending_items = msg.max.size();
      for (auto item : msg.max) {
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

  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CheckOrientationCondition1_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: axis
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "axis: ";
    rosidl_generator_traits::value_to_yaml(msg.axis, out);
    out << "\n";
  }

  // member: min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.min.size() == 0) {
      out << "min: []\n";
    } else {
      out << "min:\n";
      for (auto item : msg.min) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.max.size() == 0) {
      out << "max: []\n";
    } else {
      out << "max:\n";
      for (auto item : msg.max) {
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

  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CheckOrientationCondition1_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::CheckOrientationCondition1_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::CheckOrientationCondition1_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::CheckOrientationCondition1_Request>()
{
  return "dsr_msgs2::srv::CheckOrientationCondition1_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::CheckOrientationCondition1_Request>()
{
  return "dsr_msgs2/srv/CheckOrientationCondition1_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CheckOrientationCondition1_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::CheckOrientationCondition1_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::CheckOrientationCondition1_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const CheckOrientationCondition1_Response & msg,
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
  const CheckOrientationCondition1_Response & msg,
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

inline std::string to_yaml(const CheckOrientationCondition1_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::CheckOrientationCondition1_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::CheckOrientationCondition1_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::CheckOrientationCondition1_Response>()
{
  return "dsr_msgs2::srv::CheckOrientationCondition1_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::CheckOrientationCondition1_Response>()
{
  return "dsr_msgs2/srv/CheckOrientationCondition1_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CheckOrientationCondition1_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::CheckOrientationCondition1_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::CheckOrientationCondition1_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::CheckOrientationCondition1>()
{
  return "dsr_msgs2::srv::CheckOrientationCondition1";
}

template<>
inline const char * name<dsr_msgs2::srv::CheckOrientationCondition1>()
{
  return "dsr_msgs2/srv/CheckOrientationCondition1";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CheckOrientationCondition1>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::CheckOrientationCondition1_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::CheckOrientationCondition1_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::CheckOrientationCondition1>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::CheckOrientationCondition1_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::CheckOrientationCondition1_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::CheckOrientationCondition1>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::CheckOrientationCondition1_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::CheckOrientationCondition1_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION1__TRAITS_HPP_
