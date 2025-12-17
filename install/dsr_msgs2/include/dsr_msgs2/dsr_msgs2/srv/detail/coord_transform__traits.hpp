// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/CoordTransform.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/coord_transform__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const CoordTransform_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: pos_in
  {
    if (msg.pos_in.size() == 0) {
      out << "pos_in: []";
    } else {
      out << "pos_in: [";
      size_t pending_items = msg.pos_in.size();
      for (auto item : msg.pos_in) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ref_in
  {
    out << "ref_in: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_in, out);
    out << ", ";
  }

  // member: ref_out
  {
    out << "ref_out: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_out, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CoordTransform_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pos_in
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.pos_in.size() == 0) {
      out << "pos_in: []\n";
    } else {
      out << "pos_in:\n";
      for (auto item : msg.pos_in) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ref_in
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref_in: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_in, out);
    out << "\n";
  }

  // member: ref_out
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ref_out: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_out, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CoordTransform_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::CoordTransform_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::CoordTransform_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::CoordTransform_Request>()
{
  return "dsr_msgs2::srv::CoordTransform_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::CoordTransform_Request>()
{
  return "dsr_msgs2/srv/CoordTransform_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CoordTransform_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::CoordTransform_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::CoordTransform_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const CoordTransform_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: conv_posx
  {
    if (msg.conv_posx.size() == 0) {
      out << "conv_posx: []";
    } else {
      out << "conv_posx: [";
      size_t pending_items = msg.conv_posx.size();
      for (auto item : msg.conv_posx) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const CoordTransform_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: conv_posx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.conv_posx.size() == 0) {
      out << "conv_posx: []\n";
    } else {
      out << "conv_posx:\n";
      for (auto item : msg.conv_posx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
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

inline std::string to_yaml(const CoordTransform_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::CoordTransform_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::CoordTransform_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::CoordTransform_Response>()
{
  return "dsr_msgs2::srv::CoordTransform_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::CoordTransform_Response>()
{
  return "dsr_msgs2/srv/CoordTransform_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CoordTransform_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::CoordTransform_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::CoordTransform_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::CoordTransform>()
{
  return "dsr_msgs2::srv::CoordTransform";
}

template<>
inline const char * name<dsr_msgs2::srv::CoordTransform>()
{
  return "dsr_msgs2/srv/CoordTransform";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::CoordTransform>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::CoordTransform_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::CoordTransform_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::CoordTransform>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::CoordTransform_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::CoordTransform_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::CoordTransform>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::CoordTransform_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::CoordTransform_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__TRAITS_HPP_
