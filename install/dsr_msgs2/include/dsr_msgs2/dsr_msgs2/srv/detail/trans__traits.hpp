// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/Trans.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TRANS__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__TRANS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/trans__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const Trans_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: pos
  {
    if (msg.pos.size() == 0) {
      out << "pos: []";
    } else {
      out << "pos: [";
      size_t pending_items = msg.pos.size();
      for (auto item : msg.pos) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: delta
  {
    if (msg.delta.size() == 0) {
      out << "delta: []";
    } else {
      out << "delta: [";
      size_t pending_items = msg.delta.size();
      for (auto item : msg.delta) {
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

  // member: ref_out
  {
    out << "ref_out: ";
    rosidl_generator_traits::value_to_yaml(msg.ref_out, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Trans_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.pos.size() == 0) {
      out << "pos: []\n";
    } else {
      out << "pos:\n";
      for (auto item : msg.pos) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: delta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.delta.size() == 0) {
      out << "delta: []\n";
    } else {
      out << "delta:\n";
      for (auto item : msg.delta) {
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

inline std::string to_yaml(const Trans_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::Trans_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::Trans_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::Trans_Request>()
{
  return "dsr_msgs2::srv::Trans_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::Trans_Request>()
{
  return "dsr_msgs2/srv/Trans_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::Trans_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::Trans_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::Trans_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const Trans_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: trans_pos
  {
    if (msg.trans_pos.size() == 0) {
      out << "trans_pos: []";
    } else {
      out << "trans_pos: [";
      size_t pending_items = msg.trans_pos.size();
      for (auto item : msg.trans_pos) {
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
  const Trans_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: trans_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.trans_pos.size() == 0) {
      out << "trans_pos: []\n";
    } else {
      out << "trans_pos:\n";
      for (auto item : msg.trans_pos) {
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

inline std::string to_yaml(const Trans_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::Trans_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::Trans_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::Trans_Response>()
{
  return "dsr_msgs2::srv::Trans_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::Trans_Response>()
{
  return "dsr_msgs2/srv/Trans_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::Trans_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::Trans_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::Trans_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::Trans>()
{
  return "dsr_msgs2::srv::Trans";
}

template<>
inline const char * name<dsr_msgs2::srv::Trans>()
{
  return "dsr_msgs2/srv/Trans";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::Trans>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::Trans_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::Trans_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::Trans>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::Trans_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::Trans_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::Trans>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::Trans_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::Trans_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__TRANS__TRAITS_HPP_
