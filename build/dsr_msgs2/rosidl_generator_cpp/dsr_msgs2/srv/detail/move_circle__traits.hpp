// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/MoveCircle.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/move_circle__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'pos'
#include "std_msgs/msg/detail/float64_multi_array__traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveCircle_Request & msg,
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
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: vel
  {
    if (msg.vel.size() == 0) {
      out << "vel: []";
    } else {
      out << "vel: [";
      size_t pending_items = msg.vel.size();
      for (auto item : msg.vel) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: acc
  {
    if (msg.acc.size() == 0) {
      out << "acc: []";
    } else {
      out << "acc: [";
      size_t pending_items = msg.acc.size();
      for (auto item : msg.acc) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: time
  {
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << ", ";
  }

  // member: radius
  {
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
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
    out << ", ";
  }

  // member: angle1
  {
    out << "angle1: ";
    rosidl_generator_traits::value_to_yaml(msg.angle1, out);
    out << ", ";
  }

  // member: angle2
  {
    out << "angle2: ";
    rosidl_generator_traits::value_to_yaml(msg.angle2, out);
    out << ", ";
  }

  // member: blend_type
  {
    out << "blend_type: ";
    rosidl_generator_traits::value_to_yaml(msg.blend_type, out);
    out << ", ";
  }

  // member: sync_type
  {
    out << "sync_type: ";
    rosidl_generator_traits::value_to_yaml(msg.sync_type, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MoveCircle_Request & msg,
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
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: vel
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.vel.size() == 0) {
      out << "vel: []\n";
    } else {
      out << "vel:\n";
      for (auto item : msg.vel) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: acc
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.acc.size() == 0) {
      out << "acc: []\n";
    } else {
      out << "acc:\n";
      for (auto item : msg.acc) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
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

  // member: radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
    out << "\n";
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

  // member: angle1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angle1: ";
    rosidl_generator_traits::value_to_yaml(msg.angle1, out);
    out << "\n";
  }

  // member: angle2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angle2: ";
    rosidl_generator_traits::value_to_yaml(msg.angle2, out);
    out << "\n";
  }

  // member: blend_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "blend_type: ";
    rosidl_generator_traits::value_to_yaml(msg.blend_type, out);
    out << "\n";
  }

  // member: sync_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sync_type: ";
    rosidl_generator_traits::value_to_yaml(msg.sync_type, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MoveCircle_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::MoveCircle_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveCircle_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveCircle_Request>()
{
  return "dsr_msgs2::srv::MoveCircle_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveCircle_Request>()
{
  return "dsr_msgs2/srv/MoveCircle_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveCircle_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveCircle_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::srv::MoveCircle_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveCircle_Response & msg,
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
  const MoveCircle_Response & msg,
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

inline std::string to_yaml(const MoveCircle_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::MoveCircle_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveCircle_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveCircle_Response>()
{
  return "dsr_msgs2::srv::MoveCircle_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveCircle_Response>()
{
  return "dsr_msgs2/srv/MoveCircle_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveCircle_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveCircle_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::MoveCircle_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::MoveCircle>()
{
  return "dsr_msgs2::srv::MoveCircle";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveCircle>()
{
  return "dsr_msgs2/srv/MoveCircle";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveCircle>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::MoveCircle_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::MoveCircle_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveCircle>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::MoveCircle_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::MoveCircle_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::MoveCircle>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::MoveCircle_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::MoveCircle_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_CIRCLE__TRAITS_HPP_
