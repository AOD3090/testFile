// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:srv/MoveSpiral.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__TRAITS_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/srv/detail/move_spiral__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveSpiral_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: revolution
  {
    out << "revolution: ";
    rosidl_generator_traits::value_to_yaml(msg.revolution, out);
    out << ", ";
  }

  // member: max_radius
  {
    out << "max_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.max_radius, out);
    out << ", ";
  }

  // member: max_length
  {
    out << "max_length: ";
    rosidl_generator_traits::value_to_yaml(msg.max_length, out);
    out << ", ";
  }

  // member: target_pos
  {
    if (msg.target_pos.size() == 0) {
      out << "target_pos: []";
    } else {
      out << "target_pos: [";
      size_t pending_items = msg.target_pos.size();
      for (auto item : msg.target_pos) {
        rosidl_generator_traits::value_to_yaml(item, out);
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

  // member: task_axis
  {
    out << "task_axis: ";
    rosidl_generator_traits::value_to_yaml(msg.task_axis, out);
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

  // member: spiral_dir
  {
    out << "spiral_dir: ";
    rosidl_generator_traits::value_to_yaml(msg.spiral_dir, out);
    out << ", ";
  }

  // member: rot_dir
  {
    out << "rot_dir: ";
    rosidl_generator_traits::value_to_yaml(msg.rot_dir, out);
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
  const MoveSpiral_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: revolution
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "revolution: ";
    rosidl_generator_traits::value_to_yaml(msg.revolution, out);
    out << "\n";
  }

  // member: max_radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_radius: ";
    rosidl_generator_traits::value_to_yaml(msg.max_radius, out);
    out << "\n";
  }

  // member: max_length
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_length: ";
    rosidl_generator_traits::value_to_yaml(msg.max_length, out);
    out << "\n";
  }

  // member: target_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_pos.size() == 0) {
      out << "target_pos: []\n";
    } else {
      out << "target_pos:\n";
      for (auto item : msg.target_pos) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
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

  // member: task_axis
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "task_axis: ";
    rosidl_generator_traits::value_to_yaml(msg.task_axis, out);
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

  // member: spiral_dir
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "spiral_dir: ";
    rosidl_generator_traits::value_to_yaml(msg.spiral_dir, out);
    out << "\n";
  }

  // member: rot_dir
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rot_dir: ";
    rosidl_generator_traits::value_to_yaml(msg.rot_dir, out);
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

inline std::string to_yaml(const MoveSpiral_Request & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::MoveSpiral_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveSpiral_Request & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveSpiral_Request>()
{
  return "dsr_msgs2::srv::MoveSpiral_Request";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveSpiral_Request>()
{
  return "dsr_msgs2/srv/MoveSpiral_Request";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveSpiral_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveSpiral_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::MoveSpiral_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dsr_msgs2
{

namespace srv
{

inline void to_flow_style_yaml(
  const MoveSpiral_Response & msg,
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
  const MoveSpiral_Response & msg,
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

inline std::string to_yaml(const MoveSpiral_Response & msg, bool use_flow_style = false)
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
  const dsr_msgs2::srv::MoveSpiral_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::srv::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::srv::MoveSpiral_Response & msg)
{
  return dsr_msgs2::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::srv::MoveSpiral_Response>()
{
  return "dsr_msgs2::srv::MoveSpiral_Response";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveSpiral_Response>()
{
  return "dsr_msgs2/srv/MoveSpiral_Response";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveSpiral_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveSpiral_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dsr_msgs2::srv::MoveSpiral_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dsr_msgs2::srv::MoveSpiral>()
{
  return "dsr_msgs2::srv::MoveSpiral";
}

template<>
inline const char * name<dsr_msgs2::srv::MoveSpiral>()
{
  return "dsr_msgs2/srv/MoveSpiral";
}

template<>
struct has_fixed_size<dsr_msgs2::srv::MoveSpiral>
  : std::integral_constant<
    bool,
    has_fixed_size<dsr_msgs2::srv::MoveSpiral_Request>::value &&
    has_fixed_size<dsr_msgs2::srv::MoveSpiral_Response>::value
  >
{
};

template<>
struct has_bounded_size<dsr_msgs2::srv::MoveSpiral>
  : std::integral_constant<
    bool,
    has_bounded_size<dsr_msgs2::srv::MoveSpiral_Request>::value &&
    has_bounded_size<dsr_msgs2::srv::MoveSpiral_Response>::value
  >
{
};

template<>
struct is_service<dsr_msgs2::srv::MoveSpiral>
  : std::true_type
{
};

template<>
struct is_service_request<dsr_msgs2::srv::MoveSpiral_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dsr_msgs2::srv::MoveSpiral_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__TRAITS_HPP_
