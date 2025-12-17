// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__TRAITS_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/msg/detail/robot_state_rt__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'jacobian_matrix'
// Member 'coriolis_matrix'
// Member 'mass_matrix'
#include "std_msgs/msg/detail/float64_multi_array__traits.hpp"

namespace dsr_msgs2
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotStateRt & msg,
  std::ostream & out)
{
  out << "{";
  // member: time_stamp
  {
    out << "time_stamp: ";
    rosidl_generator_traits::value_to_yaml(msg.time_stamp, out);
    out << ", ";
  }

  // member: actual_joint_position
  {
    if (msg.actual_joint_position.size() == 0) {
      out << "actual_joint_position: []";
    } else {
      out << "actual_joint_position: [";
      size_t pending_items = msg.actual_joint_position.size();
      for (auto item : msg.actual_joint_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_joint_position_abs
  {
    if (msg.actual_joint_position_abs.size() == 0) {
      out << "actual_joint_position_abs: []";
    } else {
      out << "actual_joint_position_abs: [";
      size_t pending_items = msg.actual_joint_position_abs.size();
      for (auto item : msg.actual_joint_position_abs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_joint_velocity
  {
    if (msg.actual_joint_velocity.size() == 0) {
      out << "actual_joint_velocity: []";
    } else {
      out << "actual_joint_velocity: [";
      size_t pending_items = msg.actual_joint_velocity.size();
      for (auto item : msg.actual_joint_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_joint_velocity_abs
  {
    if (msg.actual_joint_velocity_abs.size() == 0) {
      out << "actual_joint_velocity_abs: []";
    } else {
      out << "actual_joint_velocity_abs: [";
      size_t pending_items = msg.actual_joint_velocity_abs.size();
      for (auto item : msg.actual_joint_velocity_abs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_tcp_position
  {
    if (msg.actual_tcp_position.size() == 0) {
      out << "actual_tcp_position: []";
    } else {
      out << "actual_tcp_position: [";
      size_t pending_items = msg.actual_tcp_position.size();
      for (auto item : msg.actual_tcp_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_tcp_velocity
  {
    if (msg.actual_tcp_velocity.size() == 0) {
      out << "actual_tcp_velocity: []";
    } else {
      out << "actual_tcp_velocity: [";
      size_t pending_items = msg.actual_tcp_velocity.size();
      for (auto item : msg.actual_tcp_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_flange_position
  {
    if (msg.actual_flange_position.size() == 0) {
      out << "actual_flange_position: []";
    } else {
      out << "actual_flange_position: [";
      size_t pending_items = msg.actual_flange_position.size();
      for (auto item : msg.actual_flange_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_flange_velocity
  {
    if (msg.actual_flange_velocity.size() == 0) {
      out << "actual_flange_velocity: []";
    } else {
      out << "actual_flange_velocity: [";
      size_t pending_items = msg.actual_flange_velocity.size();
      for (auto item : msg.actual_flange_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_motor_torque
  {
    if (msg.actual_motor_torque.size() == 0) {
      out << "actual_motor_torque: []";
    } else {
      out << "actual_motor_torque: [";
      size_t pending_items = msg.actual_motor_torque.size();
      for (auto item : msg.actual_motor_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_joint_torque
  {
    if (msg.actual_joint_torque.size() == 0) {
      out << "actual_joint_torque: []";
    } else {
      out << "actual_joint_torque: [";
      size_t pending_items = msg.actual_joint_torque.size();
      for (auto item : msg.actual_joint_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: raw_joint_torque
  {
    if (msg.raw_joint_torque.size() == 0) {
      out << "raw_joint_torque: []";
    } else {
      out << "raw_joint_torque: [";
      size_t pending_items = msg.raw_joint_torque.size();
      for (auto item : msg.raw_joint_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: raw_force_torque
  {
    if (msg.raw_force_torque.size() == 0) {
      out << "raw_force_torque: []";
    } else {
      out << "raw_force_torque: [";
      size_t pending_items = msg.raw_force_torque.size();
      for (auto item : msg.raw_force_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: external_joint_torque
  {
    if (msg.external_joint_torque.size() == 0) {
      out << "external_joint_torque: []";
    } else {
      out << "external_joint_torque: [";
      size_t pending_items = msg.external_joint_torque.size();
      for (auto item : msg.external_joint_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: external_tcp_force
  {
    if (msg.external_tcp_force.size() == 0) {
      out << "external_tcp_force: []";
    } else {
      out << "external_tcp_force: [";
      size_t pending_items = msg.external_tcp_force.size();
      for (auto item : msg.external_tcp_force) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_joint_position
  {
    if (msg.target_joint_position.size() == 0) {
      out << "target_joint_position: []";
    } else {
      out << "target_joint_position: [";
      size_t pending_items = msg.target_joint_position.size();
      for (auto item : msg.target_joint_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_joint_velocity
  {
    if (msg.target_joint_velocity.size() == 0) {
      out << "target_joint_velocity: []";
    } else {
      out << "target_joint_velocity: [";
      size_t pending_items = msg.target_joint_velocity.size();
      for (auto item : msg.target_joint_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_joint_acceleration
  {
    if (msg.target_joint_acceleration.size() == 0) {
      out << "target_joint_acceleration: []";
    } else {
      out << "target_joint_acceleration: [";
      size_t pending_items = msg.target_joint_acceleration.size();
      for (auto item : msg.target_joint_acceleration) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_motor_torque
  {
    if (msg.target_motor_torque.size() == 0) {
      out << "target_motor_torque: []";
    } else {
      out << "target_motor_torque: [";
      size_t pending_items = msg.target_motor_torque.size();
      for (auto item : msg.target_motor_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_tcp_position
  {
    if (msg.target_tcp_position.size() == 0) {
      out << "target_tcp_position: []";
    } else {
      out << "target_tcp_position: [";
      size_t pending_items = msg.target_tcp_position.size();
      for (auto item : msg.target_tcp_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_tcp_velocity
  {
    if (msg.target_tcp_velocity.size() == 0) {
      out << "target_tcp_velocity: []";
    } else {
      out << "target_tcp_velocity: [";
      size_t pending_items = msg.target_tcp_velocity.size();
      for (auto item : msg.target_tcp_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: jacobian_matrix
  {
    if (msg.jacobian_matrix.size() == 0) {
      out << "jacobian_matrix: []";
    } else {
      out << "jacobian_matrix: [";
      size_t pending_items = msg.jacobian_matrix.size();
      for (auto item : msg.jacobian_matrix) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: gravity_torque
  {
    if (msg.gravity_torque.size() == 0) {
      out << "gravity_torque: []";
    } else {
      out << "gravity_torque: [";
      size_t pending_items = msg.gravity_torque.size();
      for (auto item : msg.gravity_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: coriolis_matrix
  {
    if (msg.coriolis_matrix.size() == 0) {
      out << "coriolis_matrix: []";
    } else {
      out << "coriolis_matrix: [";
      size_t pending_items = msg.coriolis_matrix.size();
      for (auto item : msg.coriolis_matrix) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: mass_matrix
  {
    if (msg.mass_matrix.size() == 0) {
      out << "mass_matrix: []";
    } else {
      out << "mass_matrix: [";
      size_t pending_items = msg.mass_matrix.size();
      for (auto item : msg.mass_matrix) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: solution_space
  {
    out << "solution_space: ";
    rosidl_generator_traits::value_to_yaml(msg.solution_space, out);
    out << ", ";
  }

  // member: singularity
  {
    out << "singularity: ";
    rosidl_generator_traits::value_to_yaml(msg.singularity, out);
    out << ", ";
  }

  // member: operation_speed_rate
  {
    out << "operation_speed_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.operation_speed_rate, out);
    out << ", ";
  }

  // member: joint_temperature
  {
    if (msg.joint_temperature.size() == 0) {
      out << "joint_temperature: []";
    } else {
      out << "joint_temperature: [";
      size_t pending_items = msg.joint_temperature.size();
      for (auto item : msg.joint_temperature) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: controller_digital_input
  {
    out << "controller_digital_input: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_digital_input, out);
    out << ", ";
  }

  // member: controller_digital_output
  {
    out << "controller_digital_output: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_digital_output, out);
    out << ", ";
  }

  // member: controller_analog_input_type
  {
    if (msg.controller_analog_input_type.size() == 0) {
      out << "controller_analog_input_type: []";
    } else {
      out << "controller_analog_input_type: [";
      size_t pending_items = msg.controller_analog_input_type.size();
      for (auto item : msg.controller_analog_input_type) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: controller_analog_input
  {
    if (msg.controller_analog_input.size() == 0) {
      out << "controller_analog_input: []";
    } else {
      out << "controller_analog_input: [";
      size_t pending_items = msg.controller_analog_input.size();
      for (auto item : msg.controller_analog_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: controller_analog_output_type
  {
    if (msg.controller_analog_output_type.size() == 0) {
      out << "controller_analog_output_type: []";
    } else {
      out << "controller_analog_output_type: [";
      size_t pending_items = msg.controller_analog_output_type.size();
      for (auto item : msg.controller_analog_output_type) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: controller_analog_output
  {
    if (msg.controller_analog_output.size() == 0) {
      out << "controller_analog_output: []";
    } else {
      out << "controller_analog_output: [";
      size_t pending_items = msg.controller_analog_output.size();
      for (auto item : msg.controller_analog_output) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: flange_digital_input
  {
    out << "flange_digital_input: ";
    rosidl_generator_traits::value_to_yaml(msg.flange_digital_input, out);
    out << ", ";
  }

  // member: flange_digital_output
  {
    out << "flange_digital_output: ";
    rosidl_generator_traits::value_to_yaml(msg.flange_digital_output, out);
    out << ", ";
  }

  // member: flange_analog_input
  {
    if (msg.flange_analog_input.size() == 0) {
      out << "flange_analog_input: []";
    } else {
      out << "flange_analog_input: [";
      size_t pending_items = msg.flange_analog_input.size();
      for (auto item : msg.flange_analog_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: external_encoder_strobe_count
  {
    if (msg.external_encoder_strobe_count.size() == 0) {
      out << "external_encoder_strobe_count: []";
    } else {
      out << "external_encoder_strobe_count: [";
      size_t pending_items = msg.external_encoder_strobe_count.size();
      for (auto item : msg.external_encoder_strobe_count) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: external_encoder_count
  {
    if (msg.external_encoder_count.size() == 0) {
      out << "external_encoder_count: []";
    } else {
      out << "external_encoder_count: [";
      size_t pending_items = msg.external_encoder_count.size();
      for (auto item : msg.external_encoder_count) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: goal_joint_position
  {
    if (msg.goal_joint_position.size() == 0) {
      out << "goal_joint_position: []";
    } else {
      out << "goal_joint_position: [";
      size_t pending_items = msg.goal_joint_position.size();
      for (auto item : msg.goal_joint_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: goal_tcp_position
  {
    if (msg.goal_tcp_position.size() == 0) {
      out << "goal_tcp_position: []";
    } else {
      out << "goal_tcp_position: [";
      size_t pending_items = msg.goal_tcp_position.size();
      for (auto item : msg.goal_tcp_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: robot_mode
  {
    out << "robot_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_mode, out);
    out << ", ";
  }

  // member: robot_state
  {
    out << "robot_state: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state, out);
    out << ", ";
  }

  // member: control_mode
  {
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << ", ";
  }

  // member: reserved
  {
    if (msg.reserved.size() == 0) {
      out << "reserved: []";
    } else {
      out << "reserved: [";
      size_t pending_items = msg.reserved.size();
      for (auto item : msg.reserved) {
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
  const RobotStateRt & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: time_stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_stamp: ";
    rosidl_generator_traits::value_to_yaml(msg.time_stamp, out);
    out << "\n";
  }

  // member: actual_joint_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_joint_position.size() == 0) {
      out << "actual_joint_position: []\n";
    } else {
      out << "actual_joint_position:\n";
      for (auto item : msg.actual_joint_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_joint_position_abs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_joint_position_abs.size() == 0) {
      out << "actual_joint_position_abs: []\n";
    } else {
      out << "actual_joint_position_abs:\n";
      for (auto item : msg.actual_joint_position_abs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_joint_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_joint_velocity.size() == 0) {
      out << "actual_joint_velocity: []\n";
    } else {
      out << "actual_joint_velocity:\n";
      for (auto item : msg.actual_joint_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_joint_velocity_abs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_joint_velocity_abs.size() == 0) {
      out << "actual_joint_velocity_abs: []\n";
    } else {
      out << "actual_joint_velocity_abs:\n";
      for (auto item : msg.actual_joint_velocity_abs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_tcp_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_tcp_position.size() == 0) {
      out << "actual_tcp_position: []\n";
    } else {
      out << "actual_tcp_position:\n";
      for (auto item : msg.actual_tcp_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_tcp_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_tcp_velocity.size() == 0) {
      out << "actual_tcp_velocity: []\n";
    } else {
      out << "actual_tcp_velocity:\n";
      for (auto item : msg.actual_tcp_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_flange_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_flange_position.size() == 0) {
      out << "actual_flange_position: []\n";
    } else {
      out << "actual_flange_position:\n";
      for (auto item : msg.actual_flange_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_flange_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_flange_velocity.size() == 0) {
      out << "actual_flange_velocity: []\n";
    } else {
      out << "actual_flange_velocity:\n";
      for (auto item : msg.actual_flange_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_motor_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_motor_torque.size() == 0) {
      out << "actual_motor_torque: []\n";
    } else {
      out << "actual_motor_torque:\n";
      for (auto item : msg.actual_motor_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_joint_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_joint_torque.size() == 0) {
      out << "actual_joint_torque: []\n";
    } else {
      out << "actual_joint_torque:\n";
      for (auto item : msg.actual_joint_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: raw_joint_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.raw_joint_torque.size() == 0) {
      out << "raw_joint_torque: []\n";
    } else {
      out << "raw_joint_torque:\n";
      for (auto item : msg.raw_joint_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: raw_force_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.raw_force_torque.size() == 0) {
      out << "raw_force_torque: []\n";
    } else {
      out << "raw_force_torque:\n";
      for (auto item : msg.raw_force_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: external_joint_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.external_joint_torque.size() == 0) {
      out << "external_joint_torque: []\n";
    } else {
      out << "external_joint_torque:\n";
      for (auto item : msg.external_joint_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: external_tcp_force
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.external_tcp_force.size() == 0) {
      out << "external_tcp_force: []\n";
    } else {
      out << "external_tcp_force:\n";
      for (auto item : msg.external_tcp_force) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_joint_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_joint_position.size() == 0) {
      out << "target_joint_position: []\n";
    } else {
      out << "target_joint_position:\n";
      for (auto item : msg.target_joint_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_joint_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_joint_velocity.size() == 0) {
      out << "target_joint_velocity: []\n";
    } else {
      out << "target_joint_velocity:\n";
      for (auto item : msg.target_joint_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_joint_acceleration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_joint_acceleration.size() == 0) {
      out << "target_joint_acceleration: []\n";
    } else {
      out << "target_joint_acceleration:\n";
      for (auto item : msg.target_joint_acceleration) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_motor_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_motor_torque.size() == 0) {
      out << "target_motor_torque: []\n";
    } else {
      out << "target_motor_torque:\n";
      for (auto item : msg.target_motor_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_tcp_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_tcp_position.size() == 0) {
      out << "target_tcp_position: []\n";
    } else {
      out << "target_tcp_position:\n";
      for (auto item : msg.target_tcp_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_tcp_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_tcp_velocity.size() == 0) {
      out << "target_tcp_velocity: []\n";
    } else {
      out << "target_tcp_velocity:\n";
      for (auto item : msg.target_tcp_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: jacobian_matrix
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.jacobian_matrix.size() == 0) {
      out << "jacobian_matrix: []\n";
    } else {
      out << "jacobian_matrix:\n";
      for (auto item : msg.jacobian_matrix) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: gravity_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gravity_torque.size() == 0) {
      out << "gravity_torque: []\n";
    } else {
      out << "gravity_torque:\n";
      for (auto item : msg.gravity_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: coriolis_matrix
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.coriolis_matrix.size() == 0) {
      out << "coriolis_matrix: []\n";
    } else {
      out << "coriolis_matrix:\n";
      for (auto item : msg.coriolis_matrix) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: mass_matrix
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.mass_matrix.size() == 0) {
      out << "mass_matrix: []\n";
    } else {
      out << "mass_matrix:\n";
      for (auto item : msg.mass_matrix) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: solution_space
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "solution_space: ";
    rosidl_generator_traits::value_to_yaml(msg.solution_space, out);
    out << "\n";
  }

  // member: singularity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "singularity: ";
    rosidl_generator_traits::value_to_yaml(msg.singularity, out);
    out << "\n";
  }

  // member: operation_speed_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "operation_speed_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.operation_speed_rate, out);
    out << "\n";
  }

  // member: joint_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_temperature.size() == 0) {
      out << "joint_temperature: []\n";
    } else {
      out << "joint_temperature:\n";
      for (auto item : msg.joint_temperature) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: controller_digital_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "controller_digital_input: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_digital_input, out);
    out << "\n";
  }

  // member: controller_digital_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "controller_digital_output: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_digital_output, out);
    out << "\n";
  }

  // member: controller_analog_input_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.controller_analog_input_type.size() == 0) {
      out << "controller_analog_input_type: []\n";
    } else {
      out << "controller_analog_input_type:\n";
      for (auto item : msg.controller_analog_input_type) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: controller_analog_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.controller_analog_input.size() == 0) {
      out << "controller_analog_input: []\n";
    } else {
      out << "controller_analog_input:\n";
      for (auto item : msg.controller_analog_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: controller_analog_output_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.controller_analog_output_type.size() == 0) {
      out << "controller_analog_output_type: []\n";
    } else {
      out << "controller_analog_output_type:\n";
      for (auto item : msg.controller_analog_output_type) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: controller_analog_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.controller_analog_output.size() == 0) {
      out << "controller_analog_output: []\n";
    } else {
      out << "controller_analog_output:\n";
      for (auto item : msg.controller_analog_output) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: flange_digital_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "flange_digital_input: ";
    rosidl_generator_traits::value_to_yaml(msg.flange_digital_input, out);
    out << "\n";
  }

  // member: flange_digital_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "flange_digital_output: ";
    rosidl_generator_traits::value_to_yaml(msg.flange_digital_output, out);
    out << "\n";
  }

  // member: flange_analog_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.flange_analog_input.size() == 0) {
      out << "flange_analog_input: []\n";
    } else {
      out << "flange_analog_input:\n";
      for (auto item : msg.flange_analog_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: external_encoder_strobe_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.external_encoder_strobe_count.size() == 0) {
      out << "external_encoder_strobe_count: []\n";
    } else {
      out << "external_encoder_strobe_count:\n";
      for (auto item : msg.external_encoder_strobe_count) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: external_encoder_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.external_encoder_count.size() == 0) {
      out << "external_encoder_count: []\n";
    } else {
      out << "external_encoder_count:\n";
      for (auto item : msg.external_encoder_count) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: goal_joint_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.goal_joint_position.size() == 0) {
      out << "goal_joint_position: []\n";
    } else {
      out << "goal_joint_position:\n";
      for (auto item : msg.goal_joint_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: goal_tcp_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.goal_tcp_position.size() == 0) {
      out << "goal_tcp_position: []\n";
    } else {
      out << "goal_tcp_position:\n";
      for (auto item : msg.goal_tcp_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: robot_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_mode, out);
    out << "\n";
  }

  // member: robot_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_state: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state, out);
    out << "\n";
  }

  // member: control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << "\n";
  }

  // member: reserved
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.reserved.size() == 0) {
      out << "reserved: []\n";
    } else {
      out << "reserved:\n";
      for (auto item : msg.reserved) {
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

inline std::string to_yaml(const RobotStateRt & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace dsr_msgs2

namespace rosidl_generator_traits
{

[[deprecated("use dsr_msgs2::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dsr_msgs2::msg::RobotStateRt & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::msg::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::msg::RobotStateRt & msg)
{
  return dsr_msgs2::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::msg::RobotStateRt>()
{
  return "dsr_msgs2::msg::RobotStateRt";
}

template<>
inline const char * name<dsr_msgs2::msg::RobotStateRt>()
{
  return "dsr_msgs2/msg/RobotStateRt";
}

template<>
struct has_fixed_size<dsr_msgs2::msg::RobotStateRt>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::msg::RobotStateRt>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::msg::RobotStateRt>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__TRAITS_HPP_
