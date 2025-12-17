// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__TRAITS_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dsr_msgs2/msg/detail/robot_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'rotation_matrix'
// Member 'f_current_pos_world'
// Member 'f_rotation_matrix_world'
// Member 'f_current_pos_user'
// Member 'f_rotation_matrix_user'
#include "std_msgs/msg/detail/float64_multi_array__traits.hpp"
// Member 'modbus_state'
#include "dsr_msgs2/msg/detail/modbus_state__traits.hpp"

namespace dsr_msgs2
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotState & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_state
  {
    out << "robot_state: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state, out);
    out << ", ";
  }

  // member: robot_state_str
  {
    out << "robot_state_str: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state_str, out);
    out << ", ";
  }

  // member: actual_mode
  {
    out << "actual_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_mode, out);
    out << ", ";
  }

  // member: actual_space
  {
    out << "actual_space: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_space, out);
    out << ", ";
  }

  // member: current_posj
  {
    if (msg.current_posj.size() == 0) {
      out << "current_posj: []";
    } else {
      out << "current_posj: [";
      size_t pending_items = msg.current_posj.size();
      for (auto item : msg.current_posj) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: current_velj
  {
    if (msg.current_velj.size() == 0) {
      out << "current_velj: []";
    } else {
      out << "current_velj: [";
      size_t pending_items = msg.current_velj.size();
      for (auto item : msg.current_velj) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_abs
  {
    if (msg.joint_abs.size() == 0) {
      out << "joint_abs: []";
    } else {
      out << "joint_abs: [";
      size_t pending_items = msg.joint_abs.size();
      for (auto item : msg.joint_abs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_err
  {
    if (msg.joint_err.size() == 0) {
      out << "joint_err: []";
    } else {
      out << "joint_err: [";
      size_t pending_items = msg.joint_err.size();
      for (auto item : msg.joint_err) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_posj
  {
    if (msg.target_posj.size() == 0) {
      out << "target_posj: []";
    } else {
      out << "target_posj: [";
      size_t pending_items = msg.target_posj.size();
      for (auto item : msg.target_posj) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_velj
  {
    if (msg.target_velj.size() == 0) {
      out << "target_velj: []";
    } else {
      out << "target_velj: [";
      size_t pending_items = msg.target_velj.size();
      for (auto item : msg.target_velj) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: current_posx
  {
    if (msg.current_posx.size() == 0) {
      out << "current_posx: []";
    } else {
      out << "current_posx: [";
      size_t pending_items = msg.current_posx.size();
      for (auto item : msg.current_posx) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: current_tool_posx
  {
    if (msg.current_tool_posx.size() == 0) {
      out << "current_tool_posx: []";
    } else {
      out << "current_tool_posx: [";
      size_t pending_items = msg.current_tool_posx.size();
      for (auto item : msg.current_tool_posx) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: current_velx
  {
    if (msg.current_velx.size() == 0) {
      out << "current_velx: []";
    } else {
      out << "current_velx: [";
      size_t pending_items = msg.current_velx.size();
      for (auto item : msg.current_velx) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: task_err
  {
    if (msg.task_err.size() == 0) {
      out << "task_err: []";
    } else {
      out << "task_err: [";
      size_t pending_items = msg.task_err.size();
      for (auto item : msg.task_err) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_velx
  {
    if (msg.target_velx.size() == 0) {
      out << "target_velx: []";
    } else {
      out << "target_velx: [";
      size_t pending_items = msg.target_velx.size();
      for (auto item : msg.target_velx) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: target_posx
  {
    if (msg.target_posx.size() == 0) {
      out << "target_posx: []";
    } else {
      out << "target_posx: [";
      size_t pending_items = msg.target_posx.size();
      for (auto item : msg.target_posx) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: dynamic_tor
  {
    if (msg.dynamic_tor.size() == 0) {
      out << "dynamic_tor: []";
    } else {
      out << "dynamic_tor: [";
      size_t pending_items = msg.dynamic_tor.size();
      for (auto item : msg.dynamic_tor) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_jts
  {
    if (msg.actual_jts.size() == 0) {
      out << "actual_jts: []";
    } else {
      out << "actual_jts: [";
      size_t pending_items = msg.actual_jts.size();
      for (auto item : msg.actual_jts) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_ejt
  {
    if (msg.actual_ejt.size() == 0) {
      out << "actual_ejt: []";
    } else {
      out << "actual_ejt: [";
      size_t pending_items = msg.actual_ejt.size();
      for (auto item : msg.actual_ejt) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_ett
  {
    if (msg.actual_ett.size() == 0) {
      out << "actual_ett: []";
    } else {
      out << "actual_ett: [";
      size_t pending_items = msg.actual_ett.size();
      for (auto item : msg.actual_ett) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_bk
  {
    if (msg.actual_bk.size() == 0) {
      out << "actual_bk: []";
    } else {
      out << "actual_bk: [";
      size_t pending_items = msg.actual_bk.size();
      for (auto item : msg.actual_bk) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_mc
  {
    if (msg.actual_mc.size() == 0) {
      out << "actual_mc: []";
    } else {
      out << "actual_mc: [";
      size_t pending_items = msg.actual_mc.size();
      for (auto item : msg.actual_mc) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: actual_mt
  {
    if (msg.actual_mt.size() == 0) {
      out << "actual_mt: []";
    } else {
      out << "actual_mt: [";
      size_t pending_items = msg.actual_mt.size();
      for (auto item : msg.actual_mt) {
        rosidl_generator_traits::value_to_yaml(item, out);
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

  // member: sync_time
  {
    out << "sync_time: ";
    rosidl_generator_traits::value_to_yaml(msg.sync_time, out);
    out << ", ";
  }

  // member: actual_bt
  {
    if (msg.actual_bt.size() == 0) {
      out << "actual_bt: []";
    } else {
      out << "actual_bt: [";
      size_t pending_items = msg.actual_bt.size();
      for (auto item : msg.actual_bt) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: rotation_matrix
  {
    if (msg.rotation_matrix.size() == 0) {
      out << "rotation_matrix: []";
    } else {
      out << "rotation_matrix: [";
      size_t pending_items = msg.rotation_matrix.size();
      for (auto item : msg.rotation_matrix) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ctrlbox_digital_input
  {
    if (msg.ctrlbox_digital_input.size() == 0) {
      out << "ctrlbox_digital_input: []";
    } else {
      out << "ctrlbox_digital_input: [";
      size_t pending_items = msg.ctrlbox_digital_input.size();
      for (auto item : msg.ctrlbox_digital_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ctrlbox_digital_output
  {
    if (msg.ctrlbox_digital_output.size() == 0) {
      out << "ctrlbox_digital_output: []";
    } else {
      out << "ctrlbox_digital_output: [";
      size_t pending_items = msg.ctrlbox_digital_output.size();
      for (auto item : msg.ctrlbox_digital_output) {
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
    if (msg.flange_digital_input.size() == 0) {
      out << "flange_digital_input: []";
    } else {
      out << "flange_digital_input: [";
      size_t pending_items = msg.flange_digital_input.size();
      for (auto item : msg.flange_digital_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: flange_digital_output
  {
    if (msg.flange_digital_output.size() == 0) {
      out << "flange_digital_output: []";
    } else {
      out << "flange_digital_output: [";
      size_t pending_items = msg.flange_digital_output.size();
      for (auto item : msg.flange_digital_output) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: modbus_state
  {
    if (msg.modbus_state.size() == 0) {
      out << "modbus_state: []";
    } else {
      out << "modbus_state: [";
      size_t pending_items = msg.modbus_state.size();
      for (auto item : msg.modbus_state) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: access_control
  {
    out << "access_control: ";
    rosidl_generator_traits::value_to_yaml(msg.access_control, out);
    out << ", ";
  }

  // member: homming_completed
  {
    out << "homming_completed: ";
    rosidl_generator_traits::value_to_yaml(msg.homming_completed, out);
    out << ", ";
  }

  // member: tp_initialized
  {
    out << "tp_initialized: ";
    rosidl_generator_traits::value_to_yaml(msg.tp_initialized, out);
    out << ", ";
  }

  // member: mastering_need
  {
    out << "mastering_need: ";
    rosidl_generator_traits::value_to_yaml(msg.mastering_need, out);
    out << ", ";
  }

  // member: drl_stopped
  {
    out << "drl_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.drl_stopped, out);
    out << ", ";
  }

  // member: disconnected
  {
    out << "disconnected: ";
    rosidl_generator_traits::value_to_yaml(msg.disconnected, out);
    out << ", ";
  }

  // member: f_actual_w2b
  {
    if (msg.f_actual_w2b.size() == 0) {
      out << "f_actual_w2b: []";
    } else {
      out << "f_actual_w2b: [";
      size_t pending_items = msg.f_actual_w2b.size();
      for (auto item : msg.f_actual_w2b) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_current_pos_world
  {
    if (msg.f_current_pos_world.size() == 0) {
      out << "f_current_pos_world: []";
    } else {
      out << "f_current_pos_world: [";
      size_t pending_items = msg.f_current_pos_world.size();
      for (auto item : msg.f_current_pos_world) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_current_vel_world
  {
    if (msg.f_current_vel_world.size() == 0) {
      out << "f_current_vel_world: []";
    } else {
      out << "f_current_vel_world: [";
      size_t pending_items = msg.f_current_vel_world.size();
      for (auto item : msg.f_current_vel_world) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_world_ext_target_torque
  {
    if (msg.f_world_ext_target_torque.size() == 0) {
      out << "f_world_ext_target_torque: []";
    } else {
      out << "f_world_ext_target_torque: [";
      size_t pending_items = msg.f_world_ext_target_torque.size();
      for (auto item : msg.f_world_ext_target_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_target_pos_world
  {
    if (msg.f_target_pos_world.size() == 0) {
      out << "f_target_pos_world: []";
    } else {
      out << "f_target_pos_world: [";
      size_t pending_items = msg.f_target_pos_world.size();
      for (auto item : msg.f_target_pos_world) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_target_vel_world
  {
    if (msg.f_target_vel_world.size() == 0) {
      out << "f_target_vel_world: []";
    } else {
      out << "f_target_vel_world: [";
      size_t pending_items = msg.f_target_vel_world.size();
      for (auto item : msg.f_target_vel_world) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_rotation_matrix_world
  {
    if (msg.f_rotation_matrix_world.size() == 0) {
      out << "f_rotation_matrix_world: []";
    } else {
      out << "f_rotation_matrix_world: [";
      size_t pending_items = msg.f_rotation_matrix_world.size();
      for (auto item : msg.f_rotation_matrix_world) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_actual_user_coord_num
  {
    out << "i_actual_user_coord_num: ";
    rosidl_generator_traits::value_to_yaml(msg.i_actual_user_coord_num, out);
    out << ", ";
  }

  // member: i_coord_ref
  {
    out << "i_coord_ref: ";
    rosidl_generator_traits::value_to_yaml(msg.i_coord_ref, out);
    out << ", ";
  }

  // member: f_current_pos_user
  {
    if (msg.f_current_pos_user.size() == 0) {
      out << "f_current_pos_user: []";
    } else {
      out << "f_current_pos_user: [";
      size_t pending_items = msg.f_current_pos_user.size();
      for (auto item : msg.f_current_pos_user) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_current_vel_user
  {
    if (msg.f_current_vel_user.size() == 0) {
      out << "f_current_vel_user: []";
    } else {
      out << "f_current_vel_user: [";
      size_t pending_items = msg.f_current_vel_user.size();
      for (auto item : msg.f_current_vel_user) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_user_ext_task_torque
  {
    if (msg.f_user_ext_task_torque.size() == 0) {
      out << "f_user_ext_task_torque: []";
    } else {
      out << "f_user_ext_task_torque: [";
      size_t pending_items = msg.f_user_ext_task_torque.size();
      for (auto item : msg.f_user_ext_task_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_target_pos_user
  {
    if (msg.f_target_pos_user.size() == 0) {
      out << "f_target_pos_user: []";
    } else {
      out << "f_target_pos_user: [";
      size_t pending_items = msg.f_target_pos_user.size();
      for (auto item : msg.f_target_pos_user) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_target_vel_user
  {
    if (msg.f_target_vel_user.size() == 0) {
      out << "f_target_vel_user: []";
    } else {
      out << "f_target_vel_user: [";
      size_t pending_items = msg.f_target_vel_user.size();
      for (auto item : msg.f_target_vel_user) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_rotation_matrix_user
  {
    if (msg.f_rotation_matrix_user.size() == 0) {
      out << "f_rotation_matrix_user: []";
    } else {
      out << "f_rotation_matrix_user: [";
      size_t pending_items = msg.f_rotation_matrix_user.size();
      for (auto item : msg.f_rotation_matrix_user) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_actual_analog_input
  {
    if (msg.f_actual_analog_input.size() == 0) {
      out << "f_actual_analog_input: []";
    } else {
      out << "f_actual_analog_input: [";
      size_t pending_items = msg.f_actual_analog_input.size();
      for (auto item : msg.f_actual_analog_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: b_actual_switch_input
  {
    if (msg.b_actual_switch_input.size() == 0) {
      out << "b_actual_switch_input: []";
    } else {
      out << "b_actual_switch_input: [";
      size_t pending_items = msg.b_actual_switch_input.size();
      for (auto item : msg.b_actual_switch_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: b_actual_safety_input
  {
    if (msg.b_actual_safety_input.size() == 0) {
      out << "b_actual_safety_input: []";
    } else {
      out << "b_actual_safety_input: [";
      size_t pending_items = msg.b_actual_safety_input.size();
      for (auto item : msg.b_actual_safety_input) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_actual_analog_input_type
  {
    if (msg.i_actual_analog_input_type.size() == 0) {
      out << "i_actual_analog_input_type: []";
    } else {
      out << "i_actual_analog_input_type: [";
      size_t pending_items = msg.i_actual_analog_input_type.size();
      for (auto item : msg.i_actual_analog_input_type) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: f_target_analog_output
  {
    if (msg.f_target_analog_output.size() == 0) {
      out << "f_target_analog_output: []";
    } else {
      out << "f_target_analog_output: [";
      size_t pending_items = msg.f_target_analog_output.size();
      for (auto item : msg.f_target_analog_output) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_target_analog_output_type
  {
    if (msg.i_target_analog_output_type.size() == 0) {
      out << "i_target_analog_output_type: []";
    } else {
      out << "i_target_analog_output_type: [";
      size_t pending_items = msg.i_target_analog_output_type.size();
      for (auto item : msg.i_target_analog_output_type) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: b_actual_encorder_strove_signal
  {
    if (msg.b_actual_encorder_strove_signal.size() == 0) {
      out << "b_actual_encorder_strove_signal: []";
    } else {
      out << "b_actual_encorder_strove_signal: [";
      size_t pending_items = msg.b_actual_encorder_strove_signal.size();
      for (auto item : msg.b_actual_encorder_strove_signal) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_actual_encorder_raw_data
  {
    if (msg.i_actual_encorder_raw_data.size() == 0) {
      out << "i_actual_encorder_raw_data: []";
    } else {
      out << "i_actual_encorder_raw_data: [";
      size_t pending_items = msg.i_actual_encorder_raw_data.size();
      for (auto item : msg.i_actual_encorder_raw_data) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: b_actual_encorder_reset_signal
  {
    if (msg.b_actual_encorder_reset_signal.size() == 0) {
      out << "b_actual_encorder_reset_signal: []";
    } else {
      out << "b_actual_encorder_reset_signal: [";
      size_t pending_items = msg.b_actual_encorder_reset_signal.size();
      for (auto item : msg.b_actual_encorder_reset_signal) {
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
  const RobotState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: robot_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_state: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state, out);
    out << "\n";
  }

  // member: robot_state_str
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_state_str: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_state_str, out);
    out << "\n";
  }

  // member: actual_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "actual_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_mode, out);
    out << "\n";
  }

  // member: actual_space
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "actual_space: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_space, out);
    out << "\n";
  }

  // member: current_posj
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.current_posj.size() == 0) {
      out << "current_posj: []\n";
    } else {
      out << "current_posj:\n";
      for (auto item : msg.current_posj) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: current_velj
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.current_velj.size() == 0) {
      out << "current_velj: []\n";
    } else {
      out << "current_velj:\n";
      for (auto item : msg.current_velj) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_abs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_abs.size() == 0) {
      out << "joint_abs: []\n";
    } else {
      out << "joint_abs:\n";
      for (auto item : msg.joint_abs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_err.size() == 0) {
      out << "joint_err: []\n";
    } else {
      out << "joint_err:\n";
      for (auto item : msg.joint_err) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_posj
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_posj.size() == 0) {
      out << "target_posj: []\n";
    } else {
      out << "target_posj:\n";
      for (auto item : msg.target_posj) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_velj
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_velj.size() == 0) {
      out << "target_velj: []\n";
    } else {
      out << "target_velj:\n";
      for (auto item : msg.target_velj) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: current_posx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.current_posx.size() == 0) {
      out << "current_posx: []\n";
    } else {
      out << "current_posx:\n";
      for (auto item : msg.current_posx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: current_tool_posx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.current_tool_posx.size() == 0) {
      out << "current_tool_posx: []\n";
    } else {
      out << "current_tool_posx:\n";
      for (auto item : msg.current_tool_posx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: current_velx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.current_velx.size() == 0) {
      out << "current_velx: []\n";
    } else {
      out << "current_velx:\n";
      for (auto item : msg.current_velx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: task_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.task_err.size() == 0) {
      out << "task_err: []\n";
    } else {
      out << "task_err:\n";
      for (auto item : msg.task_err) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_velx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_velx.size() == 0) {
      out << "target_velx: []\n";
    } else {
      out << "target_velx:\n";
      for (auto item : msg.target_velx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: target_posx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_posx.size() == 0) {
      out << "target_posx: []\n";
    } else {
      out << "target_posx:\n";
      for (auto item : msg.target_posx) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: dynamic_tor
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dynamic_tor.size() == 0) {
      out << "dynamic_tor: []\n";
    } else {
      out << "dynamic_tor:\n";
      for (auto item : msg.dynamic_tor) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_jts
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_jts.size() == 0) {
      out << "actual_jts: []\n";
    } else {
      out << "actual_jts:\n";
      for (auto item : msg.actual_jts) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_ejt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_ejt.size() == 0) {
      out << "actual_ejt: []\n";
    } else {
      out << "actual_ejt:\n";
      for (auto item : msg.actual_ejt) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_ett
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_ett.size() == 0) {
      out << "actual_ett: []\n";
    } else {
      out << "actual_ett:\n";
      for (auto item : msg.actual_ett) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_bk
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_bk.size() == 0) {
      out << "actual_bk: []\n";
    } else {
      out << "actual_bk:\n";
      for (auto item : msg.actual_bk) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_mc
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_mc.size() == 0) {
      out << "actual_mc: []\n";
    } else {
      out << "actual_mc:\n";
      for (auto item : msg.actual_mc) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: actual_mt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_mt.size() == 0) {
      out << "actual_mt: []\n";
    } else {
      out << "actual_mt:\n";
      for (auto item : msg.actual_mt) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
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

  // member: sync_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sync_time: ";
    rosidl_generator_traits::value_to_yaml(msg.sync_time, out);
    out << "\n";
  }

  // member: actual_bt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.actual_bt.size() == 0) {
      out << "actual_bt: []\n";
    } else {
      out << "actual_bt:\n";
      for (auto item : msg.actual_bt) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: rotation_matrix
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.rotation_matrix.size() == 0) {
      out << "rotation_matrix: []\n";
    } else {
      out << "rotation_matrix:\n";
      for (auto item : msg.rotation_matrix) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: ctrlbox_digital_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ctrlbox_digital_input.size() == 0) {
      out << "ctrlbox_digital_input: []\n";
    } else {
      out << "ctrlbox_digital_input:\n";
      for (auto item : msg.ctrlbox_digital_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ctrlbox_digital_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ctrlbox_digital_output.size() == 0) {
      out << "ctrlbox_digital_output: []\n";
    } else {
      out << "ctrlbox_digital_output:\n";
      for (auto item : msg.ctrlbox_digital_output) {
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
    if (msg.flange_digital_input.size() == 0) {
      out << "flange_digital_input: []\n";
    } else {
      out << "flange_digital_input:\n";
      for (auto item : msg.flange_digital_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: flange_digital_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.flange_digital_output.size() == 0) {
      out << "flange_digital_output: []\n";
    } else {
      out << "flange_digital_output:\n";
      for (auto item : msg.flange_digital_output) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: modbus_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.modbus_state.size() == 0) {
      out << "modbus_state: []\n";
    } else {
      out << "modbus_state:\n";
      for (auto item : msg.modbus_state) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: access_control
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "access_control: ";
    rosidl_generator_traits::value_to_yaml(msg.access_control, out);
    out << "\n";
  }

  // member: homming_completed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "homming_completed: ";
    rosidl_generator_traits::value_to_yaml(msg.homming_completed, out);
    out << "\n";
  }

  // member: tp_initialized
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tp_initialized: ";
    rosidl_generator_traits::value_to_yaml(msg.tp_initialized, out);
    out << "\n";
  }

  // member: mastering_need
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mastering_need: ";
    rosidl_generator_traits::value_to_yaml(msg.mastering_need, out);
    out << "\n";
  }

  // member: drl_stopped
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "drl_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.drl_stopped, out);
    out << "\n";
  }

  // member: disconnected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "disconnected: ";
    rosidl_generator_traits::value_to_yaml(msg.disconnected, out);
    out << "\n";
  }

  // member: f_actual_w2b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_actual_w2b.size() == 0) {
      out << "f_actual_w2b: []\n";
    } else {
      out << "f_actual_w2b:\n";
      for (auto item : msg.f_actual_w2b) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_current_pos_world
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_current_pos_world.size() == 0) {
      out << "f_current_pos_world: []\n";
    } else {
      out << "f_current_pos_world:\n";
      for (auto item : msg.f_current_pos_world) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: f_current_vel_world
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_current_vel_world.size() == 0) {
      out << "f_current_vel_world: []\n";
    } else {
      out << "f_current_vel_world:\n";
      for (auto item : msg.f_current_vel_world) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_world_ext_target_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_world_ext_target_torque.size() == 0) {
      out << "f_world_ext_target_torque: []\n";
    } else {
      out << "f_world_ext_target_torque:\n";
      for (auto item : msg.f_world_ext_target_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_target_pos_world
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_target_pos_world.size() == 0) {
      out << "f_target_pos_world: []\n";
    } else {
      out << "f_target_pos_world:\n";
      for (auto item : msg.f_target_pos_world) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_target_vel_world
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_target_vel_world.size() == 0) {
      out << "f_target_vel_world: []\n";
    } else {
      out << "f_target_vel_world:\n";
      for (auto item : msg.f_target_vel_world) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_rotation_matrix_world
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_rotation_matrix_world.size() == 0) {
      out << "f_rotation_matrix_world: []\n";
    } else {
      out << "f_rotation_matrix_world:\n";
      for (auto item : msg.f_rotation_matrix_world) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: i_actual_user_coord_num
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "i_actual_user_coord_num: ";
    rosidl_generator_traits::value_to_yaml(msg.i_actual_user_coord_num, out);
    out << "\n";
  }

  // member: i_coord_ref
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "i_coord_ref: ";
    rosidl_generator_traits::value_to_yaml(msg.i_coord_ref, out);
    out << "\n";
  }

  // member: f_current_pos_user
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_current_pos_user.size() == 0) {
      out << "f_current_pos_user: []\n";
    } else {
      out << "f_current_pos_user:\n";
      for (auto item : msg.f_current_pos_user) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: f_current_vel_user
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_current_vel_user.size() == 0) {
      out << "f_current_vel_user: []\n";
    } else {
      out << "f_current_vel_user:\n";
      for (auto item : msg.f_current_vel_user) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_user_ext_task_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_user_ext_task_torque.size() == 0) {
      out << "f_user_ext_task_torque: []\n";
    } else {
      out << "f_user_ext_task_torque:\n";
      for (auto item : msg.f_user_ext_task_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_target_pos_user
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_target_pos_user.size() == 0) {
      out << "f_target_pos_user: []\n";
    } else {
      out << "f_target_pos_user:\n";
      for (auto item : msg.f_target_pos_user) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_target_vel_user
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_target_vel_user.size() == 0) {
      out << "f_target_vel_user: []\n";
    } else {
      out << "f_target_vel_user:\n";
      for (auto item : msg.f_target_vel_user) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_rotation_matrix_user
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_rotation_matrix_user.size() == 0) {
      out << "f_rotation_matrix_user: []\n";
    } else {
      out << "f_rotation_matrix_user:\n";
      for (auto item : msg.f_rotation_matrix_user) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: f_actual_analog_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_actual_analog_input.size() == 0) {
      out << "f_actual_analog_input: []\n";
    } else {
      out << "f_actual_analog_input:\n";
      for (auto item : msg.f_actual_analog_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: b_actual_switch_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.b_actual_switch_input.size() == 0) {
      out << "b_actual_switch_input: []\n";
    } else {
      out << "b_actual_switch_input:\n";
      for (auto item : msg.b_actual_switch_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: b_actual_safety_input
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.b_actual_safety_input.size() == 0) {
      out << "b_actual_safety_input: []\n";
    } else {
      out << "b_actual_safety_input:\n";
      for (auto item : msg.b_actual_safety_input) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: i_actual_analog_input_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.i_actual_analog_input_type.size() == 0) {
      out << "i_actual_analog_input_type: []\n";
    } else {
      out << "i_actual_analog_input_type:\n";
      for (auto item : msg.i_actual_analog_input_type) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: f_target_analog_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.f_target_analog_output.size() == 0) {
      out << "f_target_analog_output: []\n";
    } else {
      out << "f_target_analog_output:\n";
      for (auto item : msg.f_target_analog_output) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: i_target_analog_output_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.i_target_analog_output_type.size() == 0) {
      out << "i_target_analog_output_type: []\n";
    } else {
      out << "i_target_analog_output_type:\n";
      for (auto item : msg.i_target_analog_output_type) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: b_actual_encorder_strove_signal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.b_actual_encorder_strove_signal.size() == 0) {
      out << "b_actual_encorder_strove_signal: []\n";
    } else {
      out << "b_actual_encorder_strove_signal:\n";
      for (auto item : msg.b_actual_encorder_strove_signal) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: i_actual_encorder_raw_data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.i_actual_encorder_raw_data.size() == 0) {
      out << "i_actual_encorder_raw_data: []\n";
    } else {
      out << "i_actual_encorder_raw_data:\n";
      for (auto item : msg.i_actual_encorder_raw_data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: b_actual_encorder_reset_signal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.b_actual_encorder_reset_signal.size() == 0) {
      out << "b_actual_encorder_reset_signal: []\n";
    } else {
      out << "b_actual_encorder_reset_signal:\n";
      for (auto item : msg.b_actual_encorder_reset_signal) {
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

inline std::string to_yaml(const RobotState & msg, bool use_flow_style = false)
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
  const dsr_msgs2::msg::RobotState & msg,
  std::ostream & out, size_t indentation = 0)
{
  dsr_msgs2::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dsr_msgs2::msg::to_yaml() instead")]]
inline std::string to_yaml(const dsr_msgs2::msg::RobotState & msg)
{
  return dsr_msgs2::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dsr_msgs2::msg::RobotState>()
{
  return "dsr_msgs2::msg::RobotState";
}

template<>
inline const char * name<dsr_msgs2::msg::RobotState>()
{
  return "dsr_msgs2/msg/RobotState";
}

template<>
struct has_fixed_size<dsr_msgs2::msg::RobotState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dsr_msgs2::msg::RobotState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dsr_msgs2::msg::RobotState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__TRAITS_HPP_
