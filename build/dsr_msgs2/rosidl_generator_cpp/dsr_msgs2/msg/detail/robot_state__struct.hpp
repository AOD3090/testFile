// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'rotation_matrix'
// Member 'f_current_pos_world'
// Member 'f_rotation_matrix_world'
// Member 'f_current_pos_user'
// Member 'f_rotation_matrix_user'
#include "std_msgs/msg/detail/float64_multi_array__struct.hpp"
// Member 'modbus_state'
#include "dsr_msgs2/msg/detail/modbus_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__RobotState __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__RobotState __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotState_
{
  using Type = RobotState_<ContainerAllocator>;

  explicit RobotState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_state = 0l;
      this->robot_state_str = "";
      this->actual_mode = 0;
      this->actual_space = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_posj.begin(), this->current_posj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_velj.begin(), this->current_velj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_abs.begin(), this->joint_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_err.begin(), this->joint_err.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_posj.begin(), this->target_posj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_velj.begin(), this->target_velj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_posx.begin(), this->current_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_tool_posx.begin(), this->current_tool_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_velx.begin(), this->current_velx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->task_err.begin(), this->task_err.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_velx.begin(), this->target_velx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_posx.begin(), this->target_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->dynamic_tor.begin(), this->dynamic_tor.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_jts.begin(), this->actual_jts.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_ejt.begin(), this->actual_ejt.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_ett.begin(), this->actual_ett.end(), 0.0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->actual_bk.begin(), this->actual_bk.end(), 0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_mc.begin(), this->actual_mc.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_mt.begin(), this->actual_mt.end(), 0.0);
      this->solution_space = 0;
      this->sync_time = 0.0;
      std::fill<typename std::array<int8_t, 5>::iterator, int8_t>(this->actual_bt.begin(), this->actual_bt.end(), 0);
      std::fill<typename std::array<int8_t, 16>::iterator, int8_t>(this->ctrlbox_digital_input.begin(), this->ctrlbox_digital_input.end(), 0);
      std::fill<typename std::array<int8_t, 16>::iterator, int8_t>(this->ctrlbox_digital_output.begin(), this->ctrlbox_digital_output.end(), 0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->flange_digital_input.begin(), this->flange_digital_input.end(), 0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->flange_digital_output.begin(), this->flange_digital_output.end(), 0);
      this->access_control = 0l;
      this->homming_completed = false;
      this->tp_initialized = false;
      this->mastering_need = 0;
      this->drl_stopped = false;
      this->disconnected = false;
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_actual_w2b.begin(), this->f_actual_w2b.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_current_vel_world.begin(), this->f_current_vel_world.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_world_ext_target_torque.begin(), this->f_world_ext_target_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_pos_world.begin(), this->f_target_pos_world.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_vel_world.begin(), this->f_target_vel_world.end(), 0.0);
      this->i_actual_user_coord_num = 0;
      this->i_coord_ref = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_current_vel_user.begin(), this->f_current_vel_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_user_ext_task_torque.begin(), this->f_user_ext_task_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_pos_user.begin(), this->f_target_pos_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_vel_user.begin(), this->f_target_vel_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_actual_analog_input.begin(), this->f_actual_analog_input.end(), 0.0);
      std::fill<typename std::array<bool, 3>::iterator, bool>(this->b_actual_switch_input.begin(), this->b_actual_switch_input.end(), false);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_safety_input.begin(), this->b_actual_safety_input.end(), false);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_actual_analog_input_type.begin(), this->i_actual_analog_input_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->f_target_analog_output.begin(), this->f_target_analog_output.end(), 0.0);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_target_analog_output_type.begin(), this->i_target_analog_output_type.end(), 0);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_encorder_strove_signal.begin(), this->b_actual_encorder_strove_signal.end(), false);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_actual_encorder_raw_data.begin(), this->i_actual_encorder_raw_data.end(), 0);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_encorder_reset_signal.begin(), this->b_actual_encorder_reset_signal.end(), false);
    }
  }

  explicit RobotState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : robot_state_str(_alloc),
    current_posj(_alloc),
    current_velj(_alloc),
    joint_abs(_alloc),
    joint_err(_alloc),
    target_posj(_alloc),
    target_velj(_alloc),
    current_posx(_alloc),
    current_tool_posx(_alloc),
    current_velx(_alloc),
    task_err(_alloc),
    target_velx(_alloc),
    target_posx(_alloc),
    dynamic_tor(_alloc),
    actual_jts(_alloc),
    actual_ejt(_alloc),
    actual_ett(_alloc),
    actual_bk(_alloc),
    actual_mc(_alloc),
    actual_mt(_alloc),
    actual_bt(_alloc),
    ctrlbox_digital_input(_alloc),
    ctrlbox_digital_output(_alloc),
    flange_digital_input(_alloc),
    flange_digital_output(_alloc),
    f_actual_w2b(_alloc),
    f_current_vel_world(_alloc),
    f_world_ext_target_torque(_alloc),
    f_target_pos_world(_alloc),
    f_target_vel_world(_alloc),
    f_current_vel_user(_alloc),
    f_user_ext_task_torque(_alloc),
    f_target_pos_user(_alloc),
    f_target_vel_user(_alloc),
    f_actual_analog_input(_alloc),
    b_actual_switch_input(_alloc),
    b_actual_safety_input(_alloc),
    i_actual_analog_input_type(_alloc),
    f_target_analog_output(_alloc),
    i_target_analog_output_type(_alloc),
    b_actual_encorder_strove_signal(_alloc),
    i_actual_encorder_raw_data(_alloc),
    b_actual_encorder_reset_signal(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_state = 0l;
      this->robot_state_str = "";
      this->actual_mode = 0;
      this->actual_space = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_posj.begin(), this->current_posj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_velj.begin(), this->current_velj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_abs.begin(), this->joint_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_err.begin(), this->joint_err.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_posj.begin(), this->target_posj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_velj.begin(), this->target_velj.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_posx.begin(), this->current_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_tool_posx.begin(), this->current_tool_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->current_velx.begin(), this->current_velx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->task_err.begin(), this->task_err.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_velx.begin(), this->target_velx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_posx.begin(), this->target_posx.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->dynamic_tor.begin(), this->dynamic_tor.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_jts.begin(), this->actual_jts.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_ejt.begin(), this->actual_ejt.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_ett.begin(), this->actual_ett.end(), 0.0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->actual_bk.begin(), this->actual_bk.end(), 0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_mc.begin(), this->actual_mc.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_mt.begin(), this->actual_mt.end(), 0.0);
      this->solution_space = 0;
      this->sync_time = 0.0;
      std::fill<typename std::array<int8_t, 5>::iterator, int8_t>(this->actual_bt.begin(), this->actual_bt.end(), 0);
      std::fill<typename std::array<int8_t, 16>::iterator, int8_t>(this->ctrlbox_digital_input.begin(), this->ctrlbox_digital_input.end(), 0);
      std::fill<typename std::array<int8_t, 16>::iterator, int8_t>(this->ctrlbox_digital_output.begin(), this->ctrlbox_digital_output.end(), 0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->flange_digital_input.begin(), this->flange_digital_input.end(), 0);
      std::fill<typename std::array<int8_t, 6>::iterator, int8_t>(this->flange_digital_output.begin(), this->flange_digital_output.end(), 0);
      this->access_control = 0l;
      this->homming_completed = false;
      this->tp_initialized = false;
      this->mastering_need = 0;
      this->drl_stopped = false;
      this->disconnected = false;
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_actual_w2b.begin(), this->f_actual_w2b.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_current_vel_world.begin(), this->f_current_vel_world.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_world_ext_target_torque.begin(), this->f_world_ext_target_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_pos_world.begin(), this->f_target_pos_world.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_vel_world.begin(), this->f_target_vel_world.end(), 0.0);
      this->i_actual_user_coord_num = 0;
      this->i_coord_ref = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_current_vel_user.begin(), this->f_current_vel_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_user_ext_task_torque.begin(), this->f_user_ext_task_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_pos_user.begin(), this->f_target_pos_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_target_vel_user.begin(), this->f_target_vel_user.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->f_actual_analog_input.begin(), this->f_actual_analog_input.end(), 0.0);
      std::fill<typename std::array<bool, 3>::iterator, bool>(this->b_actual_switch_input.begin(), this->b_actual_switch_input.end(), false);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_safety_input.begin(), this->b_actual_safety_input.end(), false);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_actual_analog_input_type.begin(), this->i_actual_analog_input_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->f_target_analog_output.begin(), this->f_target_analog_output.end(), 0.0);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_target_analog_output_type.begin(), this->i_target_analog_output_type.end(), 0);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_encorder_strove_signal.begin(), this->b_actual_encorder_strove_signal.end(), false);
      std::fill<typename std::array<int8_t, 2>::iterator, int8_t>(this->i_actual_encorder_raw_data.begin(), this->i_actual_encorder_raw_data.end(), 0);
      std::fill<typename std::array<bool, 2>::iterator, bool>(this->b_actual_encorder_reset_signal.begin(), this->b_actual_encorder_reset_signal.end(), false);
    }
  }

  // field types and members
  using _robot_state_type =
    int32_t;
  _robot_state_type robot_state;
  using _robot_state_str_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _robot_state_str_type robot_state_str;
  using _actual_mode_type =
    int8_t;
  _actual_mode_type actual_mode;
  using _actual_space_type =
    int8_t;
  _actual_space_type actual_space;
  using _current_posj_type =
    std::array<double, 6>;
  _current_posj_type current_posj;
  using _current_velj_type =
    std::array<double, 6>;
  _current_velj_type current_velj;
  using _joint_abs_type =
    std::array<double, 6>;
  _joint_abs_type joint_abs;
  using _joint_err_type =
    std::array<double, 6>;
  _joint_err_type joint_err;
  using _target_posj_type =
    std::array<double, 6>;
  _target_posj_type target_posj;
  using _target_velj_type =
    std::array<double, 6>;
  _target_velj_type target_velj;
  using _current_posx_type =
    std::array<double, 6>;
  _current_posx_type current_posx;
  using _current_tool_posx_type =
    std::array<double, 6>;
  _current_tool_posx_type current_tool_posx;
  using _current_velx_type =
    std::array<double, 6>;
  _current_velx_type current_velx;
  using _task_err_type =
    std::array<double, 6>;
  _task_err_type task_err;
  using _target_velx_type =
    std::array<double, 6>;
  _target_velx_type target_velx;
  using _target_posx_type =
    std::array<double, 6>;
  _target_posx_type target_posx;
  using _dynamic_tor_type =
    std::array<double, 6>;
  _dynamic_tor_type dynamic_tor;
  using _actual_jts_type =
    std::array<double, 6>;
  _actual_jts_type actual_jts;
  using _actual_ejt_type =
    std::array<double, 6>;
  _actual_ejt_type actual_ejt;
  using _actual_ett_type =
    std::array<double, 6>;
  _actual_ett_type actual_ett;
  using _actual_bk_type =
    std::array<int8_t, 6>;
  _actual_bk_type actual_bk;
  using _actual_mc_type =
    std::array<double, 6>;
  _actual_mc_type actual_mc;
  using _actual_mt_type =
    std::array<double, 6>;
  _actual_mt_type actual_mt;
  using _solution_space_type =
    int8_t;
  _solution_space_type solution_space;
  using _sync_time_type =
    double;
  _sync_time_type sync_time;
  using _actual_bt_type =
    std::array<int8_t, 5>;
  _actual_bt_type actual_bt;
  using _rotation_matrix_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _rotation_matrix_type rotation_matrix;
  using _ctrlbox_digital_input_type =
    std::array<int8_t, 16>;
  _ctrlbox_digital_input_type ctrlbox_digital_input;
  using _ctrlbox_digital_output_type =
    std::array<int8_t, 16>;
  _ctrlbox_digital_output_type ctrlbox_digital_output;
  using _flange_digital_input_type =
    std::array<int8_t, 6>;
  _flange_digital_input_type flange_digital_input;
  using _flange_digital_output_type =
    std::array<int8_t, 6>;
  _flange_digital_output_type flange_digital_output;
  using _modbus_state_type =
    std::vector<dsr_msgs2::msg::ModbusState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<dsr_msgs2::msg::ModbusState_<ContainerAllocator>>>;
  _modbus_state_type modbus_state;
  using _access_control_type =
    int32_t;
  _access_control_type access_control;
  using _homming_completed_type =
    bool;
  _homming_completed_type homming_completed;
  using _tp_initialized_type =
    bool;
  _tp_initialized_type tp_initialized;
  using _mastering_need_type =
    int8_t;
  _mastering_need_type mastering_need;
  using _drl_stopped_type =
    bool;
  _drl_stopped_type drl_stopped;
  using _disconnected_type =
    bool;
  _disconnected_type disconnected;
  using _f_actual_w2b_type =
    std::array<double, 6>;
  _f_actual_w2b_type f_actual_w2b;
  using _f_current_pos_world_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _f_current_pos_world_type f_current_pos_world;
  using _f_current_vel_world_type =
    std::array<double, 6>;
  _f_current_vel_world_type f_current_vel_world;
  using _f_world_ext_target_torque_type =
    std::array<double, 6>;
  _f_world_ext_target_torque_type f_world_ext_target_torque;
  using _f_target_pos_world_type =
    std::array<double, 6>;
  _f_target_pos_world_type f_target_pos_world;
  using _f_target_vel_world_type =
    std::array<double, 6>;
  _f_target_vel_world_type f_target_vel_world;
  using _f_rotation_matrix_world_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _f_rotation_matrix_world_type f_rotation_matrix_world;
  using _i_actual_user_coord_num_type =
    int8_t;
  _i_actual_user_coord_num_type i_actual_user_coord_num;
  using _i_coord_ref_type =
    int8_t;
  _i_coord_ref_type i_coord_ref;
  using _f_current_pos_user_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _f_current_pos_user_type f_current_pos_user;
  using _f_current_vel_user_type =
    std::array<double, 6>;
  _f_current_vel_user_type f_current_vel_user;
  using _f_user_ext_task_torque_type =
    std::array<double, 6>;
  _f_user_ext_task_torque_type f_user_ext_task_torque;
  using _f_target_pos_user_type =
    std::array<double, 6>;
  _f_target_pos_user_type f_target_pos_user;
  using _f_target_vel_user_type =
    std::array<double, 6>;
  _f_target_vel_user_type f_target_vel_user;
  using _f_rotation_matrix_user_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _f_rotation_matrix_user_type f_rotation_matrix_user;
  using _f_actual_analog_input_type =
    std::array<double, 6>;
  _f_actual_analog_input_type f_actual_analog_input;
  using _b_actual_switch_input_type =
    std::array<bool, 3>;
  _b_actual_switch_input_type b_actual_switch_input;
  using _b_actual_safety_input_type =
    std::array<bool, 2>;
  _b_actual_safety_input_type b_actual_safety_input;
  using _i_actual_analog_input_type_type =
    std::array<int8_t, 2>;
  _i_actual_analog_input_type_type i_actual_analog_input_type;
  using _f_target_analog_output_type =
    std::array<double, 2>;
  _f_target_analog_output_type f_target_analog_output;
  using _i_target_analog_output_type_type =
    std::array<int8_t, 2>;
  _i_target_analog_output_type_type i_target_analog_output_type;
  using _b_actual_encorder_strove_signal_type =
    std::array<bool, 2>;
  _b_actual_encorder_strove_signal_type b_actual_encorder_strove_signal;
  using _i_actual_encorder_raw_data_type =
    std::array<int8_t, 2>;
  _i_actual_encorder_raw_data_type i_actual_encorder_raw_data;
  using _b_actual_encorder_reset_signal_type =
    std::array<bool, 2>;
  _b_actual_encorder_reset_signal_type b_actual_encorder_reset_signal;

  // setters for named parameter idiom
  Type & set__robot_state(
    const int32_t & _arg)
  {
    this->robot_state = _arg;
    return *this;
  }
  Type & set__robot_state_str(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->robot_state_str = _arg;
    return *this;
  }
  Type & set__actual_mode(
    const int8_t & _arg)
  {
    this->actual_mode = _arg;
    return *this;
  }
  Type & set__actual_space(
    const int8_t & _arg)
  {
    this->actual_space = _arg;
    return *this;
  }
  Type & set__current_posj(
    const std::array<double, 6> & _arg)
  {
    this->current_posj = _arg;
    return *this;
  }
  Type & set__current_velj(
    const std::array<double, 6> & _arg)
  {
    this->current_velj = _arg;
    return *this;
  }
  Type & set__joint_abs(
    const std::array<double, 6> & _arg)
  {
    this->joint_abs = _arg;
    return *this;
  }
  Type & set__joint_err(
    const std::array<double, 6> & _arg)
  {
    this->joint_err = _arg;
    return *this;
  }
  Type & set__target_posj(
    const std::array<double, 6> & _arg)
  {
    this->target_posj = _arg;
    return *this;
  }
  Type & set__target_velj(
    const std::array<double, 6> & _arg)
  {
    this->target_velj = _arg;
    return *this;
  }
  Type & set__current_posx(
    const std::array<double, 6> & _arg)
  {
    this->current_posx = _arg;
    return *this;
  }
  Type & set__current_tool_posx(
    const std::array<double, 6> & _arg)
  {
    this->current_tool_posx = _arg;
    return *this;
  }
  Type & set__current_velx(
    const std::array<double, 6> & _arg)
  {
    this->current_velx = _arg;
    return *this;
  }
  Type & set__task_err(
    const std::array<double, 6> & _arg)
  {
    this->task_err = _arg;
    return *this;
  }
  Type & set__target_velx(
    const std::array<double, 6> & _arg)
  {
    this->target_velx = _arg;
    return *this;
  }
  Type & set__target_posx(
    const std::array<double, 6> & _arg)
  {
    this->target_posx = _arg;
    return *this;
  }
  Type & set__dynamic_tor(
    const std::array<double, 6> & _arg)
  {
    this->dynamic_tor = _arg;
    return *this;
  }
  Type & set__actual_jts(
    const std::array<double, 6> & _arg)
  {
    this->actual_jts = _arg;
    return *this;
  }
  Type & set__actual_ejt(
    const std::array<double, 6> & _arg)
  {
    this->actual_ejt = _arg;
    return *this;
  }
  Type & set__actual_ett(
    const std::array<double, 6> & _arg)
  {
    this->actual_ett = _arg;
    return *this;
  }
  Type & set__actual_bk(
    const std::array<int8_t, 6> & _arg)
  {
    this->actual_bk = _arg;
    return *this;
  }
  Type & set__actual_mc(
    const std::array<double, 6> & _arg)
  {
    this->actual_mc = _arg;
    return *this;
  }
  Type & set__actual_mt(
    const std::array<double, 6> & _arg)
  {
    this->actual_mt = _arg;
    return *this;
  }
  Type & set__solution_space(
    const int8_t & _arg)
  {
    this->solution_space = _arg;
    return *this;
  }
  Type & set__sync_time(
    const double & _arg)
  {
    this->sync_time = _arg;
    return *this;
  }
  Type & set__actual_bt(
    const std::array<int8_t, 5> & _arg)
  {
    this->actual_bt = _arg;
    return *this;
  }
  Type & set__rotation_matrix(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->rotation_matrix = _arg;
    return *this;
  }
  Type & set__ctrlbox_digital_input(
    const std::array<int8_t, 16> & _arg)
  {
    this->ctrlbox_digital_input = _arg;
    return *this;
  }
  Type & set__ctrlbox_digital_output(
    const std::array<int8_t, 16> & _arg)
  {
    this->ctrlbox_digital_output = _arg;
    return *this;
  }
  Type & set__flange_digital_input(
    const std::array<int8_t, 6> & _arg)
  {
    this->flange_digital_input = _arg;
    return *this;
  }
  Type & set__flange_digital_output(
    const std::array<int8_t, 6> & _arg)
  {
    this->flange_digital_output = _arg;
    return *this;
  }
  Type & set__modbus_state(
    const std::vector<dsr_msgs2::msg::ModbusState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<dsr_msgs2::msg::ModbusState_<ContainerAllocator>>> & _arg)
  {
    this->modbus_state = _arg;
    return *this;
  }
  Type & set__access_control(
    const int32_t & _arg)
  {
    this->access_control = _arg;
    return *this;
  }
  Type & set__homming_completed(
    const bool & _arg)
  {
    this->homming_completed = _arg;
    return *this;
  }
  Type & set__tp_initialized(
    const bool & _arg)
  {
    this->tp_initialized = _arg;
    return *this;
  }
  Type & set__mastering_need(
    const int8_t & _arg)
  {
    this->mastering_need = _arg;
    return *this;
  }
  Type & set__drl_stopped(
    const bool & _arg)
  {
    this->drl_stopped = _arg;
    return *this;
  }
  Type & set__disconnected(
    const bool & _arg)
  {
    this->disconnected = _arg;
    return *this;
  }
  Type & set__f_actual_w2b(
    const std::array<double, 6> & _arg)
  {
    this->f_actual_w2b = _arg;
    return *this;
  }
  Type & set__f_current_pos_world(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->f_current_pos_world = _arg;
    return *this;
  }
  Type & set__f_current_vel_world(
    const std::array<double, 6> & _arg)
  {
    this->f_current_vel_world = _arg;
    return *this;
  }
  Type & set__f_world_ext_target_torque(
    const std::array<double, 6> & _arg)
  {
    this->f_world_ext_target_torque = _arg;
    return *this;
  }
  Type & set__f_target_pos_world(
    const std::array<double, 6> & _arg)
  {
    this->f_target_pos_world = _arg;
    return *this;
  }
  Type & set__f_target_vel_world(
    const std::array<double, 6> & _arg)
  {
    this->f_target_vel_world = _arg;
    return *this;
  }
  Type & set__f_rotation_matrix_world(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->f_rotation_matrix_world = _arg;
    return *this;
  }
  Type & set__i_actual_user_coord_num(
    const int8_t & _arg)
  {
    this->i_actual_user_coord_num = _arg;
    return *this;
  }
  Type & set__i_coord_ref(
    const int8_t & _arg)
  {
    this->i_coord_ref = _arg;
    return *this;
  }
  Type & set__f_current_pos_user(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->f_current_pos_user = _arg;
    return *this;
  }
  Type & set__f_current_vel_user(
    const std::array<double, 6> & _arg)
  {
    this->f_current_vel_user = _arg;
    return *this;
  }
  Type & set__f_user_ext_task_torque(
    const std::array<double, 6> & _arg)
  {
    this->f_user_ext_task_torque = _arg;
    return *this;
  }
  Type & set__f_target_pos_user(
    const std::array<double, 6> & _arg)
  {
    this->f_target_pos_user = _arg;
    return *this;
  }
  Type & set__f_target_vel_user(
    const std::array<double, 6> & _arg)
  {
    this->f_target_vel_user = _arg;
    return *this;
  }
  Type & set__f_rotation_matrix_user(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->f_rotation_matrix_user = _arg;
    return *this;
  }
  Type & set__f_actual_analog_input(
    const std::array<double, 6> & _arg)
  {
    this->f_actual_analog_input = _arg;
    return *this;
  }
  Type & set__b_actual_switch_input(
    const std::array<bool, 3> & _arg)
  {
    this->b_actual_switch_input = _arg;
    return *this;
  }
  Type & set__b_actual_safety_input(
    const std::array<bool, 2> & _arg)
  {
    this->b_actual_safety_input = _arg;
    return *this;
  }
  Type & set__i_actual_analog_input_type(
    const std::array<int8_t, 2> & _arg)
  {
    this->i_actual_analog_input_type = _arg;
    return *this;
  }
  Type & set__f_target_analog_output(
    const std::array<double, 2> & _arg)
  {
    this->f_target_analog_output = _arg;
    return *this;
  }
  Type & set__i_target_analog_output_type(
    const std::array<int8_t, 2> & _arg)
  {
    this->i_target_analog_output_type = _arg;
    return *this;
  }
  Type & set__b_actual_encorder_strove_signal(
    const std::array<bool, 2> & _arg)
  {
    this->b_actual_encorder_strove_signal = _arg;
    return *this;
  }
  Type & set__i_actual_encorder_raw_data(
    const std::array<int8_t, 2> & _arg)
  {
    this->i_actual_encorder_raw_data = _arg;
    return *this;
  }
  Type & set__b_actual_encorder_reset_signal(
    const std::array<bool, 2> & _arg)
  {
    this->b_actual_encorder_reset_signal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::RobotState_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::RobotState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__RobotState
    std::shared_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__RobotState
    std::shared_ptr<dsr_msgs2::msg::RobotState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotState_ & other) const
  {
    if (this->robot_state != other.robot_state) {
      return false;
    }
    if (this->robot_state_str != other.robot_state_str) {
      return false;
    }
    if (this->actual_mode != other.actual_mode) {
      return false;
    }
    if (this->actual_space != other.actual_space) {
      return false;
    }
    if (this->current_posj != other.current_posj) {
      return false;
    }
    if (this->current_velj != other.current_velj) {
      return false;
    }
    if (this->joint_abs != other.joint_abs) {
      return false;
    }
    if (this->joint_err != other.joint_err) {
      return false;
    }
    if (this->target_posj != other.target_posj) {
      return false;
    }
    if (this->target_velj != other.target_velj) {
      return false;
    }
    if (this->current_posx != other.current_posx) {
      return false;
    }
    if (this->current_tool_posx != other.current_tool_posx) {
      return false;
    }
    if (this->current_velx != other.current_velx) {
      return false;
    }
    if (this->task_err != other.task_err) {
      return false;
    }
    if (this->target_velx != other.target_velx) {
      return false;
    }
    if (this->target_posx != other.target_posx) {
      return false;
    }
    if (this->dynamic_tor != other.dynamic_tor) {
      return false;
    }
    if (this->actual_jts != other.actual_jts) {
      return false;
    }
    if (this->actual_ejt != other.actual_ejt) {
      return false;
    }
    if (this->actual_ett != other.actual_ett) {
      return false;
    }
    if (this->actual_bk != other.actual_bk) {
      return false;
    }
    if (this->actual_mc != other.actual_mc) {
      return false;
    }
    if (this->actual_mt != other.actual_mt) {
      return false;
    }
    if (this->solution_space != other.solution_space) {
      return false;
    }
    if (this->sync_time != other.sync_time) {
      return false;
    }
    if (this->actual_bt != other.actual_bt) {
      return false;
    }
    if (this->rotation_matrix != other.rotation_matrix) {
      return false;
    }
    if (this->ctrlbox_digital_input != other.ctrlbox_digital_input) {
      return false;
    }
    if (this->ctrlbox_digital_output != other.ctrlbox_digital_output) {
      return false;
    }
    if (this->flange_digital_input != other.flange_digital_input) {
      return false;
    }
    if (this->flange_digital_output != other.flange_digital_output) {
      return false;
    }
    if (this->modbus_state != other.modbus_state) {
      return false;
    }
    if (this->access_control != other.access_control) {
      return false;
    }
    if (this->homming_completed != other.homming_completed) {
      return false;
    }
    if (this->tp_initialized != other.tp_initialized) {
      return false;
    }
    if (this->mastering_need != other.mastering_need) {
      return false;
    }
    if (this->drl_stopped != other.drl_stopped) {
      return false;
    }
    if (this->disconnected != other.disconnected) {
      return false;
    }
    if (this->f_actual_w2b != other.f_actual_w2b) {
      return false;
    }
    if (this->f_current_pos_world != other.f_current_pos_world) {
      return false;
    }
    if (this->f_current_vel_world != other.f_current_vel_world) {
      return false;
    }
    if (this->f_world_ext_target_torque != other.f_world_ext_target_torque) {
      return false;
    }
    if (this->f_target_pos_world != other.f_target_pos_world) {
      return false;
    }
    if (this->f_target_vel_world != other.f_target_vel_world) {
      return false;
    }
    if (this->f_rotation_matrix_world != other.f_rotation_matrix_world) {
      return false;
    }
    if (this->i_actual_user_coord_num != other.i_actual_user_coord_num) {
      return false;
    }
    if (this->i_coord_ref != other.i_coord_ref) {
      return false;
    }
    if (this->f_current_pos_user != other.f_current_pos_user) {
      return false;
    }
    if (this->f_current_vel_user != other.f_current_vel_user) {
      return false;
    }
    if (this->f_user_ext_task_torque != other.f_user_ext_task_torque) {
      return false;
    }
    if (this->f_target_pos_user != other.f_target_pos_user) {
      return false;
    }
    if (this->f_target_vel_user != other.f_target_vel_user) {
      return false;
    }
    if (this->f_rotation_matrix_user != other.f_rotation_matrix_user) {
      return false;
    }
    if (this->f_actual_analog_input != other.f_actual_analog_input) {
      return false;
    }
    if (this->b_actual_switch_input != other.b_actual_switch_input) {
      return false;
    }
    if (this->b_actual_safety_input != other.b_actual_safety_input) {
      return false;
    }
    if (this->i_actual_analog_input_type != other.i_actual_analog_input_type) {
      return false;
    }
    if (this->f_target_analog_output != other.f_target_analog_output) {
      return false;
    }
    if (this->i_target_analog_output_type != other.i_target_analog_output_type) {
      return false;
    }
    if (this->b_actual_encorder_strove_signal != other.b_actual_encorder_strove_signal) {
      return false;
    }
    if (this->i_actual_encorder_raw_data != other.i_actual_encorder_raw_data) {
      return false;
    }
    if (this->b_actual_encorder_reset_signal != other.b_actual_encorder_reset_signal) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotState_

// alias to use template instance with default allocator
using RobotState =
  dsr_msgs2::msg::RobotState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_HPP_
