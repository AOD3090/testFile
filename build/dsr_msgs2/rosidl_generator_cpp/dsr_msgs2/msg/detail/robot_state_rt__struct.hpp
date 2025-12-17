// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'jacobian_matrix'
// Member 'coriolis_matrix'
// Member 'mass_matrix'
#include "std_msgs/msg/detail/float64_multi_array__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__msg__RobotStateRt __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__msg__RobotStateRt __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotStateRt_
{
  using Type = RobotStateRt_<ContainerAllocator>;

  explicit RobotStateRt_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_stamp = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_position.begin(), this->actual_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_position_abs.begin(), this->actual_joint_position_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_velocity.begin(), this->actual_joint_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_velocity_abs.begin(), this->actual_joint_velocity_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_tcp_position.begin(), this->actual_tcp_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_tcp_velocity.begin(), this->actual_tcp_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_flange_position.begin(), this->actual_flange_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_flange_velocity.begin(), this->actual_flange_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_motor_torque.begin(), this->actual_motor_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_torque.begin(), this->actual_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->raw_joint_torque.begin(), this->raw_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->raw_force_torque.begin(), this->raw_force_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_joint_torque.begin(), this->external_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_tcp_force.begin(), this->external_tcp_force.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_position.begin(), this->target_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_velocity.begin(), this->target_joint_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_acceleration.begin(), this->target_joint_acceleration.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_motor_torque.begin(), this->target_motor_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_tcp_position.begin(), this->target_tcp_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_tcp_velocity.begin(), this->target_tcp_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->gravity_torque.begin(), this->gravity_torque.end(), 0.0);
      this->solution_space = 0;
      this->singularity = 0.0;
      this->operation_speed_rate = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_temperature.begin(), this->joint_temperature.end(), 0.0);
      this->controller_digital_input = 0;
      this->controller_digital_output = 0;
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->controller_analog_input_type.begin(), this->controller_analog_input_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->controller_analog_input.begin(), this->controller_analog_input.end(), 0.0);
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->controller_analog_output_type.begin(), this->controller_analog_output_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->controller_analog_output.begin(), this->controller_analog_output.end(), 0.0);
      this->flange_digital_input = 0;
      this->flange_digital_output = 0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->flange_analog_input.begin(), this->flange_analog_input.end(), 0.0);
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->external_encoder_strobe_count.begin(), this->external_encoder_strobe_count.end(), 0);
      std::fill<typename std::array<uint16_t, 2>::iterator, uint16_t>(this->external_encoder_count.begin(), this->external_encoder_count.end(), 0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->goal_joint_position.begin(), this->goal_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->goal_tcp_position.begin(), this->goal_tcp_position.end(), 0.0);
      this->robot_mode = 0;
      this->robot_state = 0;
      this->control_mode = 0;
      std::fill<typename std::array<uint8_t, 256>::iterator, uint8_t>(this->reserved.begin(), this->reserved.end(), 0);
    }
  }

  explicit RobotStateRt_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : actual_joint_position(_alloc),
    actual_joint_position_abs(_alloc),
    actual_joint_velocity(_alloc),
    actual_joint_velocity_abs(_alloc),
    actual_tcp_position(_alloc),
    actual_tcp_velocity(_alloc),
    actual_flange_position(_alloc),
    actual_flange_velocity(_alloc),
    actual_motor_torque(_alloc),
    actual_joint_torque(_alloc),
    raw_joint_torque(_alloc),
    raw_force_torque(_alloc),
    external_joint_torque(_alloc),
    external_tcp_force(_alloc),
    target_joint_position(_alloc),
    target_joint_velocity(_alloc),
    target_joint_acceleration(_alloc),
    target_motor_torque(_alloc),
    target_tcp_position(_alloc),
    target_tcp_velocity(_alloc),
    gravity_torque(_alloc),
    joint_temperature(_alloc),
    controller_analog_input_type(_alloc),
    controller_analog_input(_alloc),
    controller_analog_output_type(_alloc),
    controller_analog_output(_alloc),
    flange_analog_input(_alloc),
    external_encoder_strobe_count(_alloc),
    external_encoder_count(_alloc),
    goal_joint_position(_alloc),
    goal_tcp_position(_alloc),
    reserved(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time_stamp = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_position.begin(), this->actual_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_position_abs.begin(), this->actual_joint_position_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_velocity.begin(), this->actual_joint_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_velocity_abs.begin(), this->actual_joint_velocity_abs.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_tcp_position.begin(), this->actual_tcp_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_tcp_velocity.begin(), this->actual_tcp_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_flange_position.begin(), this->actual_flange_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_flange_velocity.begin(), this->actual_flange_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_motor_torque.begin(), this->actual_motor_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->actual_joint_torque.begin(), this->actual_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->raw_joint_torque.begin(), this->raw_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->raw_force_torque.begin(), this->raw_force_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_joint_torque.begin(), this->external_joint_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_tcp_force.begin(), this->external_tcp_force.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_position.begin(), this->target_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_velocity.begin(), this->target_joint_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_joint_acceleration.begin(), this->target_joint_acceleration.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_motor_torque.begin(), this->target_motor_torque.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_tcp_position.begin(), this->target_tcp_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->target_tcp_velocity.begin(), this->target_tcp_velocity.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->gravity_torque.begin(), this->gravity_torque.end(), 0.0);
      this->solution_space = 0;
      this->singularity = 0.0;
      this->operation_speed_rate = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_temperature.begin(), this->joint_temperature.end(), 0.0);
      this->controller_digital_input = 0;
      this->controller_digital_output = 0;
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->controller_analog_input_type.begin(), this->controller_analog_input_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->controller_analog_input.begin(), this->controller_analog_input.end(), 0.0);
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->controller_analog_output_type.begin(), this->controller_analog_output_type.end(), 0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->controller_analog_output.begin(), this->controller_analog_output.end(), 0.0);
      this->flange_digital_input = 0;
      this->flange_digital_output = 0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->flange_analog_input.begin(), this->flange_analog_input.end(), 0.0);
      std::fill<typename std::array<uint8_t, 2>::iterator, uint8_t>(this->external_encoder_strobe_count.begin(), this->external_encoder_strobe_count.end(), 0);
      std::fill<typename std::array<uint16_t, 2>::iterator, uint16_t>(this->external_encoder_count.begin(), this->external_encoder_count.end(), 0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->goal_joint_position.begin(), this->goal_joint_position.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->goal_tcp_position.begin(), this->goal_tcp_position.end(), 0.0);
      this->robot_mode = 0;
      this->robot_state = 0;
      this->control_mode = 0;
      std::fill<typename std::array<uint8_t, 256>::iterator, uint8_t>(this->reserved.begin(), this->reserved.end(), 0);
    }
  }

  // field types and members
  using _time_stamp_type =
    double;
  _time_stamp_type time_stamp;
  using _actual_joint_position_type =
    std::array<double, 6>;
  _actual_joint_position_type actual_joint_position;
  using _actual_joint_position_abs_type =
    std::array<double, 6>;
  _actual_joint_position_abs_type actual_joint_position_abs;
  using _actual_joint_velocity_type =
    std::array<double, 6>;
  _actual_joint_velocity_type actual_joint_velocity;
  using _actual_joint_velocity_abs_type =
    std::array<double, 6>;
  _actual_joint_velocity_abs_type actual_joint_velocity_abs;
  using _actual_tcp_position_type =
    std::array<double, 6>;
  _actual_tcp_position_type actual_tcp_position;
  using _actual_tcp_velocity_type =
    std::array<double, 6>;
  _actual_tcp_velocity_type actual_tcp_velocity;
  using _actual_flange_position_type =
    std::array<double, 6>;
  _actual_flange_position_type actual_flange_position;
  using _actual_flange_velocity_type =
    std::array<double, 6>;
  _actual_flange_velocity_type actual_flange_velocity;
  using _actual_motor_torque_type =
    std::array<double, 6>;
  _actual_motor_torque_type actual_motor_torque;
  using _actual_joint_torque_type =
    std::array<double, 6>;
  _actual_joint_torque_type actual_joint_torque;
  using _raw_joint_torque_type =
    std::array<double, 6>;
  _raw_joint_torque_type raw_joint_torque;
  using _raw_force_torque_type =
    std::array<double, 6>;
  _raw_force_torque_type raw_force_torque;
  using _external_joint_torque_type =
    std::array<double, 6>;
  _external_joint_torque_type external_joint_torque;
  using _external_tcp_force_type =
    std::array<double, 6>;
  _external_tcp_force_type external_tcp_force;
  using _target_joint_position_type =
    std::array<double, 6>;
  _target_joint_position_type target_joint_position;
  using _target_joint_velocity_type =
    std::array<double, 6>;
  _target_joint_velocity_type target_joint_velocity;
  using _target_joint_acceleration_type =
    std::array<double, 6>;
  _target_joint_acceleration_type target_joint_acceleration;
  using _target_motor_torque_type =
    std::array<double, 6>;
  _target_motor_torque_type target_motor_torque;
  using _target_tcp_position_type =
    std::array<double, 6>;
  _target_tcp_position_type target_tcp_position;
  using _target_tcp_velocity_type =
    std::array<double, 6>;
  _target_tcp_velocity_type target_tcp_velocity;
  using _jacobian_matrix_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _jacobian_matrix_type jacobian_matrix;
  using _gravity_torque_type =
    std::array<double, 6>;
  _gravity_torque_type gravity_torque;
  using _coriolis_matrix_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _coriolis_matrix_type coriolis_matrix;
  using _mass_matrix_type =
    std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>>;
  _mass_matrix_type mass_matrix;
  using _solution_space_type =
    uint16_t;
  _solution_space_type solution_space;
  using _singularity_type =
    double;
  _singularity_type singularity;
  using _operation_speed_rate_type =
    double;
  _operation_speed_rate_type operation_speed_rate;
  using _joint_temperature_type =
    std::array<double, 6>;
  _joint_temperature_type joint_temperature;
  using _controller_digital_input_type =
    uint16_t;
  _controller_digital_input_type controller_digital_input;
  using _controller_digital_output_type =
    uint16_t;
  _controller_digital_output_type controller_digital_output;
  using _controller_analog_input_type_type =
    std::array<uint8_t, 2>;
  _controller_analog_input_type_type controller_analog_input_type;
  using _controller_analog_input_type =
    std::array<double, 2>;
  _controller_analog_input_type controller_analog_input;
  using _controller_analog_output_type_type =
    std::array<uint8_t, 2>;
  _controller_analog_output_type_type controller_analog_output_type;
  using _controller_analog_output_type =
    std::array<double, 2>;
  _controller_analog_output_type controller_analog_output;
  using _flange_digital_input_type =
    uint8_t;
  _flange_digital_input_type flange_digital_input;
  using _flange_digital_output_type =
    uint8_t;
  _flange_digital_output_type flange_digital_output;
  using _flange_analog_input_type =
    std::array<double, 4>;
  _flange_analog_input_type flange_analog_input;
  using _external_encoder_strobe_count_type =
    std::array<uint8_t, 2>;
  _external_encoder_strobe_count_type external_encoder_strobe_count;
  using _external_encoder_count_type =
    std::array<uint16_t, 2>;
  _external_encoder_count_type external_encoder_count;
  using _goal_joint_position_type =
    std::array<double, 6>;
  _goal_joint_position_type goal_joint_position;
  using _goal_tcp_position_type =
    std::array<double, 6>;
  _goal_tcp_position_type goal_tcp_position;
  using _robot_mode_type =
    uint8_t;
  _robot_mode_type robot_mode;
  using _robot_state_type =
    uint8_t;
  _robot_state_type robot_state;
  using _control_mode_type =
    uint16_t;
  _control_mode_type control_mode;
  using _reserved_type =
    std::array<uint8_t, 256>;
  _reserved_type reserved;

  // setters for named parameter idiom
  Type & set__time_stamp(
    const double & _arg)
  {
    this->time_stamp = _arg;
    return *this;
  }
  Type & set__actual_joint_position(
    const std::array<double, 6> & _arg)
  {
    this->actual_joint_position = _arg;
    return *this;
  }
  Type & set__actual_joint_position_abs(
    const std::array<double, 6> & _arg)
  {
    this->actual_joint_position_abs = _arg;
    return *this;
  }
  Type & set__actual_joint_velocity(
    const std::array<double, 6> & _arg)
  {
    this->actual_joint_velocity = _arg;
    return *this;
  }
  Type & set__actual_joint_velocity_abs(
    const std::array<double, 6> & _arg)
  {
    this->actual_joint_velocity_abs = _arg;
    return *this;
  }
  Type & set__actual_tcp_position(
    const std::array<double, 6> & _arg)
  {
    this->actual_tcp_position = _arg;
    return *this;
  }
  Type & set__actual_tcp_velocity(
    const std::array<double, 6> & _arg)
  {
    this->actual_tcp_velocity = _arg;
    return *this;
  }
  Type & set__actual_flange_position(
    const std::array<double, 6> & _arg)
  {
    this->actual_flange_position = _arg;
    return *this;
  }
  Type & set__actual_flange_velocity(
    const std::array<double, 6> & _arg)
  {
    this->actual_flange_velocity = _arg;
    return *this;
  }
  Type & set__actual_motor_torque(
    const std::array<double, 6> & _arg)
  {
    this->actual_motor_torque = _arg;
    return *this;
  }
  Type & set__actual_joint_torque(
    const std::array<double, 6> & _arg)
  {
    this->actual_joint_torque = _arg;
    return *this;
  }
  Type & set__raw_joint_torque(
    const std::array<double, 6> & _arg)
  {
    this->raw_joint_torque = _arg;
    return *this;
  }
  Type & set__raw_force_torque(
    const std::array<double, 6> & _arg)
  {
    this->raw_force_torque = _arg;
    return *this;
  }
  Type & set__external_joint_torque(
    const std::array<double, 6> & _arg)
  {
    this->external_joint_torque = _arg;
    return *this;
  }
  Type & set__external_tcp_force(
    const std::array<double, 6> & _arg)
  {
    this->external_tcp_force = _arg;
    return *this;
  }
  Type & set__target_joint_position(
    const std::array<double, 6> & _arg)
  {
    this->target_joint_position = _arg;
    return *this;
  }
  Type & set__target_joint_velocity(
    const std::array<double, 6> & _arg)
  {
    this->target_joint_velocity = _arg;
    return *this;
  }
  Type & set__target_joint_acceleration(
    const std::array<double, 6> & _arg)
  {
    this->target_joint_acceleration = _arg;
    return *this;
  }
  Type & set__target_motor_torque(
    const std::array<double, 6> & _arg)
  {
    this->target_motor_torque = _arg;
    return *this;
  }
  Type & set__target_tcp_position(
    const std::array<double, 6> & _arg)
  {
    this->target_tcp_position = _arg;
    return *this;
  }
  Type & set__target_tcp_velocity(
    const std::array<double, 6> & _arg)
  {
    this->target_tcp_velocity = _arg;
    return *this;
  }
  Type & set__jacobian_matrix(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->jacobian_matrix = _arg;
    return *this;
  }
  Type & set__gravity_torque(
    const std::array<double, 6> & _arg)
  {
    this->gravity_torque = _arg;
    return *this;
  }
  Type & set__coriolis_matrix(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->coriolis_matrix = _arg;
    return *this;
  }
  Type & set__mass_matrix(
    const std::vector<std_msgs::msg::Float64MultiArray_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std_msgs::msg::Float64MultiArray_<ContainerAllocator>>> & _arg)
  {
    this->mass_matrix = _arg;
    return *this;
  }
  Type & set__solution_space(
    const uint16_t & _arg)
  {
    this->solution_space = _arg;
    return *this;
  }
  Type & set__singularity(
    const double & _arg)
  {
    this->singularity = _arg;
    return *this;
  }
  Type & set__operation_speed_rate(
    const double & _arg)
  {
    this->operation_speed_rate = _arg;
    return *this;
  }
  Type & set__joint_temperature(
    const std::array<double, 6> & _arg)
  {
    this->joint_temperature = _arg;
    return *this;
  }
  Type & set__controller_digital_input(
    const uint16_t & _arg)
  {
    this->controller_digital_input = _arg;
    return *this;
  }
  Type & set__controller_digital_output(
    const uint16_t & _arg)
  {
    this->controller_digital_output = _arg;
    return *this;
  }
  Type & set__controller_analog_input_type(
    const std::array<uint8_t, 2> & _arg)
  {
    this->controller_analog_input_type = _arg;
    return *this;
  }
  Type & set__controller_analog_input(
    const std::array<double, 2> & _arg)
  {
    this->controller_analog_input = _arg;
    return *this;
  }
  Type & set__controller_analog_output_type(
    const std::array<uint8_t, 2> & _arg)
  {
    this->controller_analog_output_type = _arg;
    return *this;
  }
  Type & set__controller_analog_output(
    const std::array<double, 2> & _arg)
  {
    this->controller_analog_output = _arg;
    return *this;
  }
  Type & set__flange_digital_input(
    const uint8_t & _arg)
  {
    this->flange_digital_input = _arg;
    return *this;
  }
  Type & set__flange_digital_output(
    const uint8_t & _arg)
  {
    this->flange_digital_output = _arg;
    return *this;
  }
  Type & set__flange_analog_input(
    const std::array<double, 4> & _arg)
  {
    this->flange_analog_input = _arg;
    return *this;
  }
  Type & set__external_encoder_strobe_count(
    const std::array<uint8_t, 2> & _arg)
  {
    this->external_encoder_strobe_count = _arg;
    return *this;
  }
  Type & set__external_encoder_count(
    const std::array<uint16_t, 2> & _arg)
  {
    this->external_encoder_count = _arg;
    return *this;
  }
  Type & set__goal_joint_position(
    const std::array<double, 6> & _arg)
  {
    this->goal_joint_position = _arg;
    return *this;
  }
  Type & set__goal_tcp_position(
    const std::array<double, 6> & _arg)
  {
    this->goal_tcp_position = _arg;
    return *this;
  }
  Type & set__robot_mode(
    const uint8_t & _arg)
  {
    this->robot_mode = _arg;
    return *this;
  }
  Type & set__robot_state(
    const uint8_t & _arg)
  {
    this->robot_state = _arg;
    return *this;
  }
  Type & set__control_mode(
    const uint16_t & _arg)
  {
    this->control_mode = _arg;
    return *this;
  }
  Type & set__reserved(
    const std::array<uint8_t, 256> & _arg)
  {
    this->reserved = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__msg__RobotStateRt
    std::shared_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__msg__RobotStateRt
    std::shared_ptr<dsr_msgs2::msg::RobotStateRt_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotStateRt_ & other) const
  {
    if (this->time_stamp != other.time_stamp) {
      return false;
    }
    if (this->actual_joint_position != other.actual_joint_position) {
      return false;
    }
    if (this->actual_joint_position_abs != other.actual_joint_position_abs) {
      return false;
    }
    if (this->actual_joint_velocity != other.actual_joint_velocity) {
      return false;
    }
    if (this->actual_joint_velocity_abs != other.actual_joint_velocity_abs) {
      return false;
    }
    if (this->actual_tcp_position != other.actual_tcp_position) {
      return false;
    }
    if (this->actual_tcp_velocity != other.actual_tcp_velocity) {
      return false;
    }
    if (this->actual_flange_position != other.actual_flange_position) {
      return false;
    }
    if (this->actual_flange_velocity != other.actual_flange_velocity) {
      return false;
    }
    if (this->actual_motor_torque != other.actual_motor_torque) {
      return false;
    }
    if (this->actual_joint_torque != other.actual_joint_torque) {
      return false;
    }
    if (this->raw_joint_torque != other.raw_joint_torque) {
      return false;
    }
    if (this->raw_force_torque != other.raw_force_torque) {
      return false;
    }
    if (this->external_joint_torque != other.external_joint_torque) {
      return false;
    }
    if (this->external_tcp_force != other.external_tcp_force) {
      return false;
    }
    if (this->target_joint_position != other.target_joint_position) {
      return false;
    }
    if (this->target_joint_velocity != other.target_joint_velocity) {
      return false;
    }
    if (this->target_joint_acceleration != other.target_joint_acceleration) {
      return false;
    }
    if (this->target_motor_torque != other.target_motor_torque) {
      return false;
    }
    if (this->target_tcp_position != other.target_tcp_position) {
      return false;
    }
    if (this->target_tcp_velocity != other.target_tcp_velocity) {
      return false;
    }
    if (this->jacobian_matrix != other.jacobian_matrix) {
      return false;
    }
    if (this->gravity_torque != other.gravity_torque) {
      return false;
    }
    if (this->coriolis_matrix != other.coriolis_matrix) {
      return false;
    }
    if (this->mass_matrix != other.mass_matrix) {
      return false;
    }
    if (this->solution_space != other.solution_space) {
      return false;
    }
    if (this->singularity != other.singularity) {
      return false;
    }
    if (this->operation_speed_rate != other.operation_speed_rate) {
      return false;
    }
    if (this->joint_temperature != other.joint_temperature) {
      return false;
    }
    if (this->controller_digital_input != other.controller_digital_input) {
      return false;
    }
    if (this->controller_digital_output != other.controller_digital_output) {
      return false;
    }
    if (this->controller_analog_input_type != other.controller_analog_input_type) {
      return false;
    }
    if (this->controller_analog_input != other.controller_analog_input) {
      return false;
    }
    if (this->controller_analog_output_type != other.controller_analog_output_type) {
      return false;
    }
    if (this->controller_analog_output != other.controller_analog_output) {
      return false;
    }
    if (this->flange_digital_input != other.flange_digital_input) {
      return false;
    }
    if (this->flange_digital_output != other.flange_digital_output) {
      return false;
    }
    if (this->flange_analog_input != other.flange_analog_input) {
      return false;
    }
    if (this->external_encoder_strobe_count != other.external_encoder_strobe_count) {
      return false;
    }
    if (this->external_encoder_count != other.external_encoder_count) {
      return false;
    }
    if (this->goal_joint_position != other.goal_joint_position) {
      return false;
    }
    if (this->goal_tcp_position != other.goal_tcp_position) {
      return false;
    }
    if (this->robot_mode != other.robot_mode) {
      return false;
    }
    if (this->robot_state != other.robot_state) {
      return false;
    }
    if (this->control_mode != other.control_mode) {
      return false;
    }
    if (this->reserved != other.reserved) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotStateRt_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotStateRt_

// alias to use template instance with default allocator
using RobotStateRt =
  dsr_msgs2::msg::RobotStateRt_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_HPP_
