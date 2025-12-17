// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'jacobian_matrix'
// Member 'coriolis_matrix'
// Member 'mass_matrix'
#include "std_msgs/msg/detail/float64_multi_array__struct.h"

/// Struct defined in msg/RobotStateRt in the package dsr_msgs2.
/**
  * timestamp at the data of data acquisition
 */
typedef struct dsr_msgs2__msg__RobotStateRt
{
  double time_stamp;
  /// actual joint position from incremental encoder at motor side(used for control)
  double actual_joint_position[6];
  /// actual joint position from absolute encoder at link side (used for exact link position)
  double actual_joint_position_abs[6];
  /// actual joint velocity from incremental encoder at motor side
  double actual_joint_velocity[6];
  /// actual joint velocity from absolute encoder at link side
  double actual_joint_velocity_abs[6];
  /// actual robot tcp position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
  double actual_tcp_position[6];
  /// actual robot tcp velocity w.r.t. base coordinates [mm, deg/s]
  double actual_tcp_velocity[6];
  /// actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
  double actual_flange_position[6];
  /// robot flange velocity w.r.t. base coordinates [mm, deg/s]
  double actual_flange_velocity[6];
  /// actual motor torque applying gear ratio = gear_ratio * current2torque_constant * motor current
  double actual_motor_torque[6];
  /// estimated joint torque by robot controller
  double actual_joint_torque[6];
  /// calibrated joint torque sensor data
  double raw_joint_torque[6];
  /// calibrated force torque sensor data w.r.t. flange coordinates [N, Nm]
  double raw_force_torque[6];
  /// estimated external joint torque
  double external_joint_torque[6];
  /// estimated tcp force w.r.t. base coordinates [N, Nm]
  double external_tcp_force[6];
  /// target joint position
  double target_joint_position[6];
  /// target joint velocity
  double target_joint_velocity[6];
  /// target joint acceleration
  double target_joint_acceleration[6];
  /// target motor torque
  double target_motor_torque[6];
  /// target tcp position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
  double target_tcp_position[6];
  /// target tcp velocity w.r.t. base coordinates [mm, deg/s]
  double target_tcp_velocity[6];
  /// jacobian matrix=J(q) w.r.t. base coordinates
  std_msgs__msg__Float64MultiArray__Sequence jacobian_matrix;
  /// gravity torque=g(q)
  double gravity_torque[6];
  /// coriolis matrix=C(q,q_dot)  [6][6]
  std_msgs__msg__Float64MultiArray__Sequence coriolis_matrix;
  /// mass matrix=M(q) [6][6]
  std_msgs__msg__Float64MultiArray__Sequence mass_matrix;
  /// robot configuration
  uint16_t solution_space;
  /// minimum singular value
  double singularity;
  /// current operation speed rate(1~100 %)
  double operation_speed_rate;
  /// joint temperature(celsius)
  double joint_temperature[6];
  /// controller digital input(16 channel)
  uint16_t controller_digital_input;
  /// controller digital output(16 channel)
  uint16_t controller_digital_output;
  /// controller analog input type(2 channel)
  uint8_t controller_analog_input_type[2];
  /// controller analog input(2 channel)
  double controller_analog_input[2];
  /// controller analog output type(2 channel)
  uint8_t controller_analog_output_type[2];
  /// controller analog output(2 channel)
  double controller_analog_output[2];
  /// flange digital input(A-Series: 2 channel, M/H-Series: 6 channel)
  uint8_t flange_digital_input;
  /// flange digital output(A-Series: 2 channel, M/H-Series: 6 channel)
  uint8_t flange_digital_output;
  /// flange analog input(A-Series: 2 channel, M/H-Series: 4 channel)
  double flange_analog_input[4];
  /// strobe count(increased by 1 when detecting setting edge)
  uint8_t external_encoder_strobe_count[2];
  /// external encoder count
  uint16_t external_encoder_count[2];
  /// final goal joint position (reserved)
  double goal_joint_position[6];
  /// final goal tcp position (reserved)
  double goal_tcp_position[6];
  /// ROBOT_MODE_MANUAL(0), ROBOT_MODE_AUTONOMOUS(1), ROBOT_MODE_MEASURE(2)
  uint8_t robot_mode;
  /// STATE_INITIALIZING(0), STATE_STANDBY(1), STATE_MOVING(2), STATE_SAFE_OFF(3), STATE_TEACHING(4), STATE_SAFE_STOP(5), STATE_EMERGENCY_STOP, STATE_HOMMING, STATE_RECOVERY, STATE_SAFE_STOP2, STATE_SAFE_OFF2,
  uint8_t robot_state;
  /// position control mode, torque mode
  uint16_t control_mode;
  /// Reserved
  uint8_t reserved[256];
} dsr_msgs2__msg__RobotStateRt;

// Struct for a sequence of dsr_msgs2__msg__RobotStateRt.
typedef struct dsr_msgs2__msg__RobotStateRt__Sequence
{
  dsr_msgs2__msg__RobotStateRt * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__RobotStateRt__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__STRUCT_H_
