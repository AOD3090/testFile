// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'robot_state_str'
#include "rosidl_runtime_c/string.h"
// Member 'rotation_matrix'
// Member 'f_current_pos_world'
// Member 'f_rotation_matrix_world'
// Member 'f_current_pos_user'
// Member 'f_rotation_matrix_user'
#include "std_msgs/msg/detail/float64_multi_array__struct.h"
// Member 'modbus_state'
#include "dsr_msgs2/msg/detail/modbus_state__struct.h"

/// Struct defined in msg/RobotState in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  state of robot
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__RobotState
{
  /// 0 : STATE_INITIALIZING,   1 : STATE_STANDBY,   2 : STATE_MOVING,    3 : STATE_SAFE_OFF
  /// 4 : STATE_TEACHING,       5 : STATE_SAFE_STOP, 6 : STATE_EMERGENCY_STOP,
  /// 7 : STATE_EMERGENCY_STOP, 8 : STATE_HOMMING,   9 : STATE_RECOVERY,  10: STATE_SAFE_STOP2,
  /// 11: STATE_SAFE_OFF2,      12: STATE_RESERVED1, 13: STATE_RESERVED2, 14: STATE_RESERVED3,
  /// 15: STATE_NOT_READY       16: STATE_LAST
  int32_t robot_state;
  /// Convert robot_state id to string
  rosidl_runtime_c__String robot_state_str;
  /// position control: 0, torque control: 1
  int8_t actual_mode;
  /// joint space: 0, task space: 1
  int8_t actual_space;
  /// current joint angle list
  double current_posj[6];
  /// current joint velocity list []
  double current_velj[6];
  /// Position Actual Value in ABS
  double joint_abs[6];
  /// Joint Error
  double joint_err[6];
  /// target joint angle list
  double target_posj[6];
  /// target joint velocity list []
  double target_velj[6];
  /// current task angle list []
  double current_posx[6];
  /// current task tool angle list []
  double current_tool_posx[6];
  /// current task velocity list []
  double current_velx[6];
  /// Task Error
  double task_err[6];
  /// target task velocity list []
  double target_velx[6];
  /// target task position list []
  double target_posx[6];
  /// dynamic torque
  double dynamic_tor[6];
  /// joint torque sensor
  double actual_jts[6];
  /// external joint torque
  double actual_ejt[6];
  /// external tool torque
  double actual_ett[6];
  /// brake status
  int8_t actual_bk[6];
  /// motor current
  double actual_mc[6];
  /// motor temperature
  double actual_mt[6];
  /// Solution Space (0 ~ 7)
  int8_t solution_space;
  /// internal clock counter
  double sync_time;
  /// cockpit(robot button) info.
  int8_t actual_bt[5];
  /// Rotation Matrix [3][3]
  std_msgs__msg__Float64MultiArray__Sequence rotation_matrix;
  /// Digital Input in Control Box(0 ~ 15 ; 0 : ON, 1 : OFF)
  int8_t ctrlbox_digital_input[16];
  /// Digital Output in Control Box(0 ~ 15 ; 0 : ON, 1 : OFF)
  int8_t ctrlbox_digital_output[16];
  /// Digital Input in Flange(0 ~ 5 ; 0 : ON, 1 : OFF) x1 port : 0 ~ 2, x2 port : 3 ~ 5
  int8_t flange_digital_input[6];
  /// Digital Output in Flange(0 ~ 5 ; 0 : ON, 1 : OFF)
  int8_t flange_digital_output[6];
  /// Custom msg for modbus state(refer to ModbusState.msg)
  dsr_msgs2__msg__ModbusState__Sequence modbus_state;
  int32_t access_control;
  bool homming_completed;
  bool tp_initialized;
  int8_t mastering_need;
  bool drl_stopped;
  bool disconnected;
  /// ____________________________________________________________________________________________
  ///  The following messages have been updated since version M2.50 or higher.
  /// ____________________________________________________________________________________________
  ///  world to base releation
  double f_actual_w2b[6];
  /// Wolrd position actual value [2][6] : (0: tool, 1: flange) [mm, degree]
  std_msgs__msg__Float64MultiArray__Sequence f_current_pos_world;
  /// World velocity Actual Value [mm/sec, degree/sec]
  double f_current_vel_world[6];
  /// External Task Force/Torque [N, Nm]
  double f_world_ext_target_torque[6];
  /// World target Position [mm, degree]
  double f_target_pos_world[6];
  /// World target Velocity [mm/sec, degree/sec]
  double f_target_vel_world[6];
  /// World rotation matrix [3][3]
  std_msgs__msg__Float64MultiArray__Sequence f_rotation_matrix_world;
  /// Actual user coord number ## 101 ~ 120
  int8_t i_actual_user_coord_num;
  /// Coordinate Reference(base : 0  world : 2)
  int8_t i_coord_ref;
  /// User position Actual Value [2][6] : (0:tool, 1:flange) [mm, degree]
  std_msgs__msg__Float64MultiArray__Sequence f_current_pos_user;
  /// User velocity Actual Value [mm/sec, degree/sec]
  double f_current_vel_user[6];
  /// External Task Force/Torque [N, Nm]
  double f_user_ext_task_torque[6];
  /// User target Position [mm, degree]
  double f_target_pos_user[6];
  /// User target Velocity [mm/sec, degree/sec]
  double f_target_vel_user[6];
  /// User rotation matrix [3][3]
  std_msgs__msg__Float64MultiArray__Sequence f_rotation_matrix_user;
  /// Analog input data ## Current mode : 0~20.0[mA] , Voltage mode : 0~10.0[V]
  double f_actual_analog_input[6];
  /// Switch input data
  bool b_actual_switch_input[3];
  /// Safety input data
  bool b_actual_safety_input[2];
  /// Analog input type  index = channel, type: current(0), voltage(1)
  int8_t i_actual_analog_input_type[2];
  /// Analog output data ## Current mode : 0~20.0[mA] , Voltage mode : 0~10.0[V]
  double f_target_analog_output[2];
  /// Analog output type index = channel, type: current(0), voltage(1)
  int8_t i_target_analog_output_type[2];
  /// Encorder strove signal
  bool b_actual_encorder_strove_signal[2];
  /// Encorder raw data
  int8_t i_actual_encorder_raw_data[2];
  /// Encorder reset signal
  bool b_actual_encorder_reset_signal[2];
} dsr_msgs2__msg__RobotState;

// Struct for a sequence of dsr_msgs2__msg__RobotState.
typedef struct dsr_msgs2__msg__RobotState__Sequence
{
  dsr_msgs2__msg__RobotState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__RobotState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__STRUCT_H_
