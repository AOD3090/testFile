// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRobotSpeedMode.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SPEED_MODE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SPEED_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_robot_speed_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotSpeedMode_Request_speed_mode
{
public:
  Init_SetRobotSpeedMode_Request_speed_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotSpeedMode_Request speed_mode(::dsr_msgs2::srv::SetRobotSpeedMode_Request::_speed_mode_type arg)
  {
    msg_.speed_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotSpeedMode_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotSpeedMode_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotSpeedMode_Request_speed_mode();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotSpeedMode_Response_success
{
public:
  Init_SetRobotSpeedMode_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotSpeedMode_Response success(::dsr_msgs2::srv::SetRobotSpeedMode_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotSpeedMode_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotSpeedMode_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotSpeedMode_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SPEED_MODE__BUILDER_HPP_
