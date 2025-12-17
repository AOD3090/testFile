// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRobotControl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ROBOT_CONTROL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_ROBOT_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_robot_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotControl_Request_robot_control
{
public:
  Init_SetRobotControl_Request_robot_control()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotControl_Request robot_control(::dsr_msgs2::srv::SetRobotControl_Request::_robot_control_type arg)
  {
    msg_.robot_control = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotControl_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotControl_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotControl_Request_robot_control();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotControl_Response_success
{
public:
  Init_SetRobotControl_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotControl_Response success(::dsr_msgs2::srv::SetRobotControl_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotControl_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotControl_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotControl_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ROBOT_CONTROL__BUILDER_HPP_
