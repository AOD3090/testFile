// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRobotMode.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_robot_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotMode_Request_robot_mode
{
public:
  Init_SetRobotMode_Request_robot_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotMode_Request robot_mode(::dsr_msgs2::srv::SetRobotMode_Request::_robot_mode_type arg)
  {
    msg_.robot_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotMode_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotMode_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotMode_Request_robot_mode();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotMode_Response_success
{
public:
  Init_SetRobotMode_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotMode_Response success(::dsr_msgs2::srv::SetRobotMode_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotMode_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotMode_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotMode_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__BUILDER_HPP_
