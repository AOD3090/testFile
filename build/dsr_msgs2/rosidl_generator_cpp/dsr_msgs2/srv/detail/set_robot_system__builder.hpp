// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRobotSystem.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SYSTEM__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SYSTEM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_robot_system__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotSystem_Request_robot_system
{
public:
  Init_SetRobotSystem_Request_robot_system()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotSystem_Request robot_system(::dsr_msgs2::srv::SetRobotSystem_Request::_robot_system_type arg)
  {
    msg_.robot_system = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotSystem_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotSystem_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotSystem_Request_robot_system();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRobotSystem_Response_success
{
public:
  Init_SetRobotSystem_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRobotSystem_Response success(::dsr_msgs2::srv::SetRobotSystem_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRobotSystem_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRobotSystem_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRobotSystem_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ROBOT_SYSTEM__BUILDER_HPP_
