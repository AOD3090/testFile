// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/DrlStart.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__DRL_START__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__DRL_START__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/drl_start__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_DrlStart_Request_code
{
public:
  explicit Init_DrlStart_Request_code(::dsr_msgs2::srv::DrlStart_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::DrlStart_Request code(::dsr_msgs2::srv::DrlStart_Request::_code_type arg)
  {
    msg_.code = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::DrlStart_Request msg_;
};

class Init_DrlStart_Request_robot_system
{
public:
  Init_DrlStart_Request_robot_system()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DrlStart_Request_code robot_system(::dsr_msgs2::srv::DrlStart_Request::_robot_system_type arg)
  {
    msg_.robot_system = std::move(arg);
    return Init_DrlStart_Request_code(msg_);
  }

private:
  ::dsr_msgs2::srv::DrlStart_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::DrlStart_Request>()
{
  return dsr_msgs2::srv::builder::Init_DrlStart_Request_robot_system();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_DrlStart_Response_success
{
public:
  Init_DrlStart_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::DrlStart_Response success(::dsr_msgs2::srv::DrlStart_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::DrlStart_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::DrlStart_Response>()
{
  return dsr_msgs2::srv::builder::Init_DrlStart_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__DRL_START__BUILDER_HPP_
