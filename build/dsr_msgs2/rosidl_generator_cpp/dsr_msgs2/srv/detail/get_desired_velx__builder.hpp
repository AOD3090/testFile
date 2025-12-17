// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetDesiredVelx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_desired_velx__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetDesiredVelx_Request_ref
{
public:
  Init_GetDesiredVelx_Request_ref()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetDesiredVelx_Request ref(::dsr_msgs2::srv::GetDesiredVelx_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDesiredVelx_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDesiredVelx_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetDesiredVelx_Request_ref();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetDesiredVelx_Response_success
{
public:
  explicit Init_GetDesiredVelx_Response_success(::dsr_msgs2::srv::GetDesiredVelx_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetDesiredVelx_Response success(::dsr_msgs2::srv::GetDesiredVelx_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDesiredVelx_Response msg_;
};

class Init_GetDesiredVelx_Response_vel
{
public:
  Init_GetDesiredVelx_Response_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetDesiredVelx_Response_success vel(::dsr_msgs2::srv::GetDesiredVelx_Response::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_GetDesiredVelx_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDesiredVelx_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDesiredVelx_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetDesiredVelx_Response_vel();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__BUILDER_HPP_
