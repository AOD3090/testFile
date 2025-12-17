// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetDesiredVelj.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_desired_velj__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDesiredVelj_Request>()
{
  return ::dsr_msgs2::srv::GetDesiredVelj_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetDesiredVelj_Response_success
{
public:
  explicit Init_GetDesiredVelj_Response_success(::dsr_msgs2::srv::GetDesiredVelj_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetDesiredVelj_Response success(::dsr_msgs2::srv::GetDesiredVelj_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDesiredVelj_Response msg_;
};

class Init_GetDesiredVelj_Response_joint_vel
{
public:
  Init_GetDesiredVelj_Response_joint_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetDesiredVelj_Response_success joint_vel(::dsr_msgs2::srv::GetDesiredVelj_Response::_joint_vel_type arg)
  {
    msg_.joint_vel = std::move(arg);
    return Init_GetDesiredVelj_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDesiredVelj_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDesiredVelj_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetDesiredVelj_Response_joint_vel();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__BUILDER_HPP_
