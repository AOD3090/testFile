// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetVeljRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_VELJ_RT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_VELJ_RT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_velj_rt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetVeljRt_Request_vel
{
public:
  Init_SetVeljRt_Request_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetVeljRt_Request vel(::dsr_msgs2::srv::SetVeljRt_Request::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetVeljRt_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetVeljRt_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetVeljRt_Request_vel();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetVeljRt_Response_success
{
public:
  Init_SetVeljRt_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetVeljRt_Response success(::dsr_msgs2::srv::SetVeljRt_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetVeljRt_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetVeljRt_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetVeljRt_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_VELJ_RT__BUILDER_HPP_
