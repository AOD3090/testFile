// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetCurrentPose.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_current_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCurrentPose_Request_space_type
{
public:
  Init_GetCurrentPose_Request_space_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetCurrentPose_Request space_type(::dsr_msgs2::srv::GetCurrentPose_Request::_space_type_type arg)
  {
    msg_.space_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentPose_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentPose_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetCurrentPose_Request_space_type();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCurrentPose_Response_success
{
public:
  explicit Init_GetCurrentPose_Response_success(::dsr_msgs2::srv::GetCurrentPose_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetCurrentPose_Response success(::dsr_msgs2::srv::GetCurrentPose_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentPose_Response msg_;
};

class Init_GetCurrentPose_Response_pos
{
public:
  Init_GetCurrentPose_Response_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetCurrentPose_Response_success pos(::dsr_msgs2::srv::GetCurrentPose_Response::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_GetCurrentPose_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentPose_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentPose_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetCurrentPose_Response_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSE__BUILDER_HPP_
