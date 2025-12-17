// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetCurrentToolFlangePosx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TOOL_FLANGE_POSX__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TOOL_FLANGE_POSX__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_current_tool_flange_posx__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCurrentToolFlangePosx_Request_ref
{
public:
  Init_GetCurrentToolFlangePosx_Request_ref()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetCurrentToolFlangePosx_Request ref(::dsr_msgs2::srv::GetCurrentToolFlangePosx_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentToolFlangePosx_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentToolFlangePosx_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetCurrentToolFlangePosx_Request_ref();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCurrentToolFlangePosx_Response_success
{
public:
  explicit Init_GetCurrentToolFlangePosx_Response_success(::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response success(::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response msg_;
};

class Init_GetCurrentToolFlangePosx_Response_pos
{
public:
  Init_GetCurrentToolFlangePosx_Response_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetCurrentToolFlangePosx_Response_success pos(::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_GetCurrentToolFlangePosx_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentToolFlangePosx_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetCurrentToolFlangePosx_Response_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TOOL_FLANGE_POSX__BUILDER_HPP_
