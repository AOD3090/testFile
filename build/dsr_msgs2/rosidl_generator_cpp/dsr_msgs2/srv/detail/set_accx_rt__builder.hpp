// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetAccxRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ACCX_RT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_ACCX_RT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_accx_rt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetAccxRt_Request_rotation
{
public:
  explicit Init_SetAccxRt_Request_rotation(::dsr_msgs2::srv::SetAccxRt_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetAccxRt_Request rotation(::dsr_msgs2::srv::SetAccxRt_Request::_rotation_type arg)
  {
    msg_.rotation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetAccxRt_Request msg_;
};

class Init_SetAccxRt_Request_trans
{
public:
  Init_SetAccxRt_Request_trans()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetAccxRt_Request_rotation trans(::dsr_msgs2::srv::SetAccxRt_Request::_trans_type arg)
  {
    msg_.trans = std::move(arg);
    return Init_SetAccxRt_Request_rotation(msg_);
  }

private:
  ::dsr_msgs2::srv::SetAccxRt_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetAccxRt_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetAccxRt_Request_trans();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetAccxRt_Response_success
{
public:
  Init_SetAccxRt_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetAccxRt_Response success(::dsr_msgs2::srv::SetAccxRt_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetAccxRt_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetAccxRt_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetAccxRt_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ACCX_RT__BUILDER_HPP_
