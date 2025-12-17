// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetOrientationError.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_orientation_error__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetOrientationError_Request_axis
{
public:
  explicit Init_GetOrientationError_Request_axis(::dsr_msgs2::srv::GetOrientationError_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetOrientationError_Request axis(::dsr_msgs2::srv::GetOrientationError_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetOrientationError_Request msg_;
};

class Init_GetOrientationError_Request_xc
{
public:
  explicit Init_GetOrientationError_Request_xc(::dsr_msgs2::srv::GetOrientationError_Request & msg)
  : msg_(msg)
  {}
  Init_GetOrientationError_Request_axis xc(::dsr_msgs2::srv::GetOrientationError_Request::_xc_type arg)
  {
    msg_.xc = std::move(arg);
    return Init_GetOrientationError_Request_axis(msg_);
  }

private:
  ::dsr_msgs2::srv::GetOrientationError_Request msg_;
};

class Init_GetOrientationError_Request_xd
{
public:
  Init_GetOrientationError_Request_xd()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetOrientationError_Request_xc xd(::dsr_msgs2::srv::GetOrientationError_Request::_xd_type arg)
  {
    msg_.xd = std::move(arg);
    return Init_GetOrientationError_Request_xc(msg_);
  }

private:
  ::dsr_msgs2::srv::GetOrientationError_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetOrientationError_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetOrientationError_Request_xd();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetOrientationError_Response_success
{
public:
  explicit Init_GetOrientationError_Response_success(::dsr_msgs2::srv::GetOrientationError_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetOrientationError_Response success(::dsr_msgs2::srv::GetOrientationError_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetOrientationError_Response msg_;
};

class Init_GetOrientationError_Response_ori_error
{
public:
  Init_GetOrientationError_Response_ori_error()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetOrientationError_Response_success ori_error(::dsr_msgs2::srv::GetOrientationError_Response::_ori_error_type arg)
  {
    msg_.ori_error = std::move(arg);
    return Init_GetOrientationError_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetOrientationError_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetOrientationError_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetOrientationError_Response_ori_error();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__BUILDER_HPP_
