// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetDesiredForce.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_desired_force__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetDesiredForce_Request_mod
{
public:
  explicit Init_SetDesiredForce_Request_mod(::dsr_msgs2::srv::SetDesiredForce_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetDesiredForce_Request mod(::dsr_msgs2::srv::SetDesiredForce_Request::_mod_type arg)
  {
    msg_.mod = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Request msg_;
};

class Init_SetDesiredForce_Request_time
{
public:
  explicit Init_SetDesiredForce_Request_time(::dsr_msgs2::srv::SetDesiredForce_Request & msg)
  : msg_(msg)
  {}
  Init_SetDesiredForce_Request_mod time(::dsr_msgs2::srv::SetDesiredForce_Request::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_SetDesiredForce_Request_mod(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Request msg_;
};

class Init_SetDesiredForce_Request_ref
{
public:
  explicit Init_SetDesiredForce_Request_ref(::dsr_msgs2::srv::SetDesiredForce_Request & msg)
  : msg_(msg)
  {}
  Init_SetDesiredForce_Request_time ref(::dsr_msgs2::srv::SetDesiredForce_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_SetDesiredForce_Request_time(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Request msg_;
};

class Init_SetDesiredForce_Request_dir
{
public:
  explicit Init_SetDesiredForce_Request_dir(::dsr_msgs2::srv::SetDesiredForce_Request & msg)
  : msg_(msg)
  {}
  Init_SetDesiredForce_Request_ref dir(::dsr_msgs2::srv::SetDesiredForce_Request::_dir_type arg)
  {
    msg_.dir = std::move(arg);
    return Init_SetDesiredForce_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Request msg_;
};

class Init_SetDesiredForce_Request_fd
{
public:
  Init_SetDesiredForce_Request_fd()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetDesiredForce_Request_dir fd(::dsr_msgs2::srv::SetDesiredForce_Request::_fd_type arg)
  {
    msg_.fd = std::move(arg);
    return Init_SetDesiredForce_Request_dir(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetDesiredForce_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetDesiredForce_Request_fd();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetDesiredForce_Response_success
{
public:
  Init_SetDesiredForce_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetDesiredForce_Response success(::dsr_msgs2::srv::SetDesiredForce_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetDesiredForce_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetDesiredForce_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetDesiredForce_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_DESIRED_FORCE__BUILDER_HPP_
