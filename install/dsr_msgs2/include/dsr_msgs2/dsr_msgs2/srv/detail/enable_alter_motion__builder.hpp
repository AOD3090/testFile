// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/EnableAlterMotion.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/enable_alter_motion__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_EnableAlterMotion_Request_limit_dpos_per
{
public:
  explicit Init_EnableAlterMotion_Request_limit_dpos_per(::dsr_msgs2::srv::EnableAlterMotion_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::EnableAlterMotion_Request limit_dpos_per(::dsr_msgs2::srv::EnableAlterMotion_Request::_limit_dpos_per_type arg)
  {
    msg_.limit_dpos_per = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Request msg_;
};

class Init_EnableAlterMotion_Request_limit_dpos
{
public:
  explicit Init_EnableAlterMotion_Request_limit_dpos(::dsr_msgs2::srv::EnableAlterMotion_Request & msg)
  : msg_(msg)
  {}
  Init_EnableAlterMotion_Request_limit_dpos_per limit_dpos(::dsr_msgs2::srv::EnableAlterMotion_Request::_limit_dpos_type arg)
  {
    msg_.limit_dpos = std::move(arg);
    return Init_EnableAlterMotion_Request_limit_dpos_per(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Request msg_;
};

class Init_EnableAlterMotion_Request_ref
{
public:
  explicit Init_EnableAlterMotion_Request_ref(::dsr_msgs2::srv::EnableAlterMotion_Request & msg)
  : msg_(msg)
  {}
  Init_EnableAlterMotion_Request_limit_dpos ref(::dsr_msgs2::srv::EnableAlterMotion_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_EnableAlterMotion_Request_limit_dpos(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Request msg_;
};

class Init_EnableAlterMotion_Request_mode
{
public:
  explicit Init_EnableAlterMotion_Request_mode(::dsr_msgs2::srv::EnableAlterMotion_Request & msg)
  : msg_(msg)
  {}
  Init_EnableAlterMotion_Request_ref mode(::dsr_msgs2::srv::EnableAlterMotion_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_EnableAlterMotion_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Request msg_;
};

class Init_EnableAlterMotion_Request_n
{
public:
  Init_EnableAlterMotion_Request_n()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EnableAlterMotion_Request_mode n(::dsr_msgs2::srv::EnableAlterMotion_Request::_n_type arg)
  {
    msg_.n = std::move(arg);
    return Init_EnableAlterMotion_Request_mode(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::EnableAlterMotion_Request>()
{
  return dsr_msgs2::srv::builder::Init_EnableAlterMotion_Request_n();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_EnableAlterMotion_Response_success
{
public:
  Init_EnableAlterMotion_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::EnableAlterMotion_Response success(::dsr_msgs2::srv::EnableAlterMotion_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::EnableAlterMotion_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::EnableAlterMotion_Response>()
{
  return dsr_msgs2::srv::builder::Init_EnableAlterMotion_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__BUILDER_HPP_
