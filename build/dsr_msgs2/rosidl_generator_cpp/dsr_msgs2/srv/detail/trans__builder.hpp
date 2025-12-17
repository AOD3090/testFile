// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/Trans.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TRANS__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__TRANS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/trans__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Trans_Request_ref_out
{
public:
  explicit Init_Trans_Request_ref_out(::dsr_msgs2::srv::Trans_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::Trans_Request ref_out(::dsr_msgs2::srv::Trans_Request::_ref_out_type arg)
  {
    msg_.ref_out = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Request msg_;
};

class Init_Trans_Request_ref
{
public:
  explicit Init_Trans_Request_ref(::dsr_msgs2::srv::Trans_Request & msg)
  : msg_(msg)
  {}
  Init_Trans_Request_ref_out ref(::dsr_msgs2::srv::Trans_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_Trans_Request_ref_out(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Request msg_;
};

class Init_Trans_Request_delta
{
public:
  explicit Init_Trans_Request_delta(::dsr_msgs2::srv::Trans_Request & msg)
  : msg_(msg)
  {}
  Init_Trans_Request_ref delta(::dsr_msgs2::srv::Trans_Request::_delta_type arg)
  {
    msg_.delta = std::move(arg);
    return Init_Trans_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Request msg_;
};

class Init_Trans_Request_pos
{
public:
  Init_Trans_Request_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Trans_Request_delta pos(::dsr_msgs2::srv::Trans_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_Trans_Request_delta(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Trans_Request>()
{
  return dsr_msgs2::srv::builder::Init_Trans_Request_pos();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Trans_Response_success
{
public:
  explicit Init_Trans_Response_success(::dsr_msgs2::srv::Trans_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::Trans_Response success(::dsr_msgs2::srv::Trans_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Response msg_;
};

class Init_Trans_Response_trans_pos
{
public:
  Init_Trans_Response_trans_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Trans_Response_success trans_pos(::dsr_msgs2::srv::Trans_Response::_trans_pos_type arg)
  {
    msg_.trans_pos = std::move(arg);
    return Init_Trans_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::Trans_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Trans_Response>()
{
  return dsr_msgs2::srv::builder::Init_Trans_Response_trans_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__TRANS__BUILDER_HPP_
