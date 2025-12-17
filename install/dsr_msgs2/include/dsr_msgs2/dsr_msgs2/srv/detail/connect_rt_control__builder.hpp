// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ConnectRtControl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/connect_rt_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConnectRtControl_Request_port
{
public:
  explicit Init_ConnectRtControl_Request_port(::dsr_msgs2::srv::ConnectRtControl_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::ConnectRtControl_Request port(::dsr_msgs2::srv::ConnectRtControl_Request::_port_type arg)
  {
    msg_.port = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConnectRtControl_Request msg_;
};

class Init_ConnectRtControl_Request_ip_address
{
public:
  Init_ConnectRtControl_Request_ip_address()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConnectRtControl_Request_port ip_address(::dsr_msgs2::srv::ConnectRtControl_Request::_ip_address_type arg)
  {
    msg_.ip_address = std::move(arg);
    return Init_ConnectRtControl_Request_port(msg_);
  }

private:
  ::dsr_msgs2::srv::ConnectRtControl_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConnectRtControl_Request>()
{
  return dsr_msgs2::srv::builder::Init_ConnectRtControl_Request_ip_address();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConnectRtControl_Response_success
{
public:
  Init_ConnectRtControl_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ConnectRtControl_Response success(::dsr_msgs2::srv::ConnectRtControl_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConnectRtControl_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConnectRtControl_Response>()
{
  return dsr_msgs2::srv::builder::Init_ConnectRtControl_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CONNECT_RT_CONTROL__BUILDER_HPP_
